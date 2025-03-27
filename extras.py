# ******************************************************************************************
# All Rights Reserved
# 
# Copyright (c) 2021-2025 FZI Forschungszentrum Informatik
# 
# THE CONTENTS OF THIS SOFTWARE ARE PROPRIETARY AND CONFIDENTIAL.
# 
# UNAUTHORIZED COPYING, TRANSFERRING OR REPRODUCTION OF THE CONTENTS OF THIS SOFTWARE, VIA
# ANY MEDIUM IS STRICTLY PROHIBITED.
# 
# The software is provided "AS IS", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a particular
# purpose and non-infringement.
# 
# In no event shall the authors or copyright holders be liable for any claim, damages or
# other liability, whether in an action of contract, tort or otherwise, arising from, out
# of or in connection with the software or the use or other dealings in the software.
# 
# The licensor shall never, and without any limit, be liable for any damage, cost, expense
# or any other payment incurred by the licensee as a result of the software's actions,
# failure, bugs and/or any other interaction between the software and the licensee's
# end-equipment, computers, other software or any 3rd party, end-equipment, computer or
# services.
# ******************************************************************************************

import subprocess
import os
import random
import sys
from tqdm import tqdm
import paramiko
from scp import SCPClient

import nir
import tonic
from qtorch.quant import Quantizer
from qtorch import FixedPoint
import torch
from torch.utils.data import random_split
import numpy as np
import yaml
import pytorch_lightning as pl

from ecs_train.config import Cfg, load_yaml
from ecs_train.model import Model
from ecs_train.training import initialize_experiment
from ecs_train.utils import export_nir
from ecs_deploy.network import Network, LogLevel, generate_input_events
from ecs_deploy.fixed_point import int_to_binary_str
from ecs_common.hardware_config import AcceleratorConfig, from_metadata
from ecs_common.quant_options import options_from_config

import logging


HOST = "192.168.100.1"
USER = "root"
PASSWORD = "nice2025tutorial"
LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/experiments")
REMOTE_PATH = "/root/nice2025tutorial"


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._original_log_level = logging.getLogger("pytorch_lightning").getEffectiveLevel()
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        logging.getLogger("pytorch_lightning").setLevel(self._original_log_level)

    def print(self, *values: object):
        print(*values, file=self._original_stdout)


def check_constraints(cfg: Cfg, accelerator_config: AcceleratorConfig):
    # check if only feed_forward layers
    if "feed_forward" not in cfg.model_cfg.network_type:
        raise Exception(
            f"Network type {cfg.model_cfg.network_type} is currently not supported by the accelerator.\n"
            "Please choose a network type that is built on linear layers (torch.nn.Linear)."
        )

    # check if no bias
    if "bias" in cfg.model_cfg.network_cfg.keys():
        if cfg.model_cfg.network_cfg["bias"]:
            raise Exception(f"Having bias in linear layers is currently not supported by the accelerator.")

    # check input dim
    merge_pol_fac = 2 - float(cfg.trainer_cfg.dataset_cfg.transform_cfg.merge_polarities)
    sensor_size = getattr(tonic.datasets, cfg.trainer_cfg.dataset_cfg.dataset).sensor_size
    spatial_fac = cfg.trainer_cfg.dataset_cfg.transform_cfg.spatial_factor
    input_dim = (sensor_size[0] // spatial_fac) * (sensor_size[1] // spatial_fac) * merge_pol_fac
    if input_dim > accelerator_config.neurons_per_core:
        raise Exception(
            f"Input dimension of {input_dim} is larger than maximal supported input size, which currently is {accelerator_config.neurons_per_core}.\n"
            "Please consider increasing spatial_factor to downsample the input."
        )

    # check max fan-in/fan-out
    hidden_features = cfg.model_cfg.network_cfg['hidden_features']
    max_fan_in = max(input_dim, hidden_features)
    max_fan_out = max(hidden_features, cfg.trainer_cfg.dataset_cfg.num_output_classes)
    prune_to_max_fan_in = False
    prune_to_max_fan_out = False
    if max_fan_in > accelerator_config.synapses_per_core:
        print(
            f"\nNeuron fan in of {max_fan_in} is larger than the maximal supported per-neuron fan in, which currently is {accelerator_config.synapses_per_core}.\n"
            f"This script will now resort to iteratively pruning the network in very small steps until every neuron has fan out of maximal {accelerator_config.synapses_per_core}.\n"
        )
        prune_to_max_fan_in = True
    if max_fan_out > accelerator_config.routes_per_core:
        print(
            f"\nNeuron fan in of {max_fan_out} is larger than the maximal supported per-neuron fan in, which currently is {accelerator_config.routes_per_core}.\n"
            f"This script will now resort to iteratively pruning the network in very small steps until every neuron has fan out of maximal {accelerator_config.routes_per_core}.\n"
        )
        prune_to_max_fan_out = True

    # check if num_weights is too high for available weight memory
    global_pruning_necessary = False
    output_dim = cfg.trainer_cfg.dataset_cfg.num_output_classes

    num_weights_hidden = input_dim * hidden_features
    num_weights_output = hidden_features * output_dim
    max_weights_per_core = accelerator_config.synapses_per_core

    if num_weights_hidden > max_weights_per_core:
        print(
            f"\nNumber of weights in hidden layer resulting from fan in of {input_dim} and layer size of {hidden_features} is too large: {num_weights_hidden}.\n"
            f"This script will now resort to global pruning to reach the maximum allowed number of neuron weights ({max_weights_per_core}).\n"
        )
        global_pruning_necessary = True
    if num_weights_output > max_weights_per_core:
        raise Exception(
            f"Number of weights of output layer is too large. The maximum supported number of weights is {max_weights_per_core}.\n"
            "Global pruning can only be done for hidden layers."
        )

    return prune_to_max_fan_in or prune_to_max_fan_out, global_pruning_necessary, num_weights_hidden, input_dim, hidden_features, output_dim


def quantize_weights(model: Model, quant_wl: int, quant_fl: int):
    # Check if weights are in the fixed point format's bounds
    integer_bits = quant_wl - quant_fl
    min_value = -2 ** (integer_bits - 1)
    max_value = 2 ** (integer_bits - 1) - 2 ** (-quant_fl)

    max_weight = float('-inf')
    min_weight = float('inf')

    for name, module in model.network.named_modules():
        if hasattr(module, 'weight'):
            max_weight = max(max_weight, module.weight.max())
            min_weight = min(min_weight, module.weight.min())

    if max_weight > max_value:
        print(
            f"Maximum weight value in network ({max_weight}) exceeds the representable fixed point range: {min_value} to {max_value}\n"
            "Network performance will suffer from clamping the weight values."
        )
    if min_weight < min_value:
        print(
            f"Minimum weight value in network ({min_weight}) exceeds the representable fixed point range: {min_value} to {max_value}\n"
            "Network performance will suffer from clamping the weight values."
        )

    # Quantize weights
    weight_quantizer = Quantizer(FixedPoint(wl=quant_wl, fl=quant_fl), forward_rounding="nearest")
    quantized_weights = model.state_dict()
    for name, param in quantized_weights.items():
        if "weight" in name:
            quantized_weights[name] = weight_quantizer(param)
    model.load_state_dict(quantized_weights)

    return model


def export_input_data(data_module, num_samples, output_path):
    gen = torch.Generator().manual_seed(hash("NICEWorkshop2025") % (2**32))
    test_set = data_module.test_dataloader().dataset
    os.makedirs(output_path, exist_ok=True)

    # Ensure num_samples is not greater than the total number of samples
    total_samples = len(test_set)
    if num_samples > total_samples:
        num_samples = total_samples

    # Choose random samples from test dataset and format
    samples, _ = random_split(test_set, [num_samples, total_samples - num_samples], gen)
    sample_loader = torch.utils.data.DataLoader(samples, batch_size=num_samples)
    sample_batch_data, sample_batch_targets = next(iter(sample_loader))
    sample_batch_data = sample_batch_data.permute([1, 0, 2, 3, 4])

    sample_info_dict = {}

    # Save samples to disk
    for idx in range(num_samples):
        sample_name = f"sample_{idx}"
        sample_info_dict[idx] = {
            "target": sample_batch_targets[idx].item(),
            "file": f"{sample_name}.txt"
        }

        sample_file = os.path.join(output_path, f"{sample_name}.npy")
        np.save(sample_file, sample_batch_data[:, idx].cpu().detach().numpy())

    sample_info_file = os.path.join(output_path, "sample_info.yaml")
    with open(sample_info_file, "w") as f:
        yaml.safe_dump(sample_info_dict, f)

    return sample_batch_data


def training_export(dataset_type: str, num_samples: int):
    # Load configuration
    config_path = f"config/{dataset_type}_feed_forward.yaml"
    config = load_yaml(config_path)
    config.trainer_cfg.output_path = f"./output/bulk/{dataset_type.upper()}"

    checkpoint_paths = [
        f"checkpoints/{dataset_type}.ckpt",
        f"checkpoints/{dataset_type}_30_pruning.ckpt"
    ]

    samples_output_dir = os.path.join(config.trainer_cfg.output_path, "test_samples")
    networks_output_dir = os.path.join(config.trainer_cfg.output_path, "networks")

    # Generate input sample data
    print(f"Exporting input data...")

    with SuppressOutput():
        _, _, data_module = initialize_experiment(config)
        os.makedirs(samples_output_dir, exist_ok=True)
        sample_batch_data = export_input_data(data_module, num_samples, samples_output_dir)

    print(f"Exporting networks...")

    progress_bar = tqdm(checkpoint_paths, f"{'':17}")
    for checkpoint_path in progress_bar:
        exp_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        progress_bar.set_description(f"{exp_name:17}")
        with SuppressOutput() as s:
            # Export NIR file
            config.trainer_cfg.checkpoint_path = checkpoint_path
            _, model, _ = initialize_experiment(config)

            experiment_output_dir = os.path.join(networks_output_dir, f"network_{exp_name}")
            network_output_file = os.path.join(experiment_output_dir, "network.nir")
            os.makedirs(experiment_output_dir, exist_ok=True)

            sample_data = next(iter(data_module.train_dataloader()))[0][0, 0:1, :]
            hardware_cfg = config.hardware_cfg
            quant_cfg = config.model_cfg.network_cfg["quant_cfg"]
            metadata = {"hardware_cfg": hardware_cfg, "quant_cfg": quant_cfg}
            export_nir(model.network, metadata, network_output_file, sample_data, dt=config.model_cfg.network_cfg["dt"], broadcast_params=False)

            # Generate network output data
            model.network.reset()
            model.eval()
            sample_batch_outputs = []

            for frame in sample_batch_data:
                sample_batch_output = model.network(frame)
                sample_batch_outputs.append(sample_batch_output.detach().numpy())

            sample_batch_outputs = np.stack(sample_batch_outputs)
            sample_outputs_dir = os.path.join(experiment_output_dir, "output_traces")
            os.makedirs(sample_outputs_dir, exist_ok=True)

            # Save samples to disk
            for sample in range(num_samples):
                file_output = os.path.join(sample_outputs_dir, f"neuron_state_trace_{sample}.npy")
                np.save(file_output, sample_batch_outputs[:, sample])


def deployment_export(dataset_type: str):
    # Load configuration
    config_path = f"config/{dataset_type}_feed_forward.yaml"
    config = load_yaml(config_path)
    config.trainer_cfg.output_path = f"./output/bulk/{dataset_type.upper()}"

    accelerator_config = from_metadata(
        config.hardware_cfg,
        config.model_cfg.network_cfg["quant_cfg"]
    )

    samples_output_dir = os.path.join(config.trainer_cfg.output_path, "test_samples")
    networks_output_dir = os.path.join(config.trainer_cfg.output_path, "networks")

    # Get all directories in the networks_output_dir
    network_dirs = [
        os.path.join(networks_output_dir, d) for d in os.listdir(networks_output_dir)
        if os.path.isdir(os.path.join(networks_output_dir, d))
    ]
    # Find all .npy file names in samples_output_dir
    sample_files = [
        os.path.join(samples_output_dir, f) for f in os.listdir(samples_output_dir)
        if os.path.isfile(os.path.join(samples_output_dir, f)) and f.endswith(".npy")
    ]

    # Convert all samples and write to disk
    print(f"\nSaving input traces...")
    input_events_list = []
    for sample_file in sorted(sample_files):
        input_sample = np.load(sample_file)
        input_events = generate_input_events(input_sample)
        input_events_list.append((len(input_sample), input_events))

        input_trace_file = sample_file.replace(".npy", ".txt")
        with open(input_trace_file, 'w') as f:
            for timestep, pre_neuron_id in input_events:
                f.write(f"{timestep - 1} {pre_neuron_id:0{accelerator_config.neuron_id_bits}b}\n")

    # Iterate over all networks
    for network_dir in network_dirs:
        print(f"\nStarting deployment for network {os.path.basename(network_dir)}...")
        # Load NIR graph and metadata
        nir_graph = nir.read(os.path.join(network_dir, "network.nir"))
        metadata = nir_graph.metadata
        quant_options = options_from_config(metadata["quant_cfg"])
        accelerator_config = from_metadata(metadata["hardware_cfg"], quant_options.weight_format)

        network = Network(nir_graph, accelerator_config, quant_options, log_level=LogLevel.WARNING)

        max_diffs = []
        mses = []

        # Iterate over all samples
        for idx, (input_length, input_events) in enumerate(tqdm(input_events_list, "Sample")):
            network.reset()

            # Simulate execution
            output_neuron_states = network.simulate(input_events, input_length + 3, progressbar=False)

            # Compare to Norse output
            train_output_file = os.path.join(network_dir, "output_traces", f"neuron_state_trace_{idx}.npy")
            train_output_np = np.load(train_output_file)
            sim_output_np = np.array([[state.to_float() for state in states_ts] for states_ts in output_neuron_states])
            # Calculate error between Norse and our simulation
            diff = sim_output_np[3:] - train_output_np
            mses.append(np.mean((diff) ** 2))
            max_diffs.append(np.max(np.abs(diff)))

            # Write neuron states to file
            neuron_state_trace_file = train_output_file.replace(".npy", ".txt")
            with open(neuron_state_trace_file, 'w') as f:
                for timestep, neuron_states_ts in enumerate(output_neuron_states[1:]):
                    for neuron_idx, neuron_state in enumerate(neuron_states_ts):
                        f.write(f"{timestep} {neuron_idx} {int_to_binary_str(neuron_state.value, quant_options.state_format.wl)}\n")

        # Printe average differences
        print(f"Average Maximum Difference = {np.mean(np.array(max_diffs))}")
        print(f"Average Mean Squared Error = {np.mean(np.array(mses))}\n")

        # Generate memory files
        network.generate_mem_files(network_dir, print_util=True)


def _extract_artifacts():
    source_directory = "output/bulk/"
    target_directory = "output/experiments/"

    try:
        subprocess.run(
            [
                "rsync",
                "-av",
                "--prune-empty-dirs",
                "--include", "*/",
                "--include", "*.txt",
                "--include", "**/sample_info.yaml",
                "--exclude", "*",
                source_directory,
                target_directory
            ],
            check=True
        )
        print("Files successfully copied.")
    except subprocess.CalledProcessError as e:
        print(f"Error during rsync execution: {e}")


def _recursive_copy(host, username, password, local_path, remote_path):
    """
    Recursively copies a directory to a remote host via SSH

    Args:
        host: Remote IP address (str)
        username: SSH username (str)
        password: SSH password (str)
        local_path: Local directory path (str)
        remote_path: Remote target path (str)
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(host, username=username, password=password)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path, recursive=True)
    finally:
        ssh.close()

def transfer_files():
    _extract_artifacts()
    _recursive_copy(HOST, USER, PASSWORD, LOCAL_PATH, REMOTE_PATH)
