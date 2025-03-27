# Tutorial: Development and Deployment of SNNs on FPGA for Embedded Applications

This tutorial provides a comprehensive walkthrough of our toolchain, designed to streamline the development and deployment of Spiking Neural Networks (SNNs) onto our (unreleased) custom event-based neuromorphic accelerator hardware. Attendees will gain hands-on experience with model configuration, training, quantization, and hardware export, enabling them to optimize neural networks for real-world applications.

The tutorial is presented at the NICE Workshop 2025 in Heidelberg in a hands-on live session. A Docker container is provided that packages all dependencies:

[Docker Container Repository](https://hub.docker.com/r/nice2025tutorial/nice2025tutorial)

To use it, simply install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, Linux, MacOS) or the [Docker Engine](https://docs.docker.com/engine/install/) (Windows with WSL, Linux) and pull our container:
```bash
docker pull nice2025tutorial/nice2025tutorial:latest
```

Then navigate to the `docker` subdirectory in this repository and start the container:
```bash
docker compose up -d
```
To attach to the container using a shell, run:
```bash
docker exec -it docker-nice-1 bash
```

Once inside a container shell, start the Jupyter server by running:
```bash
./start_jupyter.sh
```
This script will print out a URL containing `127.0.0.1` (or localhost) with an access token. Copy that URL into a browser on your host system and run any of the notebooks in either the `mapping` or `toolchain` subdirectories.

**`toolchain`**: Overview of our end-to-end toolchain for training, quantizing and deploying SNNs onto our custom neuromorphic hardware accelerator

**`mapping`**: Sneak peek into our mapping framework for mapping SNNs to arbitrary NoC-based many-core neuromorphic accelerators.

Once you are done with using the Docker container, remove it from the host by running:
```bash
docker compose down
```

### License note
We are planning on releasing both the hardware sources and the software stack as open-source projects within 2025, but at this point in time all code and information shown in this tutorial is proprietary (see `LICENSE`). We thank you for your understanding.
