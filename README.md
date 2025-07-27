# FPGA_CNN_Hardware

## Project Overview
This repository contains the HDL source, simulation scripts, and documentation for a custom FPGA-based hardware accelerator targeting convolutional neural networks (CNNs) for real-time deep learning inference. The project focuses on designing, implementing, and benchmarking a VLSI neural network accelerator optimized for speed, energy efficiency, and accuracy.

## Goals
- Design and implement a parallel MAC datapath optimized for CNN inference workloads.
- Develop hardware-friendly activation function circuits.
- Validate the design using simulation and FPGA synthesis.
- Benchmark performance versus software-based inference.
- Document design trade-offs and analysis.

## Repository Structure
- `/src/` — HDL (Verilog/VHDL) source files for accelerator modules.
- `/sim/` — Testbenches and simulation scripts.
- `/python/` — Reference Python models or testing scripts.
- `/doc/` — Architecture diagrams, thesis drafts, and documentation.

## How to Use This Repository
1. Clone the repo:
   git clone https://github.com/your-username/FPGA_CNN_Hardware.git
   cd FPGA_CNN_Hardware

2. Add or modify HDL modules in `/src/`.

3. Run simulations with your chosen tool (ModelSim, Vivado) using testbenches from `/sim/`.

4. Use `/python/` to validate hardware results against reference CNN models.

5. Refer to `/doc/` for project documentation.



