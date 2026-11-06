# Analog Resonant Recurrent Neural Network (R<sup>2</sup>NN) via Metacircuit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23998285.svg)](https://doi.org/10.5281/zenodo.23998285)

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Tasks](#tasks)
- [Training protocol](#training-protocol)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Circuit simulations](#circuit-simulations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This repository accompanies our work on the resonant recurrent neural network (R<sup>2</sup>NN), an analog recurrent network built from coupled local resonators. It contains the training code, datasets, trained models and LTspice circuits used in the paper.

The R<sup>2</sup>NN is a two-dimensional lattice of unit cells. An input waveform drives one node of the lattice. The waves it excites propagate, interfere and decay, so the lattice itself carries the recurrent memory. Each class is assigned one output node, and the predicted class is the node with the largest integrated squared response.

The workflow has two stages:

1. **Digital training.** The lattice is modelled as a mass-spring system and integrated in time with a second-order central-difference scheme. The local-resonator stiffness `k_n` and the coupling stiffness `k_c` are trained by back-propagation through time.
2. **Analog realization.** The trained stiffnesses are mapped to circuit elements through the reformulated mechanical-electrical analogy. The resulting metacircuit is functionally equivalent to the trained network and is verified in LTspice and on hardware.

The code covers four tasks: synthetic pulses (also used for tactile texture recognition), voice commands, drone vibration conditions and three-bit temporal parity.

## Repository layout

| Path | Description |
|---|---|
| [`R2NN_pulse/`](R2NN_pulse) | Synthetic pulses (30/50/70 Hz), shared with the tactile task |
| [`R2NN_voice_command/`](R2NN_voice_command) | Single-speaker voice commands (hou / qian / ting) |
| [`R2NN_drone/`](R2NN_drone) | Drone vibration conditions (three classes) |
| [`R2NN_parity/`](R2NN_parity) | Three-bit temporal parity (linear and cubic-nonlinear) |
| [`LTspiceProgram/`](LTspiceProgram) | LTspice circuits for every task (see [`LTspiceProgram/README.md`](LTspiceProgram/README.md)) |
| `R2NN_5x5_PCB_Source_Data.xlsx` | Source data for the 5 x 5 PCB |
| `requirements.txt` | Python dependencies |
| `LICENSE` | License |

Each `R2NN_*` task folder has the same structure:

| Path | Description |
|---|---|
| `main.py` | Training script (cross-validation + final full-pool fit) |
| `metarnn/` | R2NN simulator (lattice assembly, time stepping, probes) |
| `data/` | Datasets (MAT files) and the MATLAB scripts that built them |
| `log/` | Training log of the reported model |
| `model/` | Trained model of the reported run (`*.pkl`, full PyTorch module) |
| `parameter_save/` | Trained `k_n` / `k_c` (and `k_cubic`) as text files |

## Tasks

| Folder | Task | Classes | Sequence length | Samples per class (train / test) | Lattice | Entry script |
|---|---|---|---|---|---|---|
| `R2NN_pulse` | Synthetic pulses | 3 (30, 50, 70 Hz) | 1000 (Fs = 1 kHz) | 1000 / 400, plus 20 in `experiment` | 5 x 5, linear | `main.py` |
| `R2NN_voice_command` | Voice commands | 3 (hou, qian, ting) | 10000 | 80 / 20 | 5 x 5, linear | `main.py` |
| `R2NN_drone` | Drone vibration | 3 | 10240 (1 s at 10.24 kHz) | 240 / 60 | 5 x 5, linear | `main.py` |
| `R2NN_parity` | 3-bit parity, 40/60 Hz | 2 (even, odd) | 600 (3 x 200) | 1000 / 400 | 7 x 7, cubic | `main.py` (data `N600`) |
| `R2NN_parity` | 3-bit parity, 40/62 Hz | 2 (even, odd) | 600 (3 x 200) | 1000 / 400 | 5 x 5, linear | `main-for-40-62.py` (data `AN600`) |

`R2NN_pulse/main_full_train.py` trains one model on the complete training pool for a fixed number of epochs, without cross-validation, and saves `k_n` / `k_c` after every epoch.

The lattice dynamics are selected with `model_mode` at the top of each script: `linear` for a purely linear lattice, or `cubic` for a fixed cubic restoring force in each local resonator.

## Training protocol

All scripts follow the same protocol:

1. The training split is divided by 5-fold stratified cross-validation (the parity task stratifies by bit pattern). In `R2NN_pulse` and `R2NN_parity`, 10% of each training fold is further held out for validation and the remaining fold is reported as the CV holdout.
2. In each fold, the checkpoint with the highest validation accuracy is kept (validation cross-entropy breaks ties). Early stopping monitors validation cross-entropy.
3. A final model is trained on the full training pool. Its epoch count is the rounded median of the best CV epochs, and its learning rate in each epoch is the median over the folds.
4. The independent test split is loaded and evaluated once, after the final model is fixed. Exact waveform overlap between splits is rejected.

The loss is cross-entropy on the log probe energies plus an optional energy-ratio margin term. After every optimizer step, the stiffnesses are projected to the admissible range and rescaled, if necessary, to keep the explicit time integration stable.

## Requirements

- Python 3.8
- PyTorch 1.11.0
- numpy 1.22.3, scipy 1.8.0, scikit-learn 1.0.2, matplotlib 3.5.1
- MATLAB (only to regenerate datasets; R2018b or later, with the Communications Toolbox for `awgn`)
- LTspice (only for the circuit simulations)

The reported models were trained on a server with an NVIDIA GeForce RTX 4090 GPU (24 GB), 10 vCPUs and 50 GB of RAM. The scripts use a CUDA GPU automatically when one is available and fall back to the CPU otherwise. All computations use float64.

## Installation

```bash
conda create -n r2nn python=3.8
conda activate r2nn
# Install the PyTorch 1.11.0 build for your platform (CPU or CUDA), see
# https://pytorch.org/get-started/previous-versions/#v1110
pip install -r requirements.txt
```

## Usage

Run each script from inside its task folder, because the outputs and the parity dataset path are relative to the working directory:

```bash
conda activate r2nn
cd R2NN_pulse
python main.py
```

Replace `R2NN_pulse` with `R2NN_voice_command`, `R2NN_drone` or `R2NN_parity`. For the 40/62 Hz parity network, run `python main-for-40-62.py` in `R2NN_parity`.

### Outputs

- `log/train_log_<mode>_double_<timestamp>.txt`: per-batch losses, learning rates and gradient norms; per-epoch train and validation metrics; per-class (or per-pattern) accuracy and confusion matrices; final independent-test results.
- `model/model_<mode>_double_<timestamp>_fold<k>_best.pkl` and `..._final.pkl`: the selected model of each fold and the final model.
- `parameter_save/parameter_kn_*.txt`, `parameter_kc_*.txt` (and `parameter_kcubic_*.txt` in cubic mode): the trained stiffnesses used to build the circuits.

> **Note:** The `log/`, `model/` and `parameter_save/` folders must exist before `main.py` is run.

### Run time

A complete run (five folds plus the final fit) on the server above took about 1.5 to 2 h for the pulse and parity tasks, and about 7 to 8 h for the drone and voice-command tasks.

### Loading a trained model

The `.pkl` files store the full PyTorch module, so the `metarnn` package must be importable:

```python
import torch
model = torch.load("model/model_linear_double_2026_08_13_16_36_final.pkl",
                   map_location="cpu")          # PyTorch >= 2.6: add weights_only=False
print(model.cell.coup.k_n, model.cell.coup.k_c)
```

## Datasets

The datasets, the LTspice input waveforms and additional training logs are archived on Zenodo ([10.5281/zenodo.23998285](https://doi.org/10.5281/zenodo.23998285)). Download `R2NN_datasets.zip`, `R2NN_LTspice_inputs.zip` and, optionally, `R2NN_parity_extra_logs.zip`, and unzip them in the repository root. The files are restored to the locations listed below.

| Task | Location | Content |
|---|---|---|
| Pulses | `R2NN_pulse/data/N1000/{train,test,experiment}` | Generated by `GeneratePulse.m`: 11-cycle tone bursts at 30, 50 or 70 Hz with additive white Gaussian noise (0 dB SNR) |
| Voice commands | `R2NN_voice_command/data/{train,test}` | Recorded commands "hou", "qian" and "ting" from a single speaker |
| Drone | `R2NN_drone/data/{train,test}` | Built by `build_drone_vibration_dataset.m` from one continuous vibration record per class (`drone_vib_raw_data/`), see below |
| Parity | `R2NN_parity/data/{N600,N900,N1200,AN600}` | Generated by `generate_balanced_3bit_parity_dataset*.m`, see `R2NN_parity/data/README.md` |

Each class is stored as one `sampleK.mat` file (`K` = class index starting at 1), with one waveform per column.

**Drone dataset.** Each class comes from one long vibration record sampled at 10.24 kHz. The record is cut into non-overlapping 1 s segments. The first 240 segments (the first 240 s of the record) form the training pool, and the last 60 segments (the final 60 s) form the independent test set. The middle part of the record is not used, so training and test samples are separated in time and never share data. The segment positions are stored in `data/dataset_metadata.mat`.

## Circuit simulations

`LTspiceProgram/` contains the LTspice netlists, schematics, input waveforms and amplifier models for every task, the analog baselines, the tolerance and gain-bandwidth studies, and the unit-cell characterisation. Start from `LTspiceProgram/README.md`. The input waveforms are in `R2NN_LTspice_inputs.zip` on Zenodo (see [Datasets](#datasets)).

## License

This project is released under the terms in [LICENSE](LICENSE).

## Acknowledgements

This code builds on [WaveTorch](https://github.com/fancompute/wavetorch). We use its framework for wave-propagation simulation and gradient-based optimization of physical systems, and extend it to R<sup>2</sup>NN dynamics. We thank its authors for making their work open source.
