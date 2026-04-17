# Fully Analog Resonant Recurrent Neural Network (R<sup>2</sup>NN)

## Overview

This project includes the R<sup>2</sup>NN training pipeline.

The metacircuit-based R<sup>2</sup>NN is realized by first training a R<sup>2</sup>NN on a designated dataset using a digital training framework to achieve accurate classification of signal categories. Once the iteration meets the predefined criterion, the parameters in the R<sup>2</sup>NN are then converted into corresponding metacircuit elements using the reformulated mechanical-electrical analogy. This parameter transformation yields a functionally equivalent metacircuit-based R<sup>2</sup>NN.

## System Requirements

**Hardware Resources:** The training and testing processes are conducted on a remote host with an Intel(R) Xeon(R) CPU Platinum 8260L at 2.30 GHz and 40 GB of RAM.

**Dependencies:**

* Python 3.8
* PyTorch 1.11.0
* matplotlib==3.5.1
* numpy==1.22.3
* scikit-learn==1.0.2
* scipy==1.8.0

## Usage

**Steps:**

1. Set up the required environment (e.g., `env`) and install the listed dependencies.
2. Clone the repository to your local machine or a custom directory (e.g., `/customfile`).
3. Run the following commands in your terminal:

```bash
conda activate env
cd /customfile/R2NN/circuitRNN_pulse/
python main.py
```

## Demo

* **Instructions to run on data:** Follow the steps outlined in the **Usage** section.
* **Expected output:** The script will output classification metrics in `/log/train_log_date.txt`. The optimized parameters are saved in the `/parameter_save/` directory, and the trained model is saved after each epoch in the `/model/` directory.
* **Expected run time:** A standard demo run on the specified remote host takes approximately a few hours.

## Acknowledgements

We would like to express our sincere gratitude to the authors of the **[WaveTorch](https://github.com/fancompute/wavetorch.git)** project. Our codebase is developed based on the pioneering work of `wavetorch`. We have utilized their foundational framework for wave propagation simulation and gradient-based optimization in physical systems, extending it to support R<sup>2</sup>NN dynamics. This project would not have been possible without their open-source contribution to the community.
