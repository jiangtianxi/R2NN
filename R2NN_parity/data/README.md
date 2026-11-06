# Balanced 3-Bit Sequential Parity Dataset

The `sample*.mat` files are not stored in this repository. Download `R2NN_datasets.zip` from Zenodo (DOI: [10.5281/zenodo.23998285](https://doi.org/10.5281/zenodo.23998285)) and unzip it in the repository root. The generators and the `dataset_config.mat` files are included here.

This directory contains balanced datasets for a three-bit sequential parity
classification task. Each bit is represented by a noisy sinusoidal segment,
with continuous phase across the three segments:

- `0` is represented by 40 Hz.
- `1` is represented by 60 Hz (62 Hz in `AN600`, see below).
- `sample1.mat` contains even-parity sequences with label `0`.
- `sample2.mat` contains odd-parity sequences with label `1`.

## Directory layout

Datasets with different sequence lengths are stored in separate subdirectories
so that generating a new dataset does not overwrite an existing one:

```text
data/
|-- generate_balanced_3bit_parity_dataset.m      (default N = 1200)
|-- generate_balanced_3bit_parity_dataset_en.m   (default N = 900)
|-- README.md
|-- N600/      40/60 Hz, N = 600   -> used by main.py (cubic, 7 x 7)
|-- N900/      40/60 Hz, N = 900
|-- N1200/     40/60 Hz, N = 1200
`-- AN600/     40/62 Hz, N = 600   -> used by main-for-40-62.py (linear, 5 x 5)
```

Each dataset directory has the same structure:

```text
Nxxx/
|-- dataset_config.mat
|-- train/
|   |-- sample1.mat
|   `-- sample2.mat
`-- test/
    |-- sample1.mat
    `-- sample2.mat
```

The output directory is selected automatically from the sequence length `N`.
For example, `N = 900` writes to `data/N900`.

## Generation settings

| Parameter | Value |
|---|---:|
| Sampling rate (`Fs`) | 1000 Hz |
| Sequence length (`N`) | 600, 900 or 1200 points |
| Number of bit segments (`L`) | 3 |
| Bit frequencies | 40 / 60 Hz (40 / 62 Hz in `AN600`) |
| Noise standard deviation | 0.15 |
| Training samples per class | 1000 |
| Test samples per class | 400 |
| Training RNG seed | 20260731 |
| Test RNG seed | 20260801 |

With `Fs = 1000` Hz, `N600` lasts 0.6 s (200 points per bit), `N900` lasts
0.9 s (300 points per bit) and `N1200` lasts 1.2 s (400 points per bit).

`AN600` uses the same settings as `N600` except that bit `1` is 62 Hz
(`bit_frequencies_hz = [40, 62]` in its `dataset_config.mat`). The generators
in this folder produce 40/60 Hz; to generate 40/62 Hz data, change
`frequencies = 40 + 20 * bits` to `frequencies = 40 + 22 * bits` and the
`bit_frequencies_hz` entry to `[40, 62]`.

## Class balance

Each class contains four bit patterns in equal numbers:

- Even parity, label `0`: `000`, `011`, `101`, and `110`.
- Odd parity, label `1`: `001`, `010`, `100`, and `111`.

The training and independent test splits use different fixed random seeds.
The generator verifies class balance, pattern balance, array dimensions, parity
labels, and the absence of exactly duplicated waveforms between the two splits.

## MAT-file format

Every `sample*.mat` file contains:

- `data`: waveform matrix with shape `N x num_samples`. Each column is one
  waveform.
- `bit_sequences`: integer matrix with shape `num_samples x 3`. Each row gives
  the three bits associated with the corresponding waveform column.
- `class_label`: scalar class label, either `0` or `1`.
- `metadata`: split name, random seed, class label, sample count, and counts for
  all bit patterns from `000` through `111`.

Each `dataset_config.mat` file stores the generation configuration, including
`Fs`, `N`, the number of segments, points per segment, frequencies, noise
level, sample counts, and random seeds.

## Generating a dataset

From the `R2NN_parity` folder, run in MATLAB:

```matlab
run('data/generate_balanced_3bit_parity_dataset.m')
```

To select a sequence length, edit `N` near the beginning of the script:

```matlab
N = 600;  % writes to data/N600
```

`N` must be divisible by `L`. The number of samples per class must be divisible
by four because every parity class contains four bit patterns. Running the
generator again with the same `N` overwrites the MAT files in that dataset's
own `Nxxx` directory, but it does not modify datasets with other lengths.

## Selecting a dataset for training

Set the expected length in `main.py` (or `main-for-40-62.py`); the dataset path
is built from that value:

```python
EXPECTED_SEQUENCE_LENGTH = 600
dataset_root_candidates = (
    Path("./data") / f"N{EXPECTED_SEQUENCE_LENGTH}",   # "AN..." in main-for-40-62.py
)
```

Run the scripts from the `R2NN_parity` folder, because this path is relative to
the working directory.
