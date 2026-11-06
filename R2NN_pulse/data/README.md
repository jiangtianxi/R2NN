# Pulse dataset

The data files are not stored in this repository. Download `R2NN_datasets.zip` from Zenodo (DOI: [10.5281/zenodo.23998285](https://doi.org/10.5281/zenodo.23998285)) and unzip it in the repository root; the files below are then restored in place.

```text
data/N1000/
|-- train/sample{1,2,3}.mat        30 / 50 / 70 Hz, Sample: 1000 x 1000 per class
|-- test/sample{1,2,3}.mat         Sample: 1000 x 400 per class
`-- experiment/sample{1,2,3}.mat   Sample: 1000 x 20 per class
```

`N1000/GeneratePulse.m` regenerates all three splits.
