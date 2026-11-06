# Drone vibration dataset

The data files are not stored in this repository. Download `R2NN_datasets.zip` from Zenodo (DOI: [10.5281/zenodo.23998285](https://doi.org/10.5281/zenodo.23998285)) and unzip it in the repository root; the files below are then restored in place.

```text
data/
|-- drone_vib_raw_data/            Raw vibration records (one per class)
|   |-- 4full_1500_1545.txt        class 1
|   |-- 4full_1500_1438.txt        class 2
|   `-- 4full_1300_1335.txt        class 3
|-- train/sample{1,2,3}.mat        Sample: 10240 x 240 per class
`-- test/sample{1,2,3}.mat         Sample: 10240 x 60 per class
```

`build_drone_vibration_dataset.m` rebuilds `train/` and `test/` from the raw records. `dataset_metadata.mat` (kept in this repository) stores the segment positions.
