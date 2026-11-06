# Analog baselines for temporal parity

These are the three analog baselines in Supplementary Note 8. Each directory contains a transient-analysis schematic (`.asc`) and its equivalent text netlist (`.net`). Open either in LTspice for Windows and run the transient analysis. Keep the sibling `models` and `stimuli` directories in place.

| Directory | Circuit and output nodes | Original simulation archive |
|---|---|---|
| `bpf2` | Two TL082 filters, 40 and 62 Hz, common nominal bandwidth 10 Hz. Save `V(out1)` and `V(out2)`. | `BPF2-spectral/responseSample_BPFBaseline_AN600_40_62_input1_BW10` |
| `resonators2` | Two isolated TL082-based R²NN unit cells. The bleed resistance is 2 MΩ on each input node. The internal resistances are 3,521,209 Ω and 860,931 Ω. Save `V(out1)` and `V(out2)`. | `PRB2-spectral/responseSample_PRBBaseline_AN600_40_62_input1e-7` |
| `bpf9` | Nine TL082 filters centered at 30, 35, …, 70 Hz with common nominal bandwidth 1 Hz. The outputs are `V(out1)` through `V(out9)` in ascending frequency. | `BPF9-spectral/responseSample_BPFsBaseline_AN600_40_62_input1_BW1` |

All three examples start with `sample1_1_101.txt`, from the even class. The input is active during 2.000–2.599 s. The simulation finishes at 3 s and uses a maximum time step of 0.1 ms. The circuit values and topologies come from the archived simulation pairs. The portable versions reference the bundled TL082 subcircuit and input files and use the transient/output settings of the batch analysis.

## Test inputs and readout

The `stimuli/voltage` directory contains the 400 phase-accumulating 40/62 Hz parity waveforms used by the voltage-driven filters. The original input files for the two filter banks are byte-identical and are stored once here. The `stimuli/current` directory contains the corresponding 400 input files for the isolated resonators, with the original current scaling. Class 1 is even, class 2 is odd. Each filename includes the source class, index and bit pattern. To run another waveform, change the filename on the voltage source or on **both** current sources of the resonator pair.

For each output, linearly interpolate its voltage at 600 times spaced 1 ms apart from 2.000 to 2.599 s, then sum the squared values. For `bpf2` and `resonators2`, a larger energy at LTspice node `out1` selects the odd class and a larger energy at `out2` selects the even class. Use the node names here when comparing with the figures, whose channels are presented in class order. For `bpf9`, multiply the nine energies by each row of `bpf9_linear_weights.csv` and select the class with the higher score. The weights act on raw energies, with zero bias. The MATLAB model is also included.

As a check on the default example, the two accumulated energies are approximately 102.42 and 157.08 V² for `bpf2`, and 20.12 and 22.19 V² for `resonators2`. Both give the even class. The two nine-filter scores are approximately −0.57914 and +0.57914. This particular waveform is misclassified by the nine-filter baseline, as in the archived simulation.
