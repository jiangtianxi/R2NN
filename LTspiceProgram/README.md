# LTspice circuits for the resonant recurrent neural network

Each task directory contains a circuit, its input waveforms or analytic excitation, and the amplifier models required by the text netlist. The links below identify the entry points. Keep the directory structure when moving the package.

The input waveforms are archived on Zenodo (DOI: [10.5281/zenodo.23998285](https://doi.org/10.5281/zenodo.23998285)) as `R2NN_LTspice_inputs.zip`. Unzip it in the repository root before running a classification example.

| Paper result | Circuit directory | Where to start |
|---|---|---|
| Fig. 2, linear temporal parity | [5 × 5 linear network](parity_linear/README.md) | Use the [TL082 classification circuit](parity_linear/parity_linear.net) with the 40/62 Hz phase-accumulating input. |
| Nonlinear temporal parity and circuit correspondence | [7 × 7 nonlinear network](parity_nonlinear/README.md) | Use the [nominal 4 MHz circuit](parity_nonlinear/R2NN77TestWithoutCell.net) with the 40/60 Hz phase-neutral input. The TL082 comparison model is a separate simulation. |
| Figs. 3 and 4, synthetic pulses and frequency-coded tactile textures | [Shared pulse and tactile network](pulse_tactile/README.md) | [Classification](pulse_tactile/pulse_tactile.net) or [frequency sweep](pulse_tactile/pulse_tactile_sweep.net). The tactile device reuses the trained pulse network. The example stimuli in this folder are synthetic pulses. |
| Fig. 5, single-speaker voice commands | [Speech network](speech/README.md) | [Classification](speech/speech.net) or [frequency sweep](speech/speech_sweep.net). |
| Fig. 6, drone conditions | [Drone network](drone/README.md) | [Classification](drone/drone.net) or [frequency sweep](drone/drone_sweep.net). |
| Supplementary Note 8, analog baselines | [Parity baselines](baselines_parity/README.md) | Two band-pass filters, two isolated resonators, and a nine-filter bank with its fitted readout. |
| Supplementary Note 6, component tolerances and amplifier bandwidth | [Parity tolerance and gain bandwidth circuits](parity_tolerances/README.md) | Representative Monte Carlo configurations and gain bandwidth settings from 1 to 10 MHz for both network sizes. The local index identifies archived and regenerated configurations. |
| Supplementary Note 3, unit-cell dynamics | [Unit-cell circuits](unit_cells/README.md) | Damping and nonlinear sweeps, with separate amplifier libraries and a damping case index. |

## Running an example

1. With LTspice installed, open the task's `.net` or `.cir` file and run it. If using batch mode, start in the directory containing the selected file. For example, from `pulse_tactile` run `LTspice.exe -b pulse_tactile.net` if the executable is on your PATH.
2. The packaged classification examples have one input selected. To test another waveform, change the input source's `PWL file` path to the corresponding file in the same task's input collection. The local README identifies the input interval and output nodes.
3. Inspect the output probe voltages over the specified decision window. For the R²NN, compare the integrated squared voltage at each probe and select the largest. For the filter-bank baseline, apply the supplied readout weights as described in its README.

The three application networks include editable `.asc` schematics as well as text netlists. A `.raw` file written by LTspice is simulation output. The package contains circuit stimuli and model dependencies; the experimental sensor-record archives are maintained separately.

## Verification scope

Representative 3 s classification runs completed in LTspice for the pulse/tactile, speech and drone circuits, the linear parity circuit, both nonlinear parity amplifier models, and all three analog parity baselines. The pulse, speech and drone examples each selected probe 1 for the default class-1 input. Selected tolerance, amplifier and unit-cell cases were also checked. These checks exercised the released entry points; they did not repeat the full classification or sweep studies. Consult the task READMEs and case indices for the source of each circuit and for configurations reconstructed from archived templates.
