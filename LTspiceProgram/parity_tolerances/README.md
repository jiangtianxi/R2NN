# Parity-network component tolerance and amplifier bandwidth

These LTspice circuits correspond to the network-level tolerance and gain–bandwidth product (GBW) experiments in Supplementary Note 6, Part 3. The 5 × 5 linear circuits use the 40/62 Hz phase-accumulating encoding. The 7 × 7 nonlinear circuits use the 40/60 Hz phase-neutral encoding. The [case index](cases.csv) identifies each circuit's source and parameter settings.

## Running a circuit

Open a `.net` in LTspice and run it from its own directory. Keep the package structure intact so that the relative model and waveform paths resolve. Each example contains one input current waveform and runs for 3 s, with a maximum time step of 0.1 ms. The two example waveforms are `101` for the 5 × 5 network and `011` for the 7 × 7 network. To apply another waveform, change the PWL filename on I1 and keep the corresponding time and current scaling.

The recorded voltage probes are `out11` and `out15` for the linear network and `out22` and `out28` for the nonlinear network. The study computes the sum of squared probe voltages at 1 kHz from 2.000 through 2.599 s. The first probe corresponds to even parity and the second to odd parity. Each published board accuracy uses 400 waveforms.

## Component tolerances

- [5 × 5 circuits](linear_5x5/component_tolerances/) cover all passive elements together, GIC capacitors, GIC resistors, coupling resistors (`Rc`) and internal resistors (`Rn`). The accompanying `.net` files were generated from the project's saved Monte Carlo schematics. The templates retain the saved nominal values, tolerance parameters and board selectors. Individual 5 × 5 per-board run decks were unavailable in the inspected Monte Carlo folders, so these files are representative templates.
- [7 × 7 circuits](nonlinear_7x7/component_tolerances/) cover the same groups and separate the linear (`RNL`) and nonlinear (`RNNL`) terms of the internal branches. For each group the netlist and schematic come from the archived 1% case, board selector 1. The original analysis studied 0.1%, 0.5%, 1% and 5%, with an additional 10% condition for the coupling and nonlinear-term groups. Five or ten boards were evaluated at each setting. These six examples cover the perturbation groups; they are not the complete set of historical board instances.

The `*_tol` parameters express fractional deviations, such that 0.01 means ±1%. A given integer `*_sample` selects a reproducible set of component perturbations. Change the tolerance and board selector to examine another circuit instance, keeping the same selector across all inputs for one board.

## Amplifier bandwidth

Both networks include one netlist per GBW setting from 1 to 10 MHz. The circuits use the parameterized UniversalOpAmp4 model for this sweep. These simulations vary amplifier bandwidth; the TL082 model in the linear component-tolerance circuits is used for a different experiment.

- [5 × 5 GBW](linear_5x5/amplifier_gbw/). Each netlist was taken from its corresponding archived GBW run.
- [7 × 7 GBW](nonlinear_7x7/amplifier_gbw/). The archive contains responses at all ten bandwidths but does not retain the run netlists. These decks were regenerated from the project's nominal 7 × 7 schematic and set to the indicated `GBW_val`. The 4 MHz schematic is included.

Independent 3 s checks were run on both 4 MHz bandwidth examples, the linear `Rn` tolerance template and the nonlinear `ALL` tolerance example. For the two GBW cases and nonlinear `ALL`, the reproduced two-probe waveforms agreed with the corresponding archived responses to within 5 × 10⁻⁹ relative RMS over the 600 decision samples. This check used one waveform per circuit.
