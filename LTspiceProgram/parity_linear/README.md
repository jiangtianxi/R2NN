# Linear temporal parity with a 5 x 5 TL082 network

## Classification

Open `parity_linear.asc` in LTspice, or run `parity_linear.net` in batch mode from this directory. Keep the local `TL082.lib` and the `inputs` directory alongside either entry point.

Source I1 injects one waveform. The default is `inputs/test_400/class1/sample1_1_101.txt` (even parity). Change the PWL filename on I1 to select another input. Each filename identifies its true class, sample number and three-bit pattern. Class 1 is even and class 2 is odd. Each directory has 200 waveforms (50 per bit pattern). Inputs use 40 Hz for bit zero and 62 Hz for bit one, with continuous phase across the three segments. The two PWL columns are time in seconds and injected current in amperes.

Run the 3 s transient. The stimulus occupies 2.0 to 2.6 s. Save the two probe voltages and compare the sums of squared voltages over the 0.6 s stimulus window. The first probe is `V(out11)` (even), and the second probe is `V(out15)` (odd). The larger integral selects the class.

## Provenance

The packaged circuit derives from `R2NN55ForParityCheck_Linear_LTspice/responseSimulationSampleTL082/_pyltspice_work/R2NN55TestWithoutCellTL082_batch.asc` and its generated text netlist. The original batch run sets a 3 s stop time, a 0.1 ms maximum time step, zero initial conditions via UIC, and compression disabled. The portable entries reproduce these settings and use relative filenames.

All 400 input files were copied from the project directory `R2NN55ForParityCheck_Linear_LTspice/inputSample/1e-6`. The archived run manifest records a successful simulation for each of these filenames. At the input node their amplitudes equal the 400 generator-source test waveforms multiplied by 1e-6.

## Probe frequency response

Use the [TL082 sweep netlist](parity_linear_sweep.net) or [schematic](parity_linear_sweep.asc) to excite the input at `in3` with a 0.25 µA peak chirp from 1 to 100 Hz over 100 s. The circuit saves the input current and both probe voltages. This entry uses the project's sweep configuration instead of a classification waveform. [Other amplifier configurations](opamp_variants/) have their own instructions and supply settings.

## Op-amp comparisons

[ADA4622 and AD8642 sweep entries](opamp_variants/README.md) use the case-specific supply rails and retain the original linear network parameters.
