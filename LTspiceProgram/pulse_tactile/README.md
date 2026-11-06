# Pulse and frequency-coded tactile network

This is the trained 5 x 5 pulse network. The tactile demonstration uses the same circuit. Class 1, 2 and 3 contain 30, 50 and 70 Hz pulses, respectively. The supplied waveforms are synthetic pulse inputs.

## Running the example

Open `pulse_tactile.asc` in LTspice and run the transient analysis, or run `pulse_tactile.net` from this directory in batch mode. Keep the accompanying `TL082.lib` and `inputs` folders in place. The `.asc` and `.net` are two entry points for the same circuit.

The simulation runs for 3 s. Source I1 plays the selected waveform during the 2 s to 3 s decision window. Its input file has two columns, time in seconds and injected current in amperes. The three output traces are `V(out11)`, `V(out15)` and `V(out23)` in class order. To select another sample, replace the I1 PWL filename in the entry point you use.

## Included inputs

`inputs/hardware_60` contains the 60 synthetic signals used as input sources for the hardware experiment (20 per class). The default input is class 1, sample 1 from this collection. `inputs/test_1200` contains a separate set of 1,200 simulation inputs (400 per class). Both use a scale of 8e-7 from source sample amplitude to current in amperes. The archived batch manifest records 1,200 successful runs on the larger set.

## Experiment provenance

The circuit sources are the generated `R2NN55TestWithoutCellTL082_batch.asc` and `.net` in the original run directory `R2NN55ForGaussPulse_LTspice/responseExperimentSampleTL082/_pyltspice_work`. The portable versions point to local inputs and the accompanying TL082 model. Their transient duration, maximum step of 0.1 ms, saved traces and disabled waveform compression match the archived batch settings.

Source waveform directory in the project: `R2NN55ForGaussPulse_LTspice/inputSample/N1000/8e-7/experiment` and `R2NN55ForGaussPulse_LTspice/inputSample/N1000/8e-7/test`.

## Frequency sweep

Open `pulse_tactile_sweep.asc` or run `pulse_tactile_sweep.net` from this directory. These entries use a 1 microampere peak sinusoidal current injected at `in3`, swept linearly from 1 to 100 Hz in 100 s. They save `I(B1)` and the three probe voltages. The separate PWL test source sits outside the network and is set to zero.

The circuit and waveform come from the project's `R2NN55ForGaussPulse_LTspice/R2NN55TestWithoutCellTL082.asc`. The sweep frequency, duration and current agree with the archived source-current column in `R2NN55ForGaussPulse_LTspice/sweep1uA1to100hz100s100us_GaussPulse.txt`. The archive stores time-domain outputs. The Methods describe extracting the frequency-dependent amplitude along the short-time Fourier transform ridge.
