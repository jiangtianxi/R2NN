# Single-speaker voice-command network

This is the 5 x 5 network used for the three-command circuit simulation. Class 1, 2 and 3 correspond to Hou, Qian and Ting.

## Running the example

Open `speech.asc` in LTspice and run the transient analysis, or run `speech.net` from this directory in batch mode. Keep the accompanying `TL082.lib` and `inputs` folders in place. The `.asc` and `.net` are two entry points for the same circuit.

The simulation runs for 3 s. Source I1 plays the selected waveform during the 2 s to 3 s decision window. Its input file has two columns, time in seconds and injected current in amperes. The three output traces are `V(out11)`, `V(out15)` and `V(out23)` in class order. To select another sample, replace the I1 PWL filename in the entry point you use.

## Included inputs

`inputs/test_60` contains the 60 held-out utterances prepared for hardware playback (20 per class). The default input is class 1, sample 1. Each input current uses a scale of 2e-5 from the processed voice sample amplitude to amperes. The archived batch manifest records 60 successful simulations.

## Experiment provenance

The circuit sources are the generated `R2NN55TestWithoutCellTL082_batch.asc` and `.net` in the original run directory `R2NN55ForSpeechCommand_LTspice/responseExperimentSampleTL082_input2e-5/_pyltspice_work`. The portable versions point to local inputs and the accompanying TL082 model. Their transient duration, maximum step of 0.1 ms, saved traces and disabled waveform compression match the archived batch settings.

Source waveform directory in the project: `R2NN55ForSpeechCommand_LTspice/inputSample/2e-5`.

## Frequency sweep

Open `speech_sweep.asc` or run `speech_sweep.net` from this directory. These entries use a 1 microampere peak sinusoidal current injected at `in3`, swept linearly from 1 to 300 Hz in 300 s. They save `I(B1)` and the three probe voltages. The separate PWL test source sits outside the network and is set to zero.

The circuit and waveform come from the project's `R2NN55ForSpeechCommand_LTspice/R2NN55TestWithoutCellTL082.asc`. The sweep frequency, duration and current agree with the archived source-current column in `R2NN55ForSpeechCommand_LTspice/sweep1uA1to300hz300s100us_SpeechCommand.txt`. The archived schematic was saved with the PWL classification source connected. The sweep entries reconnect the existing chirp source B1 to the network input and set the run time to 300 s. The archive stores time-domain outputs. The Methods describe extracting the frequency-dependent amplitude along the short-time Fourier transform ridge.
