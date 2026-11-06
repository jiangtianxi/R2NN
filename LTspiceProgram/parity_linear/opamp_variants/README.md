# Op-amp comparisons in the 5 x 5 linear parity network

These entries compare the small-signal response of the same trained circuit with two op-amp types. They are separate from the TL082 classification example in the parent directory and from the measured 400-waveform accuracy tests.

| Op-amp | Supply rails | Sweep input and output | Entry |
|---|---|---|---|
| ADA4622 | +15 V and -15 V | 0.25 µA AC at the trained input node, 1 to 100 Hz | [AC netlist](ADA4622/ada4622_ac.net) · [schematic](ADA4622/ada4622_ac.asc) |
| AD8642 | +12 V and -12 V | 0.25 µA AC at the trained input node, 1 to 100 Hz | [AC netlist](AD8642/ad8642_ac.net) · [schematic](AD8642/ad8642_ac.asc) |

Run each netlist from its own folder. For the transfer impedance in ohms, divide the complex output `V(out11)` or `V(out15)` by 0.25e-6 A. The logarithmic grid has 400 points per decade and 801 points including the endpoints. Each folder includes its original LTspice vendor model library and symbol. The netlists use the adjacent copy of `ADA4622.lib` or `ADI.lib`. Keep these files in place when moving the examples.

## ADA4622

The separate [time-domain entry](ADA4622/ada4622_chirp.net) ([editable schematic](ADA4622/ada4622_chirp.asc)) applies the original 0.25 µA peak linear chirp through B1 over 100 s, from 1 to 100 Hz. Its source is the project schematic `R2NN55ForParityCheck_Linear_LTspice/R2NN55TestWithoutCellADA4622.asc`. The sweep formula was checked against the current trace in `sweep0p25uA1to100hz100s100us_ParityCheck_linear_ADA4622.txt`. The archived trace contains the original chirp simulation outputs. The separate AC entry uses the same circuit with a small-signal source at the network input. A standalone run gave probe peaks at 24.6888 and 28.5102 Hz.

## AD8642

The AC entry uses the unmodified network and the vendor AD8642 macro-model. It derives from `R2NN55ForParityCheck_Linear_LTspice/R2NN55TestWithoutCellADA8642.asc` (the original project filename spells the device ADA8642). `native_dc.bias` contains LTspice operating-point starting estimates saved from a converged run of this circuit. Keep it beside the AC netlist. LTspice solves the operating point before performing the small-signal analysis.

[Archived complex AC response](AD8642/reference_ac.csv) is copied from `ad8642_sweep/validation/original_ac.csv` in the linear parity project. The two peaks in a new run fell at the same frequency bins as the archived reference, 24.8313 and 28.6748 Hz. Over all 801 frequencies the maximum relative complex-response differences were 0.537% for out11 and 0.533% for out15 (rounded up). This comparison characterizes the low-frequency small-signal response. The project separately records out-of-band instability of the unmodified AD8642 macromodel and uses a compensated circuit or low-frequency equivalent for long transient studies.
