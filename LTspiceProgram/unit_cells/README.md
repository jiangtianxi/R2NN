# Unit-cell circuit simulations

These circuits support the damping and nonlinear unit-cell analyses in Supplementary Note 3. Both retain the amplifier models and excitation used in the corresponding study.

| Analysis | Example to open in LTspice | Other cases |
|---|---|---|
| Nonlinear unit cell, upward frequency sweep | [Nominal 1–200 Hz chirp](nonlinear/sweeps/alpha5e-3beta2e-4_100nA1to200hz200s100us.cir) | [60 upward and downward sweeps](nonlinear/sweeps/) spanning linear stiffness, cubic stiffness and input current. The component topology can also be inspected in the [source schematic](nonlinear/R2NN77UnitCellTest.asc). |
| Damping, reference circuit | [TL082 AC response](damping/baseline/tl082_coarse.net) | [Reference cases](damping/baseline/) include TL082 and ideal amplifiers with 50 pF at the ground nodes, across the internal resistor, or at both locations. Each case has coarse and fine AC sweeps and a chirp. |
| Damping, capacitance sweep | [AC example with 10 pF to ground and 20 pF across the internal resistor](damping/ac_sweep/both_cg10_cn20.net) | [AC cases](damping/ac_sweep/) and [chirp cases](damping/chirp_sweep/) cover 120 ideal-amplifier capacitance combinations and one shared TL082 reference for each analysis. |

For the damping circuits, `Rn` is 200 kΩ. AC sweeps use a unit current source. The time-domain excitation is a 0.05 μA peak chirp from 1 to 200 Hz over 200 s. Each case uses the maximum integration step recorded for its final evaluation.

The [damping case index](damping/case_index.csv) distinguishes the archived netlists from the seven chirp netlists regenerated using the original project template and the final case parameters. The reference shared between the two studies appears in both folders. Run the text netlist in its own directory so it can find the matching sibling `models` directory. A full 200 s transient can produce a large waveform file.
