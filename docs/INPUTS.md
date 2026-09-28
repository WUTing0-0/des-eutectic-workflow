# Benchmark inputs

The input is a CSV with one experimental liquidus observation per row. Required columns include `source_row`, `Component1`, `Component2`, `SMILES1`, `SMILES2`, `X1`, `X2`, `T1`, `T2`, `H1`, `H2`, `T_exp`, `T_BA`, `hardmax_branch1_K`, `hardmax_branch2_K` and `hardmax_success`. Preserve any source classification fields used by the archived training table.

Temperatures are in kelvin, fusion enthalpies in kJ mol−1, and compositions are mole fractions. `T_BA` is the frozen non-ideal physical reference. Its two branch columns must be computed using the same inputs and activity convention. The original benchmark used activity coefficients evaluated at 298.15 K in the simplified COSMO-based implementation. Changing that convention changes the benchmark and requires a new evaluation.

The evaluation retains successful finite observations, constructs canonical unordered-pair groups, fits transformations within each fold and evaluates held-out pairs. The two physical references are not fitted. The direct and residual learned formulations share their feature matrix and model capacity.

The original input contained 3,798 admitted observations from 389 canonical pairs. A different input file defines a different experiment; its metrics must not be labelled as reproduction of the manuscript results.
