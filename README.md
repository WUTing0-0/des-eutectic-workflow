# Eutectic mixture design workflow

Code for composition-resolved eutectic prediction using pure-component properties, solid–liquid equilibrium and machine learning of liquidus measurements.

## Available here

- The six-model, pair-disjoint liquidus benchmark and its exact molecular-feature definitions.
- Pinned CPU dependencies and a portable command-line entry point.
- Independent-environment verification records for the benchmark, production-feature regression and matched pure-property replacement.

For the downloadable local graphical interface, examples and installation instructions, see [DES Design Workbench](https://github.com/WUTing0-0/des-design-workbench).

## Run the benchmark

Use Python 3.12 in a fresh virtual environment and install `requirements.txt`. Supply the legally obtained, prepared benchmark table using the `DES_BENCHMARK_INPUT` environment variable. Then run:

```text
python workflow/benchmark.py
```

`DES_BENCHMARK_OUTPUT` optionally selects an output directory; the default is `outputs`. All compositions of a canonical pair are kept in one of five folds. The program trains the four learned formulations, evaluates the two physical references and writes held-out predictions and metrics. See [the input specification](docs/INPUTS.md).

## Reproducibility scope

The independent test retrained the complete six-model mixture benchmark from its prepared thermodynamic inputs: all 22,788 model–observation predictions matched the frozen reference within the stated tolerance. It also reran the actual production-feature regression and complete matched-input replacement analysis. These checks do not constitute regeneration of quantum-chemical surfaces, retraining of all four pure-property models, or a new run of every Atlas curve. Those stages require additional inputs, weights and compute resources.

This public repository does not contain third-party experimental tables whose redistribution terms remain unverified, or third-party foundation-model checkpoints. Public availability of an upstream repository is not, by itself, permission to redistribute its contents. See [data access and licensing](docs/DATA_ACCESS.md).

## Interpretation

The workflow estimates liquidus behaviour for screening. Chemical-space support and calibrated intervals are separate from numerical validity. Sensitivity ranges are not calibrated probabilities of DES formation. Proposed formulations are not completed measurements.

## Licence

The original code in this repository is MIT-licensed. Third-party packages retain their own licences. No blanket data licence is applied to third-party data, pretrained weights or restricted model outputs. In particular, TabPFN v2.6 has terms distinct from the licence of its Python package; obtain that model from its official provider and check the terms for the intended use.
