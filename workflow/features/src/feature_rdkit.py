"""RDKit molecular descriptors for F4 models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError as e:
    raise ImportError(
        "RDKit is required. Install the pinned requirements before running."
    ) from e

DESCRIPTOR_SPECS: list[tuple[str, str]] = [
    ("MolWt", "MolWt"),
    ("HeavyAtomMolWt", "HeavyAtomMolWt"),
    ("MolLogP", "MolLogP"),
    ("TPSA", "TPSA"),
    ("NumHAcceptors", "NumHAcceptors"),
    ("NumHDonors", "NumHDonors"),
    ("NumRotatableBonds", "NumRotatableBonds"),
    ("RingCount", "RingCount"),
    ("NHOHCount", "NHOHCount"),
    ("NOCount", "NOCount"),
    ("FractionCSP3", "FractionCSP3"),
    ("HeavyAtomCount", "HeavyAtomCount"),
    ("ExactMolWt", "ExactMolWt"),
    ("LabuteASA", "LabuteASA"),
    ("BalabanJ", "BalabanJ"),
    ("BertzCT", "BertzCT"),
]

_DESCRIPTOR_FUNCS: dict[str, callable] = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "ExactMolWt": Descriptors.ExactMolWt,
    "LabuteASA": Descriptors.LabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
}


def descriptor_column_names(prefix: str) -> list[str]:
    return [f"{prefix}{name}" for name, _ in DESCRIPTOR_SPECS]


def smiles_to_descriptor_vector(smiles: str, prefix: str) -> np.ndarray:
    """Parse SMILES and return descriptor vector; invalid SMILES -> zeros."""
    n = len(DESCRIPTOR_SPECS)
    out = np.zeros(n, dtype=np.float64)
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn(f"Empty SMILES for prefix {prefix}; using zero descriptors", stacklevel=2)
        return out

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        warnings.warn(f"Invalid SMILES '{smiles[:40]}...' for prefix {prefix}; using zero descriptors", stacklevel=2)
        return out

    for i, (_, key) in enumerate(DESCRIPTOR_SPECS):
        try:
            val = float(_DESCRIPTOR_FUNCS[key](mol))
            if np.isfinite(val):
                out[i] = val
        except Exception:
            out[i] = 0.0
    return out


def build_rdkit_matrix(smiles_series: pd.Series, prefix: str) -> pd.DataFrame:
    """Build descriptor DataFrame with columns like mol1_MolWt."""
    cols = descriptor_column_names(prefix)
    rows = [smiles_to_descriptor_vector(str(s) if pd.notna(s) else "", prefix) for s in smiles_series]
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)
