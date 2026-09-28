"""Chemistry mechanism router features for expert residual models."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError as e:
    raise ImportError(
        "Expert router requires RDKit. Install with: pip install rdkit-pypi or conda install -c conda-forge rdkit"
    ) from e


EXPERT_COLUMNS = [
    "expert_ionic_halide",
    "expert_quaternary_ammonium",
    "expert_phosphonium",
    "expert_metal_salt",
    "expert_organic_acid",
    "expert_carboxylate",
    "expert_amide_urea",
    "expert_amine_basic",
    "expert_alcohol_polyol",
    "expert_sugar_polyol",
    "expert_phenolic_aromatic",
    "expert_ether_ester",
    "expert_hydrophobic_weak",
    "expert_high_polarity",
    "expert_high_flexibility",
    "expert_bulky_asymmetric",
    "expert_strong_hbond_network",
    "expert_weak_hbond_low_polarity",
    "expert_extreme_composition",
    "expert_large_component_tm_gap",
    "expert_baseline_below_components",
    "expert_baseline_above_components",
]

NUMERIC_MECHANISM_COLUMNS = [
    "router_hbond_capacity",
    "router_charge_count",
    "router_halide_count",
    "router_acid_count",
    "router_polyol_score",
    "router_hydrophobic_score",
    "router_polarity_score",
    "router_flexibility_score",
    "router_component_tm_gap",
    "router_min_x",
    "router_tpred_position",
]

_SMARTS = {
    "halide": ["[Cl-]", "[Br-]", "[I-]", "[F-]"],
    "quaternary_ammonium": ["[N+;X4]"],
    "phosphonium": ["[P+;X4]"],
    "metal": ["[Li+]", "[Na+]", "[K+]", "[Mg+2]", "[Ca+2]", "[Zn+2]", "[Al+3]"],
    "organic_acid": ["C(=O)[O;H1]", "C(=O)O"],
    "carboxylate": ["C(=O)[O-]"],
    "amide": ["C(=O)N"],
    "urea": ["NC(=O)N"],
    "amine": ["[NX3;H2,H1,H0;!$(NC=O)]"],
    "alcohol": ["[OX2H]"],
    "phenol": ["c[OX2H]"],
    "ether": ["[OD2]([#6])[#6]"],
    "ester": ["C(=O)O[#6]"],
    "aromatic": ["a"],
}


def _mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def _count(mol, key: str) -> int:
    if mol is None:
        return 0
    total = 0
    for smarts in _SMARTS[key]:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            total += len(mol.GetSubstructMatches(patt))
    return int(total)


def _formal_charge_abs(mol) -> int:
    if mol is None:
        return 0
    return int(sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _safe_desc(mol, func) -> float:
    if mol is None:
        return 0.0
    try:
        val = float(func(mol))
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _name_hint(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, str(text or ""), flags=re.IGNORECASE))


def _load_llm_router_labels(path: Path | None) -> dict[str, dict[str, float]]:
    """Optional cache keyed by 'SMILES1||SMILES2' with expert_* scores from an LLM."""
    if path is None or not path.is_file():
        return {}

    out: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj.get("key", "")).strip()
            if not key:
                continue
            scores = obj.get("expert_scores", obj)
            out[key] = {
                col: float(scores.get(col, 0.0))
                for col in EXPERT_COLUMNS
                if col in scores
            }
    return out


def build_expert_router_features(
    df: pd.DataFrame,
    smiles1_col: str,
    smiles2_col: str,
    *,
    name1_col: str | None = None,
    name2_col: str | None = None,
    baseline_col: str = "Tpred",
    llm_router_path: Path | None = None,
    llm_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build broad chemistry mechanism indicators for LLM/expert-style routing.

    If llm_router_path is supplied, it may contain JSONL rows:
    {"key": "SMILES1||SMILES2", "expert_scores": {"expert_ionic_halide": 1, ...}}
    Those scores are blended with the rule router by llm_blend.
    """
    llm_scores = _load_llm_router_labels(llm_router_path)
    llm_blend = float(np.clip(llm_blend, 0.0, 1.0))

    rows: list[dict[str, float]] = []
    labels: list[str] = []

    for _, row in df.iterrows():
        s1 = str(row.get(smiles1_col, "") or "")
        s2 = str(row.get(smiles2_col, "") or "")
        n1 = str(row.get(name1_col, "") or "") if name1_col else ""
        n2 = str(row.get(name2_col, "") or "") if name2_col else ""
        text = f"{n1} {n2} {s1} {s2}"

        m1 = _mol(s1)
        m2 = _mol(s2)
        mols = [m for m in (m1, m2) if m is not None]

        halide = sum(_count(m, "halide") for m in mols)
        quat = sum(_count(m, "quaternary_ammonium") for m in mols)
        phosph = sum(_count(m, "phosphonium") for m in mols)
        metal = sum(_count(m, "metal") for m in mols)
        acid = sum(_count(m, "organic_acid") for m in mols)
        carboxylate = sum(_count(m, "carboxylate") for m in mols)
        amide_urea = sum(_count(m, "amide") + _count(m, "urea") for m in mols)
        amine = sum(_count(m, "amine") for m in mols)
        alcohol = sum(_count(m, "alcohol") for m in mols)
        phenol = sum(_count(m, "phenol") for m in mols)
        ether_ester = sum(_count(m, "ether") + _count(m, "ester") for m in mols)
        aromatic = sum(_count(m, "aromatic") for m in mols)
        charge = sum(_formal_charge_abs(m) for m in mols)

        donors = sum(_safe_desc(m, Descriptors.NumHDonors) for m in mols)
        acceptors = sum(_safe_desc(m, Descriptors.NumHAcceptors) for m in mols)
        tpsa = sum(_safe_desc(m, Descriptors.TPSA) for m in mols)
        logp = sum(_safe_desc(m, Descriptors.MolLogP) for m in mols)
        rot = sum(_safe_desc(m, Descriptors.NumRotatableBonds) for m in mols)
        heavy = sum(_safe_desc(m, Descriptors.HeavyAtomCount) for m in mols)
        mw_delta = abs(_safe_desc(m1, Descriptors.MolWt) - _safe_desc(m2, Descriptors.MolWt))

        x1 = float(pd.to_numeric(row.get("X1", np.nan), errors="coerce"))
        x2 = float(pd.to_numeric(row.get("X2", np.nan), errors="coerce"))
        t1 = float(pd.to_numeric(row.get("T1", np.nan), errors="coerce"))
        t2 = float(pd.to_numeric(row.get("T2", np.nan), errors="coerce"))
        tp = float(pd.to_numeric(row.get(baseline_col, np.nan), errors="coerce"))
        min_x = np.nanmin([x1, x2]) if np.isfinite([x1, x2]).all() else np.nan
        tm_gap = abs(t1 - t2) if np.isfinite([t1, t2]).all() else np.nan
        tmin = min(t1, t2) if np.isfinite([t1, t2]).all() else np.nan
        tmax = max(t1, t2) if np.isfinite([t1, t2]).all() else np.nan

        sugar_hint = _name_hint(text, r"glucose|fructose|sorbitol|xylitol|mannitol|sucrose|sugar|glycol|glycerol")
        acid_hint = _name_hint(text, r"acid|acetate|lactate|citrate|malate|oxalate")
        amide_hint = _name_hint(text, r"urea|acetamide|amide")
        choline_hint = _name_hint(text, r"choline|betaine")

        flags = {
            "expert_ionic_halide": float((halide > 0 and charge > 0) or choline_hint),
            "expert_quaternary_ammonium": float(quat > 0 or choline_hint),
            "expert_phosphonium": float(phosph > 0),
            "expert_metal_salt": float(metal > 0),
            "expert_organic_acid": float(acid > 0 or acid_hint),
            "expert_carboxylate": float(carboxylate > 0),
            "expert_amide_urea": float(amide_urea > 0 or amide_hint),
            "expert_amine_basic": float(amine > 0),
            "expert_alcohol_polyol": float(alcohol >= 2),
            "expert_sugar_polyol": float(alcohol >= 4 or sugar_hint),
            "expert_phenolic_aromatic": float(phenol > 0 or (aromatic > 0 and alcohol > 0)),
            "expert_ether_ester": float(ether_ester > 0),
            "expert_hydrophobic_weak": float(logp > 3.0 and donors + acceptors <= 3),
            "expert_high_polarity": float(tpsa > 90 or donors + acceptors >= 8),
            "expert_high_flexibility": float(rot >= 8),
            "expert_bulky_asymmetric": float(mw_delta > 150 or heavy > 60),
            "expert_strong_hbond_network": float((donors + acceptors >= 8 and alcohol >= 2) or (halide > 0 and alcohol >= 2)),
            "expert_weak_hbond_low_polarity": float(donors + acceptors <= 3 and tpsa < 45),
            "expert_extreme_composition": float(np.isfinite(min_x) and min_x < 0.15),
            "expert_large_component_tm_gap": float(np.isfinite(tm_gap) and tm_gap > 150),
            "expert_baseline_below_components": float(np.isfinite(tp) and np.isfinite(tmin) and tp < tmin - 20),
            "expert_baseline_above_components": float(np.isfinite(tp) and np.isfinite(tmax) and tp > tmax + 20),
        }

        key = f"{s1}||{s2}"
        if llm_scores and key in llm_scores and llm_blend > 0:
            for col in EXPERT_COLUMNS:
                flags[col] = (1.0 - llm_blend) * flags[col] + llm_blend * float(llm_scores[key].get(col, flags[col]))

        hbond_capacity = donors + acceptors
        row_out = {
            **flags,
            "router_hbond_capacity": float(hbond_capacity),
            "router_charge_count": float(charge),
            "router_halide_count": float(halide),
            "router_acid_count": float(acid + carboxylate),
            "router_polyol_score": float(alcohol),
            "router_hydrophobic_score": float(logp),
            "router_polarity_score": float(tpsa),
            "router_flexibility_score": float(rot),
            "router_component_tm_gap": float(tm_gap) if np.isfinite(tm_gap) else np.nan,
            "router_min_x": float(min_x) if np.isfinite(min_x) else np.nan,
            "router_tpred_position": float((tp - tmin) / (tmax - tmin + 1e-9))
            if np.isfinite([tp, tmin, tmax]).all()
            else np.nan,
        }
        rows.append(row_out)

        active = [col for col in EXPERT_COLUMNS if row_out[col] >= 0.5]
        labels.append(active[0].replace("expert_", "") if active else "global")

    feature_df = pd.DataFrame(rows, index=df.index)
    return feature_df[EXPERT_COLUMNS + NUMERIC_MECHANISM_COLUMNS], pd.Series(labels, index=df.index, name="expert_primary_label")
