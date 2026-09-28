"""Shared utilities for Model F robustness evaluation and library screening."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_utils import read_csv_flex
from src.feature_expert_router import EXPERT_COLUMNS, build_expert_router_features
from src.feature_rdkit import build_rdkit_matrix, descriptor_column_names

RANDOM_STATE = 42
PHYSICS_ALIASES = {
    "T1": ["T1", "T1_K", "Tm1", "T_m1"],
    "T2": ["T2", "T2_K", "Tm2", "T_m2"],
    "H1": ["H1", "H1_kJmol", "Hfus1", "dH1"],
    "H2": ["H2", "H2_kJmol", "Hfus2", "dH2"],
    "X1": ["X1", "x1", "best_X1_grid"],
    "X2": ["X2", "x2", "best_X2_grid"],
}


def resolve_col(df: pd.DataFrame, preferred: str | None, aliases: list[str], required: bool = True) -> str | None:
    """Resolve a column by exact name or aliases and raise a clear error when required."""
    candidates = [c for c in [preferred, *aliases] if c]
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    if required:
        shown = ", ".join(map(str, list(df.columns)[:80]))
        raise ValueError(f"Cannot find required column. Tried {candidates}. Available columns: {shown}")
    return None


def normalize_training_columns(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    cols = {
        "target": resolve_col(out, args.target_col, ["T_exp", "T_true", "Texp"], True),
        "baseline": resolve_col(out, args.baseline_col, ["T_BA", "Tpred", "T_pred", "best_T_grid"], True),
        "smiles1": resolve_col(out, args.smiles1_col, ["SMILES1", "smiles1", "HBA_SMILES"], False),
        "smiles2": resolve_col(out, args.smiles2_col, ["SMILES2", "smiles2", "HBD_SMILES"], False),
        "name1": resolve_col(out, args.name1_col, ["Component1", "HBA", "HBA_name", "Name1"], False),
        "name2": resolve_col(out, args.name2_col, ["Component2", "HBD", "HBD_name", "Name2"], False),
        "hba_family": resolve_col(out, args.hba_family_col, ["HBA_family", "hba_family", "Family1"], False),
        "hbd_family": resolve_col(out, args.hbd_family_col, ["HBD_family", "hbd_family", "Family2"], False),
    }
    for canonical, aliases in PHYSICS_ALIASES.items():
        src = resolve_col(out, None, aliases, True)
        if src != canonical:
            out[canonical] = out[src]
    if cols["baseline"] != "T_BA":
        out["T_BA"] = out[cols["baseline"]]
        cols["baseline"] = "T_BA"
    if cols["target"] != "T_exp":
        out["T_exp"] = out[cols["target"]]
        cols["target"] = "T_exp"
    if cols["smiles1"] is None:
        out["SMILES1"] = ""
        cols["smiles1"] = "SMILES1"
    if cols["smiles2"] is None:
        out["SMILES2"] = ""
        cols["smiles2"] = "SMILES2"
    return out, cols


def normalize_library_columns(df: pd.DataFrame, baseline_kind: str) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    baseline_aliases = ["T_pred", "T_BA", "Tpred", "T_SLE"] if baseline_kind == "ratio" else ["best_T_grid", "T_pred", "T_BA"]
    cols = {
        "baseline": resolve_col(out, None, baseline_aliases, True),
        "smiles1": resolve_col(out, None, ["SMILES1", "smiles1", "HBA_SMILES"], False),
        "smiles2": resolve_col(out, None, ["SMILES2", "smiles2", "HBD_SMILES"], False),
        "name1": resolve_col(out, None, ["Component1", "HBA", "HBA_name", "Name1"], False),
        "name2": resolve_col(out, None, ["Component2", "HBD", "HBD_name", "Name2"], False),
    }
    for canonical, aliases in PHYSICS_ALIASES.items():
        src = resolve_col(out, None, aliases, True)
        if src != canonical:
            out[canonical] = out[src]
    out["T_BA"] = out[cols["baseline"]]
    if cols["smiles1"] is None:
        out["SMILES1"] = ""
        cols["smiles1"] = "SMILES1"
    if cols["smiles2"] is None:
        out["SMILES2"] = ""
        cols["smiles2"] = "SMILES2"
    return out, cols


def infer_family(name: object, smiles: object = "") -> str:
    text = f"{name or ''} {smiles or ''}".lower()
    patterns = [
        ("choline_quat", r"choline|ammonium|quat|betaine"),
        ("phosphonium", r"phosphonium"),
        ("metal_salt", r"chloride|bromide|iodide|nitrate|sulfate|acetate"),
        ("acid", r"acid|lactate|citrate|malate|formic|acetic"),
        ("amide_urea", r"urea|amide"),
        ("polyol_sugar", r"glycol|glycerol|sorbitol|xylitol|glucose|fructose|sucrose"),
        ("alcohol", r"alkanol|alcohol|ethanol|propanol|butanol"),
        ("amine", r"amine"),
    ]
    for label, pat in patterns:
        if re.search(pat, text):
            return label
    return "other"


def pair_key(df: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    """Return an unordered canonical component-structure pair key.

    Component roles and input order must not affect fold membership.  RDKit
    canonical SMILES are preferred; normalized names are used only when a
    structure is missing or invalid.
    """
    s1 = df[cols["smiles1"]].fillna("").astype(str)
    s2 = df[cols["smiles2"]].fillna("").astype(str)
    n1 = (df[cols["name1"]].fillna("").astype(str) if cols.get("name1") in df.columns
          else pd.Series("", index=df.index))
    n2 = (df[cols["name2"]].fillna("").astype(str) if cols.get("name2") in df.columns
          else pd.Series("", index=df.index))

    def component_id(smiles: object, name: object) -> str:
        text = str(smiles).strip()
        mol = Chem.MolFromSmiles(text) if text else None
        if mol is not None:
            return "smi:" + Chem.MolToSmiles(mol, canonical=True)
        return "name:" + re.sub(r"\s+", " ", str(name).strip().lower())

    a = pd.Series((component_id(s, n) for s, n in zip(s1, n1)), index=df.index, dtype="object")
    b = pd.Series((component_id(s, n) for s, n in zip(s2, n2)), index=df.index, dtype="object")
    lo = pd.concat([a, b], axis=1).min(axis=1)
    hi = pd.concat([a, b], axis=1).max(axis=1)
    return lo + "||" + hi


def ensure_family_columns(df: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    if not cols.get("hba_family"):
        src = out[cols.get("name1")] if cols.get("name1") in out.columns else out[cols["smiles1"]]
        out["HBA_family_inferred"] = [infer_family(n, s) for n, s in zip(src, out[cols["smiles1"]])]
        cols["hba_family"] = "HBA_family_inferred"
    if not cols.get("hbd_family"):
        src = out[cols.get("name2")] if cols.get("name2") in out.columns else out[cols["smiles2"]]
        out["HBD_family_inferred"] = [infer_family(n, s) for n, s in zip(src, out[cols["smiles2"]])]
        cols["hbd_family"] = "HBD_family_inferred"
    return out


def component_text(df: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    n1 = df[cols.get("name1")].astype(str).fillna("") if cols.get("name1") in df.columns else ""
    n2 = df[cols.get("name2")].astype(str).fillna("") if cols.get("name2") in df.columns else ""
    s1 = df[cols["smiles1"]].astype(str).fillna("")
    s2 = df[cols["smiles2"]].astype(str).fillna("")
    return pd.Series([f"component_1: {a}; smiles_1: {b}; component_2: {c}; smiles_2: {d}" for a, b, c, d in zip(n1, s1, n2, s2)], index=df.index)


def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    t1, t2 = pd.to_numeric(df["T1"], errors="coerce"), pd.to_numeric(df["T2"], errors="coerce")
    h1, h2 = pd.to_numeric(df["H1"], errors="coerce"), pd.to_numeric(df["H2"], errors="coerce")
    x1, x2 = pd.to_numeric(df["X1"], errors="coerce"), pd.to_numeric(df["X2"], errors="coerce")
    tb = pd.to_numeric(df["T_BA"], errors="coerce")
    return pd.DataFrame(
        {
            "T_BA": tb,
            "T1": t1,
            "T2": t2,
            "H1": h1,
            "H2": h2,
            "X1": x1,
            "X2": x2,
            "weighted_T": x1 * t1 + x2 * t2,
            "weighted_H": x1 * h1 + x2 * h2,
            "abs_T1_minus_T2": (t1 - t2).abs(),
            "abs_H1_minus_H2": (h1 - h2).abs(),
            "min_X": np.minimum(x1, x2),
            "T_BA_minus_weighted_T": tb - (x1 * t1 + x2 * t2),
            "T_BA_minus_max_T": tb - np.maximum(t1, t2),
            "T_BA_minus_min_T": tb - np.minimum(t1, t2),
            "H_ratio": np.maximum(h1, h2) / (np.minimum(h1, h2).abs() + 1e-9),
        },
        index=df.index,
    )


def rdkit_or_zero(df: pd.DataFrame, smiles_col: str, prefix: str) -> pd.DataFrame:
    s = df[smiles_col].astype(str).fillna("") if smiles_col in df.columns else pd.Series("", index=df.index)
    if s.str.strip().eq("").all():
        return pd.DataFrame(np.zeros((len(df), len(descriptor_column_names(prefix)))), columns=descriptor_column_names(prefix), index=df.index)
    # Atlas rows reuse fewer than two thousand components.  Compute each unique
    # structure once per chunk; this is numerically identical to row-wise RDKit
    # evaluation and makes the frozen 709k rerun tractable.
    unique = pd.Series(pd.unique(s), dtype="object")
    umat = build_rdkit_matrix(unique, prefix)
    lookup = {key: umat.iloc[i].to_numpy(float) for i, key in enumerate(unique)}
    values = np.vstack([lookup[key] for key in s])
    return pd.DataFrame(values, columns=umat.columns, index=df.index)


def static_feature_frames(df: pd.DataFrame, cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    phys = physics_features(df)
    mol1 = rdkit_or_zero(df, cols["smiles1"], "mol1_")
    mol2 = rdkit_or_zero(df, cols["smiles2"], "mol2_")
    router, labels = build_expert_router_features(
        df,
        cols["smiles1"],
        cols["smiles2"],
        name1_col=cols.get("name1") if cols.get("name1") in df.columns else None,
        name2_col=cols.get("name2") if cols.get("name2") in df.columns else None,
        baseline_col="T_BA",
    )
    return pd.concat([phys, mol1, mol2], axis=1), router, labels


def make_estimator(name: str, seed: int):
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=350, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(n_estimators=350, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    return HistGradientBoostingRegressor(max_iter=450, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05, random_state=seed)


def metrics(y_true: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(baseline) & np.isfinite(pred)
    if not m.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "me": np.nan, "r2": np.nan, "baseline_mae": np.nan}
    err = pred[m] - y_true[m]
    berr = baseline[m] - y_true[m]
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "me": float(np.mean(err)),
        "r2": float(r2_score(y_true[m], pred[m])),
        "baseline_mae": float(np.mean(np.abs(berr))),
    }


@dataclass
class FoldPack:
    text_vectorizer: TfidfVectorizer
    text_svd: TruncatedSVD
    imputer_expert: SimpleImputer
    imputer_router: SimpleImputer
    scaler: StandardScaler
    model: object
    backend: str
    target_mode: str


def build_design(train_df: pd.DataFrame, test_df: pd.DataFrame, train_text: pd.Series, test_text: pd.Series, train_expert: pd.DataFrame, test_expert: pd.DataFrame, train_router: pd.DataFrame, test_router: pd.DataFrame, seed: int, pack: FoldPack | None = None):
    """Fit or apply the fold-local text and imputation preprocessing."""
    if pack is None:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000, lowercase=True)
        Xt = vec.fit_transform(train_text)
        Xv = vec.transform(test_text)
        n_comp = max(1, min(64, Xt.shape[0] - 1, Xt.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        Ztr, Zte = svd.fit_transform(Xt), svd.transform(Xv)
        imp_e = SimpleImputer(strategy="median")
        imp_r = SimpleImputer(strategy="median")
        Etr, Ete = imp_e.fit_transform(train_expert), imp_e.transform(test_expert)
        Rtr, Rte = imp_r.fit_transform(train_router), imp_r.transform(test_router)
    else:
        vec, svd, imp_e, imp_r = pack.text_vectorizer, pack.text_svd, pack.imputer_expert, pack.imputer_router
        Ztr, Zte = svd.transform(vec.transform(train_text)), svd.transform(vec.transform(test_text))
        Etr, Ete = imp_e.transform(train_expert), imp_e.transform(test_expert)
        Rtr, Rte = imp_r.transform(train_router), imp_r.transform(test_router)
    gate_tr = Rtr[:, : min(16, Rtr.shape[1])].mean(axis=1, keepdims=True)
    gate_te = Rte[:, : min(16, Rte.shape[1])].mean(axis=1, keepdims=True)
    Ctr = Etr[:, : min(48, Etr.shape[1])] * (1.0 + gate_tr)
    Cte = Ete[:, : min(48, Ete.shape[1])] * (1.0 + gate_te)
    return np.hstack([Etr, Ztr, Rtr, Ctr]), np.hstack([Ete, Zte, Rte, Cte]), (vec, svd, imp_e, imp_r)


def fit_predict_fold(train_idx, test_idx, df, cols, expert_df, router_df, text, target_mode: str, backend: str, seed: int):
    Xtr, Xte, prep = build_design(df.iloc[train_idx], df.iloc[test_idx], text.iloc[train_idx], text.iloc[test_idx], expert_df.iloc[train_idx], expert_df.iloc[test_idx], router_df.iloc[train_idx], router_df.iloc[test_idx], seed)
    y = pd.to_numeric(df["T_exp"], errors="coerce").to_numpy(float)
    baseline = pd.to_numeric(df["T_BA"], errors="coerce").to_numpy(float)
    yfit = y[train_idx] if target_mode == "direct" else y[train_idx] - baseline[train_idx]
    m = np.isfinite(yfit) & np.all(np.isfinite(Xtr), axis=1)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr[m])
    Xte_s = scaler.transform(Xte)
    model = make_estimator(backend, seed)
    model.fit(Xtr_s, yfit[m])
    raw = model.predict(Xte_s)
    pred = raw if target_mode == "direct" else baseline[test_idx] + raw
    pack = FoldPack(*prep, scaler, model, backend, target_mode)
    return pred, pack


def ad_fit(train_df: pd.DataFrame, cols: dict[str, str], train_expert: pd.DataFrame, train_router: pd.DataFrame) -> dict:
    numeric = ["T_BA", "T1", "T2", "H1", "H2", "X1", "X2"]
    q01 = train_df[numeric].apply(pd.to_numeric, errors="coerce").quantile(0.01)
    q99 = train_df[numeric].apply(pd.to_numeric, errors="coerce").quantile(0.99)
    base_features = pd.concat([train_expert, train_router], axis=1)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imp.fit_transform(base_features))
    rng = np.random.default_rng(RANDOM_STATE)
    ref_n = min(1024, len(X))
    ref_idx = rng.choice(len(X), size=ref_n, replace=False) if len(X) > ref_n else np.arange(len(X))
    ref_X = X[ref_idx]
    train_dist = min_reference_distance(X, ref_X)
    return {
        "numeric": numeric,
        "q01": q01,
        "q99": q99,
        "imp": imp,
        "scaler": scaler,
        "ref_X": ref_X,
        "d50": float(np.nanquantile(train_dist, 0.50)),
        "d90": float(np.nanquantile(train_dist, 0.90)),
        "d99": float(np.nanquantile(train_dist, 0.99)),
        "pairs": set(pair_key(train_df, cols)),
        "hba_families": set(train_df[cols["hba_family"]].astype(str)),
        "hbd_families": set(train_df[cols["hbd_family"]].astype(str)),
        "router_covered": set(train_router.columns[(train_router.fillna(0) != 0).any(axis=0)]),
    }


def min_reference_distance(X: np.ndarray, ref_X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """Approximate nearest-reference distance without sklearn threadpool dependencies."""
    out = np.empty(len(X), dtype=float)
    ref_norm = np.sum(ref_X * ref_X, axis=1)[None, :]
    for start in range(0, len(X), batch_size):
        block = X[start : start + batch_size]
        d2 = np.sum(block * block, axis=1)[:, None] + ref_norm - 2.0 * block @ ref_X.T
        out[start : start + batch_size] = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
    return out


def ad_score(test_df: pd.DataFrame, cols: dict[str, str], test_expert: pd.DataFrame, test_router: pd.DataFrame, state: dict) -> pd.DataFrame:
    vals = test_df[state["numeric"]].apply(pd.to_numeric, errors="coerce")
    outside = ((vals.lt(state["q01"], axis=1)) | (vals.gt(state["q99"], axis=1))).sum(axis=1).to_numpy()
    X = state["scaler"].transform(state["imp"].transform(pd.concat([test_expert, test_router], axis=1)))
    dist = min_reference_distance(X, state["ref_X"])
    pair_seen = pair_key(test_df, cols).isin(state["pairs"]).to_numpy()
    hba_seen = test_df[cols["hba_family"]].astype(str).isin(state["hba_families"]).to_numpy()
    hbd_seen = test_df[cols["hbd_family"]].astype(str).isin(state["hbd_families"]).to_numpy()
    active_router = (test_router.reindex(columns=list(state["router_covered"]), fill_value=0).fillna(0) != 0).sum(axis=1).to_numpy()
    router_score = np.clip(active_router / 3.0, 0, 1)
    range_score = np.clip(1.0 - outside / max(1, len(state["numeric"])), 0, 1)
    dist_score = np.where(dist <= state["d90"], 1.0, np.where(dist <= state["d99"], 0.55, 0.1))
    score = 0.30 * range_score + 0.25 * dist_score + 0.15 * pair_seen + 0.10 * hba_seen + 0.10 * hbd_seen + 0.10 * router_score
    level = np.where((~hba_seen) | (~hbd_seen) | (outside >= 3) | (dist > state["d99"] * 1.25), "OOD", np.where(score >= 0.80, "high", np.where(score >= 0.55, "medium", "low")))
    return pd.DataFrame({"AD_score": score, "AD_level": level, "OOD": level == "OOD", "nn_distance": dist, "numeric_outside_count": outside, "pair_seen": pair_seen, "hba_family_seen": hba_seen, "hbd_family_seen": hbd_seen, "router_active_covered_count": active_router}, index=test_df.index)


def missing_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c in df.columns:
            rows.append({"column": c, "missing_fraction": float(pd.to_numeric(df[c], errors="coerce").isna().mean() if c in PHYSICS_ALIASES else df[c].isna().mean())})
    return pd.DataFrame(rows)


def conformal_radius(y_cal: np.ndarray, pred_cal: np.ndarray, alpha: float = 0.10) -> float:
    resid = np.abs(y_cal - pred_cal)
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0:
        return float("nan")
    q = math.ceil((len(resid) + 1) * (1 - alpha)) / len(resid)
    return float(np.quantile(resid, min(q, 1.0), method="higher"))
