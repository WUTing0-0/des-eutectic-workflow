"""Data loading and numeric feature engineering for F4."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from sklearn.impute import SimpleImputer

F4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = F4_ROOT.parents[1]
if str(F4_ROOT) not in sys.path:
    sys.path.insert(0, str(F4_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_csv_flex(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    if last:
        raise last
    return pd.read_csv(path)


def load_config(config_path: Path | str | None) -> dict:
    if config_path is None:
        default = F4_ROOT / "config.yaml"
        if default.is_file():
            with open(default, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_column(df: pd.DataFrame, col: str, aliases: list[str] | None = None) -> str:
    if col in df.columns:
        return col
    for alt in aliases or []:
        if alt in df.columns:
            return alt
    raise ValueError(f"列 '{col}' 不存在；可用列: {list(df.columns)[:20]}...")


def resolve_groups(
    df: pd.DataFrame,
    group_col: str,
    smiles1_col: str,
    smiles2_col: str,
) -> np.ndarray:
    # Structure identity takes precedence over historical ordered/named group
    # identifiers.  This prevents A+B and B+A, or aliases of the same
    # structure, from entering different folds.
    if smiles1_col in df.columns and smiles2_col in df.columns:
        def canonical(value: object) -> str:
            text = str(value).strip()
            mol = Chem.MolFromSmiles(text) if text else None
            return Chem.MolToSmiles(mol, canonical=True) if mol is not None else text.lower()

        s1 = df[smiles1_col].fillna("").map(canonical)
        s2 = df[smiles2_col].fillna("").map(canonical)
        lo = pd.concat([s1, s2], axis=1).min(axis=1)
        hi = pd.concat([s1, s2], axis=1).max(axis=1)
        return pd.factorize(lo + "||" + hi)[0]

    if group_col in df.columns:
        return pd.factorize(df[group_col].astype(str))[0]
    raise ValueError("Cannot construct canonical pair groups: both SMILES columns are required")


def resolve_tpred(
    df: pd.DataFrame,
    baseline_col: str,
    *,
    compute_mode: str = "none",
    tau: float = 10.0,
    newton_steps: int = 6,
    device: str = "cpu",
) -> pd.DataFrame:
    out = df.copy()
    compute_mode = (compute_mode or "none").lower()

    if compute_mode == "max_t1_t2":
        for col in ("T1", "T2"):
            if col not in out.columns:
                raise ValueError(f"--compute-tpred max_t1_t2 requires column {col}")
        t1 = pd.to_numeric(out["T1"], errors="coerce")
        t2 = pd.to_numeric(out["T2"], errors="coerce")
        out[baseline_col] = np.maximum(t1.values.astype(np.float64), t2.values.astype(np.float64))
        return out

    if compute_mode == "max_branch":
        for col in ("T_liq_1", "T_liq_2"):
            if col not in out.columns:
                raise ValueError(f"--compute-tpred max_branch requires column {col}")
        t1 = pd.to_numeric(out["T_liq_1"], errors="coerce")
        t2 = pd.to_numeric(out["T_liq_2"], errors="coerce")
        out[baseline_col] = np.maximum(t1.values.astype(np.float64), t2.values.astype(np.float64))
        return out

    if compute_mode == "branch_aware":
        try:
            import torch
            from experiments.baseline_branchaware_followup import compute_branch_aware_arrays
        except Exception as e:
            raise ImportError(
                "--compute-tpred branch_aware 需要可用的 torch 环境及 "
                "experiments.baseline_branchaware_followup。"
                f"\n当前错误: {e}"
                "\n可临时改用预先计算好的列（如 T_BA）并设置 --compute-tpred none。"
            ) from e

        dev = torch.device(device)
        t_ba, _, _, _, _ = compute_branch_aware_arrays(
            out, device=dev, tau=float(tau), n_steps=int(newton_steps)
        )
        out[baseline_col] = t_ba
        return out

    if baseline_col not in out.columns:
        for alt in ("Tpred", "T_pred", "T_BA", "T_ba"):
            if alt in out.columns:
                out[baseline_col] = pd.to_numeric(out[alt], errors="coerce")
                return out
        raise ValueError(
            f"缺少基线列 '{baseline_col}'。请提供 CSV 列或使用 --compute-tpred branch_aware"
        )

    out[baseline_col] = pd.to_numeric(out[baseline_col], errors="coerce")
    return out


NUMERIC_BASE_COLS = ["T1", "T2", "Tpred", "H1", "H2", "X1", "X2"]


def build_numeric_features(df: pd.DataFrame, tpred_col: str = "Tpred") -> pd.DataFrame:
    """Build numeric feature matrix; T_true must NOT be included."""
    req = ["T1", "T2", "H1", "H2", "X1", "X2"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"缺少数值特征列 {c}")

    t1 = pd.to_numeric(df["T1"], errors="coerce").values.astype(np.float64)
    t2 = pd.to_numeric(df["T2"], errors="coerce").values.astype(np.float64)
    h1 = pd.to_numeric(df["H1"], errors="coerce").values.astype(np.float64)
    h2 = pd.to_numeric(df["H2"], errors="coerce").values.astype(np.float64)
    x1 = pd.to_numeric(df["X1"], errors="coerce").values.astype(np.float64)
    x2 = pd.to_numeric(df["X2"], errors="coerce").values.astype(np.float64)
    tpred = pd.to_numeric(df[tpred_col], errors="coerce").values.astype(np.float64)

    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    t_diff = t1 - t2

    data = {
        "T1": t1,
        "T2": t2,
        "Tpred": tpred,
        "H1": h1,
        "H2": h2,
        "X1": x1,
        "X2": x2,
        "T1_minus_T2": t_diff,
        "abs_T1_minus_T2": np.abs(t_diff),
        "max_T1_T2": t_max,
        "min_T1_T2": t_min,
        "Tpred_minus_min_T1_T2": tpred - t_min,
        "Tpred_minus_max_T1_T2": tpred - t_max,
    }
    return pd.DataFrame(data, index=df.index)


def build_gap_labels(
    df: pd.DataFrame,
    target_col: str,
    tpred_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_true = pd.to_numeric(df[target_col], errors="coerce").values.astype(np.float64)
    tpred = pd.to_numeric(df[tpred_col], errors="coerce").values.astype(np.float64)
    gap_true = t_true - tpred
    return t_true, tpred, gap_true


class FoldPreprocessor:
    """Per-fold median imputer for numeric features only."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> FoldPreprocessor:
        self.feature_names_ = feature_names
        self.imputer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.imputer.transform(X)


def load_data(
    data_path: Path | str,
    *,
    target_col: str = "T_exp",
    baseline_col: str = "Tpred",
    smiles1_col: str = "SMILES1",
    smiles2_col: str = "SMILES2",
    group_col: str = "des_group_id",
    compute_tpred: str = "none",
    tau: float = 10.0,
    newton_steps: int = 6,
    device: str = "cpu",
) -> tuple[pd.DataFrame, np.ndarray, str, str]:
    df = read_csv_flex(data_path)

    target_col = resolve_column(df, target_col, aliases=["T_true", "T_exp"])
    smiles1_col = resolve_column(df, smiles1_col, aliases=["SMILES1", "smiles1"])
    smiles2_col = resolve_column(df, smiles2_col, aliases=["SMILES2", "smiles2"])

    df = resolve_tpred(
        df,
        baseline_col,
        compute_mode=compute_tpred,
        tau=tau,
        newton_steps=newton_steps,
        device=device,
    )

    t_true = pd.to_numeric(df[target_col], errors="coerce")
    tpred = pd.to_numeric(df[baseline_col], errors="coerce")
    mask = np.isfinite(t_true.values) & np.isfinite(tpred.values)
    if not mask.all():
        n_drop = int((~mask).sum())
        warnings.warn(f"丢弃 {n_drop} 行（T_true 或 Tpred 非有限）")
        df = df.loc[mask].reset_index(drop=True)

    groups = resolve_groups(df, group_col, smiles1_col, smiles2_col)
    return df, groups, target_col, baseline_col


def ensure_output_dirs(out_root: Path) -> dict[str, Path]:
    out_root = Path(out_root)
    dirs = {
        "root": out_root,
        "metrics": out_root / "metrics",
        "predictions": out_root / "predictions",
        "figures": out_root / "figures",
        "models": out_root / "models",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
