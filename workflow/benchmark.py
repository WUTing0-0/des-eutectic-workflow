from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


PKG = Path(__file__).resolve().parent
ROOT = PKG
OUT = Path(os.environ.get("DES_BENCHMARK_OUTPUT", str(PKG.parent / "outputs")))
DATA = Path(os.environ["DES_BENCHMARK_INPUT"])
RC = ROOT / "features"
sys.path.insert(0, str(RC))
from model_f_robustness_utils import (  # noqa: E402
    component_text,
    ensure_family_columns,
    normalize_training_columns,
    pair_key,
    rdkit_or_zero,
    static_feature_frames,
)

R = 8.31446261815324
SEED = 42
MODEL_CONFIG = dict(max_iter=450, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05)


class Args:
    target_col = "T_exp"
    baseline_col = "T_BA"
    smiles1_col = "SMILES1"
    smiles2_col = "SMILES2"
    name1_col = "Component1"
    name2_col = "Component2"
    hba_family_col = ""
    hbd_family_col = ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ideal_branch(x, tm, hfus_kj):
    x = np.clip(pd.to_numeric(x).to_numpy(float), 1e-12, 1.0)
    tm = pd.to_numeric(tm).to_numpy(float)
    h = pd.to_numeric(hfus_kj).to_numpy(float) * 1000.0
    out = 1.0 / (1.0 / tm - R / h * np.log(x))
    out[(~np.isfinite(out)) | (out >= tm) | (tm <= 0) | (h <= 0)] = np.nan
    return out


def prepare_data():
    raw = pd.read_csv(DATA)
    raw = raw[raw.hardmax_success.astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    df, cols = normalize_training_columns(raw, Args())
    df = ensure_family_columns(df, cols).dropna(subset=["T_exp", "T_BA"]).reset_index(drop=True)
    keys = pair_key(df, cols).reset_index(drop=True)
    text = component_text(df, cols).reset_index(drop=True)
    mol1 = rdkit_or_zero(df, "SMILES1", "mol1_")
    mol2 = rdkit_or_zero(df, "SMILES2", "mol2_")
    _, router, _ = static_feature_frames(df, cols)

    x1, x2 = pd.to_numeric(df.X1), pd.to_numeric(df.X2)
    t1, t2 = pd.to_numeric(df.T1), pd.to_numeric(df.T2)
    h1, h2 = pd.to_numeric(df.H1), pd.to_numeric(df.H2)
    hard = pd.to_numeric(df.T_BA)
    ib1, ib2 = ideal_branch(x1, t1, h1), ideal_branch(x2, t2, h2)
    ideal = np.fmax(ib1, ib2)

    comp = pd.DataFrame({"X1": x1, "X2": x2, "min_X": np.minimum(x1, x2)})
    structural_router_cols = [c for c in router.columns if not any(k in c for k in
        ["baseline", "tpred", "tm_gap", "min_x", "extreme_composition", "large_component_tm_gap"])]
    structural = pd.concat([comp, mol1, mol2, router[structural_router_cols]], axis=1)
    thermal = pd.DataFrame({
        "T1": t1, "T2": t2, "H1": h1, "H2": h2,
        "weighted_T": x1 * t1 + x2 * t2, "weighted_H": x1 * h1 + x2 * h2,
        "abs_T_gap": (t1 - t2).abs(), "abs_H_gap": (h1 - h2).abs(),
    })
    ideal_f = pd.DataFrame({
        "ideal_branch1": ib1, "ideal_branch2": ib2, "ideal_envelope": ideal,
        "ideal_minus_weighted_T": ideal - (x1 * t1 + x2 * t2),
    })
    nonideal = pd.DataFrame({
        "hardmax_branch1": pd.to_numeric(df.hardmax_branch1_K),
        "hardmax_branch2": pd.to_numeric(df.hardmax_branch2_K),
        "T_hardmax": hard, "ideal_minus_hardmax": ideal - hard,
        "hardmax_minus_min_T": hard - np.minimum(t1, t2),
        "hardmax_minus_max_T": hard - np.maximum(t1, t2),
    })
    physics_router_cols = [c for c in router.columns if c not in structural_router_cols]
    blocks = {
        "B2_structure_composition": structural,
        "B3_plus_pure_ideal": pd.concat([structural, thermal, ideal_f], axis=1),
        "B4_direct_plus_nonideal": pd.concat([structural, thermal, ideal_f, nonideal, router[physics_router_cols]], axis=1),
        "B5_residual_plus_nonideal": pd.concat([structural, thermal, ideal_f, nonideal, router[physics_router_cols]], axis=1),
    }
    return df, cols, keys, text, blocks, ideal


def metrics(y, pred, pairs):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    e = pred - y
    pmae = pd.DataFrame({"pair": np.asarray(pairs), "ae": np.abs(e)}).groupby("pair").ae.mean()
    return {
        "records": len(y), "pairs": pd.Series(pairs).nunique(),
        "record_MAE_K": np.abs(e).mean(), "pair_macro_MAE_K": pmae.mean(),
        "RMSE_K": np.sqrt(np.mean(e ** 2)), "bias_K": e.mean(), "R2": r2_score(y, pred),
    }


def fit_split(df, keys, text, blocks, ideal, tr, te, split_id):
    y = pd.to_numeric(df.T_exp).to_numpy(float)
    hard = pd.to_numeric(df.T_BA).to_numpy(float)
    rows, predictions = [], []
    fixed = {"B0_ideal_SLE": ideal[te], "B1_hardmax_COSMO_SLE": hard[te]}

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          max_features=12000, lowercase=True)
    ztr0, zte0 = vec.fit_transform(text.iloc[tr]), vec.transform(text.iloc[te])
    nc = max(1, min(64, ztr0.shape[0] - 1, ztr0.shape[1] - 1))
    svd = TruncatedSVD(n_components=nc, random_state=SEED + split_id)
    ztr, zte = svd.fit_transform(ztr0), svd.transform(zte0)

    for model_name, pred in fixed.items():
        row = {"split": "canonical_pair_disjoint", "split_id": split_id, "model": model_name}
        row.update(metrics(y[te], pred, keys.iloc[te])); rows.append(row)
        predictions.extend({"source_row": int(df.iloc[ix].source_row), "canonical_pair": keys.iloc[ix],
                            "fold": split_id, "model": model_name, "observed_K": y[ix],
                            "predicted_K": value} for ix, value in zip(te, pred))

    prepared = {}
    for model_name, block in blocks.items():
        if model_name == "B5_residual_plus_nonideal":
            continue
        imp = SimpleImputer(strategy="median")
        ntr, nte = imp.fit_transform(block.iloc[tr]), imp.transform(block.iloc[te])
        scaler = StandardScaler()
        prepared[model_name] = (scaler.fit_transform(np.hstack([ntr, ztr])),
                                scaler.transform(np.hstack([nte, zte])))
    prepared["B5_residual_plus_nonideal"] = prepared["B4_direct_plus_nonideal"]

    for model_name in blocks:
        xtr, xte = prepared[model_name]
        target = y[tr] - hard[tr] if model_name.startswith("B5_") else y[tr]
        model = HistGradientBoostingRegressor(random_state=SEED + split_id, **MODEL_CONFIG)
        model.fit(xtr, target)
        pred = model.predict(xte) + (hard[te] if model_name.startswith("B5_") else 0)
        row = {"split": "canonical_pair_disjoint", "split_id": split_id, "model": model_name}
        row.update(metrics(y[te], pred, keys.iloc[te])); rows.append(row)
        predictions.extend({"source_row": int(df.iloc[ix].source_row), "canonical_pair": keys.iloc[ix],
                            "fold": split_id, "model": model_name, "observed_K": y[ix],
                            "predicted_K": value} for ix, value in zip(te, pred))
    return rows, predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, cols, keys, text, blocks, ideal = prepare_data()
    groups = pd.factorize(keys)[0]
    rows, predictions = [], []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, df.T_exp, groups)):
        r, p = fit_split(df, keys, text, blocks, ideal, tr, te, fold)
        rows.extend(r); predictions.extend(p)
        print(f"completed fold {fold + 1}/5", flush=True)

    fm = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    fm.to_csv(OUT / "UNIFIED_HARDMAX_ABLATION_FOLD_METRICS.csv", index=False)
    pred.to_csv(OUT / "UNIFIED_HARDMAX_ABLATION_OOF.csv", index=False)
    summary = fm.groupby("model", as_index=False).agg(
        folds=("split_id", "nunique"), records_mean=("records", "mean"), pairs_mean=("pairs", "mean"),
        record_MAE_K_mean=("record_MAE_K", "mean"), record_MAE_K_SD=("record_MAE_K", "std"),
        pair_macro_MAE_K_mean=("pair_macro_MAE_K", "mean"), RMSE_K_mean=("RMSE_K", "mean"),
        R2_mean=("R2", "mean"),
    )
    summary.to_csv(OUT / "UNIFIED_HARDMAX_ABLATION_SUMMARY.csv", index=False)

    wide = pred.pivot(index="source_row", columns="model", values="predicted_K")
    obs = pred.drop_duplicates("source_row").set_index("source_row")[["observed_K", "canonical_pair"]]
    wide = wide.join(obs)
    gain = wide.assign(gain=np.abs(wide.B4_direct_plus_nonideal - wide.observed_K) -
                            np.abs(wide.B5_residual_plus_nonideal - wide.observed_K)).groupby("canonical_pair").gain.mean()
    rng = np.random.default_rng(20260916)
    boot = np.array([rng.choice(gain.to_numpy(), len(gain), replace=True).mean() for _ in range(20000)])
    contrast = {"pair_mean_B4_minus_B5_gain_K": float(gain.mean()), "pairs": len(gain),
                "bootstrap_95_low_K": float(np.quantile(boot, .025)),
                "bootstrap_95_high_K": float(np.quantile(boot, .975))}
    (OUT / "UNIFIED_HARDMAX_B4_B5_CONTRAST.json").write_text(json.dumps(contrast, indent=2), encoding="utf-8")
    manifest = {
        "status": "COMPLETE", "training_file": str(DATA), "training_sha256": sha256(DATA),
        "admitted_rows": len(df), "canonical_pairs": keys.nunique(), "split": "5-fold GroupKFold canonical unordered pair",
        "physical_baseline": "experiment-first properties; deployment-matched hard max; gamma evaluated at 298.15 K",
        "model_config": MODEL_CONFIG, "B4_B5_exact_same_preprocessed_matrix": True,
        "definitions": {"B0": "ideal hard-max SLE", "B1": "nonideal hard-max COSMO-SLE",
                        "B2": "structure+composition direct ML", "B3": "B2+pure properties+ideal SLE direct ML",
                        "B4": "B3+nonideal hard-max features direct ML", "B5": "same matrix as B4; residual target"},
    }
    (OUT / "UNIFIED_HARDMAX_ABLATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(contrast, indent=2))


if __name__ == "__main__":
    main()
