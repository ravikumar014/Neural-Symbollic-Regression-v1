"""
    python nsr_ablation.py                 # full (but serial) run
    python nsr_ablation.py --quick         # quick debug run (small grid)
    python nsr_ablation.py --main nsr_main.py  # if your main is elsewhere
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def id_fn(x):
    return x

def cube_fn(x):
    return x ** 3

SAFE_FUNC_REGISTRY: Dict[str, Callable] = {
    "id": id_fn,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "square": np.square,
    "tanh": np.tanh,
    "log1p": np.log1p,
    "cube": cube_fn,
}



def import_main_from_path(path: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find file at: {path}")

    sha1 = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    module_name = f"nsr_main_module_{path.stem}_{sha1}"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)


    sys.modules[module_name] = module
    module.__spec__ = spec

    try:
        spec.loader.exec_module(module)
    except Exception:

        try:
            del sys.modules[module_name]
        except Exception:
            pass
        raise

    return module


DEFAULT_MAIN_PY = "/Users/ravikumaru/Desktop/MTech/Data_Modelling_Project/nsr_main.py"


FEATURE_LIBRARIES = {
    "small": ["id", "sin", "cos"],
    "medium": ["id", "sin", "cos", "exp", "tanh"],
    "full": ["id", "sin", "cos", "exp", "square", "tanh", "log1p", "cube"],
}
MAX_ORDERS = [1, 2]
DEPTHS = [1, 2, 3, 4]
HIDDENS = [64, 128, 256]
ACTIVATIONS = ["relu", "gelu", "tanh"]
EXTRACTORS = ["lasso", "elastic"]
SCHEDULERS = ["onecycle", "steplr", "none"]
USE_INTERACTIONS = [True, False]
LASSO_ALPHA = [1e-3]
GT_LASSO_ALPHA = 1e-6
BENCH_IDS = list(range(1, 8)) 


DEFAULT_EPOCHS = 150
BATCH_SIZE = 128
SEED = 1337


OUT_DIR = Path("ablation_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / "ablation_results_full.csv"
JSON_DIR = OUT_DIR / "per_benchmark_json"
JSON_DIR.mkdir(parents=True, exist_ok=True)


def safe_get(module, name: str, required: bool = True):
    if not hasattr(module, name):
        if required:
            raise AttributeError(f"Required symbol '{name}' not found in the imported module.")
        return None
    return getattr(module, name)


def run_single_experiment(nsr, bench_id: int, funcs: List[str], max_order: int,
                          depth: int, hidden: int, activation: str, extractor_name: str,
                          scheduler: str, include_interactions: bool,
                          sample_n: int = 2000, noise: float = 0.01,
                          epochs: int = DEFAULT_EPOCHS, batch_size: int = BATCH_SIZE) -> Dict[str, Any]:
    
    FeatureLibrary = safe_get(nsr, "FeatureLibrary")
    LibraryConfig = safe_get(nsr, "LibraryConfig")
    ModelConfig = safe_get(nsr, "ModelConfig")
    TrainingConfig = safe_get(nsr, "TrainingConfig")
    Benchmarks = safe_get(nsr, "Benchmarks")
    NeuralSRModel = safe_get(nsr, "NeuralSRModel")
    Trainer = safe_get(nsr, "Trainer")
    EquationExtractor = safe_get(nsr, "EquationExtractor")
    set_seed = safe_get(nsr, "set_seed")


    set_seed(SEED)


    expr, vars_ = Benchmarks.nguyen(bench_id)
    X, y = Benchmarks.sample(expr, vars_, n_samples=sample_n, noise=noise, seed=SEED)


    if hasattr(nsr, "train_val_split"):
        X_tr, y_tr, X_va, y_va = nsr.train_val_split(X, y, val_split=0.2, seed=SEED)
    else:

        idx = np.arange(len(X))
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        cut = int(len(X) * 0.8)
        tr, va = idx[:cut], idx[cut:]
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

    lib_cfg = LibraryConfig(funcs=funcs, include_interactions=include_interactions, max_order=max_order, n_jobs=-1)
    library = FeatureLibrary(lib_cfg)


    Phi_tr, names = library.transform(X_tr)
    Phi_va, _ = library.transform(X_va)


    mcfg = ModelConfig(hidden=hidden, depth=depth, dropout=0.1, activation=activation, weight_decay=1e-4)
    model = NeuralSRModel(mcfg)
    model.build(Phi_tr.shape[1])


    tcfg = TrainingConfig(lr=1e-3, batch_size=batch_size, epochs=epochs, early_stop_patience=30, scheduler=scheduler)
    trainer = Trainer(model, tcfg)
    history = trainer.fit(Phi_tr, y_tr, Phi_va, y_va, verbose=False)


    pred_val = model.predict_from_phi(Phi_va)
    rmse_neural = float(math.sqrt(mean_squared_error(y_va, pred_val)))


    try:
        lasso = Lasso(alpha=1e-3, max_iter=10000)
        lasso.fit(Phi_tr, y_tr)
        yhat_sindy = lasso.predict(Phi_va)
        rmse_sindy = float(math.sqrt(mean_squared_error(y_va, yhat_sindy)))
    except Exception:
        rmse_sindy = float("nan")
        yhat_sindy = np.zeros_like(pred_val)


    extractor = EquationExtractor(alpha=LASSO_ALPHA[0], method=extractor_name)
    Phi_all = np.vstack([Phi_tr, Phi_va])
    y_all = np.hstack([y_tr, y_va])
    _, coefs = extractor.fit_predict(Phi_all, y_all)
    n_terms_extracted = int(np.sum(np.abs(coefs) > 1e-6))
    active_terms = extractor.get_active_term_names(coefs, library.last_feature_names, threshold=1e-6) if hasattr(extractor, "get_active_term_names") else []


    gt_support = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                  "tp_names": [], "fp_names": [], "fn_names": [], "n_true": 0}
    if hasattr(nsr, "compute_ground_truth_coeffs_via_library") and hasattr(nsr, "compare_supports"):
        coef_true, true_names = nsr.compute_ground_truth_coeffs_via_library(expr, vars_, library,
                                                                            sample_n=3000, sample_low=-1.0, sample_high=1.0,
                                                                            lasso_alpha=GT_LASSO_ALPHA, seed=SEED)
        if coef_true.size > 0:
            comp = nsr.compare_supports(coef_true, coefs, true_names, threshold=1e-6)
            gt_support = {
                "precision": comp.get("precision", float("nan")),
                "recall": comp.get("recall", float("nan")),
                "f1": comp.get("f1", float("nan")),
                "tp_names": comp.get("tp_names", []),
                "fp_names": comp.get("fp_names", []),
                "fn_names": comp.get("fn_names", []),
                "n_true": comp.get("n_true_nonzero", 0)
            }

    result = {
        "benchmark": f"Nguyen-{bench_id}",
        "funcs": ",".join(funcs),
        "max_order": max_order,
        "include_interactions": include_interactions,
        "depth": depth,
        "hidden": hidden,
        "activation": activation,
        "extractor": extractor_name,
        "scheduler": scheduler,
        "rmse_neural": rmse_neural,
        "rmse_sindy": rmse_sindy,
        "n_terms_extracted": n_terms_extracted,
        "n_true_terms": gt_support["n_true"],
        "precision": gt_support["precision"],
        "recall": gt_support["recall"],
        "f1": gt_support["f1"],
        "tp_names": gt_support["tp_names"],
        "fp_names": gt_support["fp_names"],
        "fn_names": gt_support["fn_names"],
        "history_train": history.get("train", []),
        "history_val": history.get("val", []),
        "active_terms": active_terms,
    }

    return result



def run_full_ablation(nsr,
                      bench_ids: List[int],
                      feature_libs: Dict[str, List[str]],
                      max_orders: List[int],
                      depths: List[int],
                      hiddens: List[int],
                      activations: List[str],
                      extractors: List[str],
                      schedulers: List[str],
                      use_interactions: List[bool],
                      epochs: int = DEFAULT_EPOCHS,
                      sample_n: int = 2000,
                      noise: float = 0.01,
                      quick: bool = False,
                      out_csv: Path = CSV_OUT) -> pd.DataFrame:

    combos = []
    for bench_id in bench_ids:
        for lib_key, funcs in feature_libs.items():
            for max_order in max_orders:
                for include_inter in use_interactions:
                    for depth in depths:
                        for hidden in hiddens:
                            for activation in activations:
                                for extractor in extractors:
                                    for scheduler in schedulers:
                                        combos.append((bench_id, funcs, max_order, depth, hidden, activation, extractor, scheduler, include_inter))

    if quick:
        combos = combos[: min(40, len(combos))]

    print(f"Running {len(combos)} experiments (serial mode). This may take a while.")
    results = []
    t0 = time.time()

    for i, c in enumerate(combos, start=1):
        bench_id, funcs, max_order, depth, hidden, activation, extractor, scheduler, include_inter = c
        print(f"[{i}/{len(combos)}] Bench {bench_id} lib={len(funcs)} order={max_order} depth={depth} hidden={hidden} act={activation} ext={extractor} sched={scheduler} int={include_inter}")
        try:
            res = run_single_experiment(nsr, bench_id, funcs, max_order, depth, hidden, activation, extractor, scheduler, include_inter, sample_n=sample_n, noise=noise, epochs=epochs)
            results.append(res)
        except Exception as e:
            print(f"Experiment failed: {e}")

            results.append({
                "benchmark": f"Nguyen-{bench_id}",
                "funcs": ",".join(funcs),
                "max_order": max_order,
                "include_interactions": include_inter,
                "depth": depth,
                "hidden": hidden,
                "activation": activation,
                "extractor": extractor,
                "scheduler": scheduler,
                "rmse_neural": float("nan"),
                "rmse_sindy": float("nan"),
                "n_terms_extracted": -1,
                "n_true_terms": -1,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "tp_names": [],
                "fp_names": [],
                "fn_names": [],
                "history_train": [],
                "history_val": [],
                "active_terms": [],
                "error": str(e)
            })

        pd.DataFrame(results).to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"Completed {len(results)} experiments in {elapsed/60:.2f} minutes.")
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    return df



def summary_plots(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    if df.empty:
        print("No results to plot.")
        return


    def make_label(r):
        num_funcs = len(r["funcs"].split(",")) if isinstance(r["funcs"], str) else 0
        return f"{num_funcs}f_d{r['depth']}_h{r['hidden']}_{r['activation']}_{'int' if r['include_interactions'] else 'noint'}_{r['extractor']}_{r['scheduler']}"

    df["config_label"] = df.apply(make_label, axis=1)


    agg = df.groupby("config_label")[["rmse_neural"]].mean().sort_values("rmse_neural")
    fig, ax = plt.subplots(figsize=(10, max(4, len(agg)/6)))
    agg["rmse_neural"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg RMSE (neural)")
    ax.set_title("Avg Validation RMSE by Configuration (mean over benchmarks)")
    plt.tight_layout()
    p1 = out_dir / "agg_rmse_by_config.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    print("Saved", p1)


    if "precision" in df.columns and not df["precision"].dropna().empty:
        pr = df.dropna(subset=["precision", "recall"])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pr["precision"], pr["recall"], alpha=0.7)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        ax.set_title("Precision vs Recall (term recovery) across runs")
        ax.grid(True)
        p2 = out_dir / "precision_recall_scatter.png"
        fig.savefig(p2, dpi=160)
        plt.close(fig)
        print("Saved", p2)



def main_cli():
    parser = argparse.ArgumentParser(description="Ablation runner for NSR project (robust serial mode)")
    parser.add_argument("--main", type=str, default=DEFAULT_MAIN_PY, help="Path to main.py implementing the NSR classes/functions")
    parser.add_argument("--out", type=str, default=str(CSV_OUT), help="CSV output path")
    parser.add_argument("--quick", action="store_true", help="Run a short, quick debug grid")
    parser.add_argument("--parallel", action="store_true", help="(NOT RECOMMENDED) Attempt limited parallelism (disabled by default)")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs (only if --parallel is used; may still fail due to pickling)")
    args = parser.parse_args()

    print("Importing main.py from:", args.main)
    nsr = import_main_from_path(args.main)


    required = ["FeatureLibrary", "LibraryConfig", "ModelConfig", "TrainingConfig", "Benchmarks", "NeuralSRModel", "Trainer", "EquationExtractor", "set_seed"]
    missing = [name for name in required if not hasattr(nsr, name)]
    if missing:
        print("WARNING: The imported main.py is missing expected symbols:", missing)
        print("The runner will still attempt to proceed where possible, but many experiments may fail.")


    bench_ids = BENCH_IDS
    feature_libs = FEATURE_LIBRARIES
    max_orders = MAX_ORDERS
    depths = DEPTHS
    hiddens = HIDDENS
    activations = ACTIVATIONS
    extractors = EXTRACTORS
    schedulers = SCHEDULERS
    use_interactions = USE_INTERACTIONS
    epochs = DEFAULT_EPOCHS
    if args.quick:
        bench_ids = bench_ids[:2]
        depths = depths[:2]
        hiddens = hiddens[:2]
        epochs = 60


    df = run_full_ablation(nsr, bench_ids, feature_libs, max_orders, depths, hiddens, activations, extractors, schedulers, use_interactions, epochs=epochs, quick=args.quick, out_csv=Path(args.out))


    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench]
        sub.to_json(JSON_DIR / f"{bench}.json", orient="records", indent=2)


    summary_plots(df, out_dir=OUT_DIR)

    print("Ablation run finished. CSV:", args.out)
    print("Per-benchmark JSONs:", JSON_DIR)
    print("Plots:", OUT_DIR)


if __name__ == "__main__":
    main_cli()
