"""
    python noise_ood_analysis.py --main nsr_main.py [--out OUT_DIR] [--quick] [--seed SEED]   
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import sympy as sp
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pysr import PySRRegressor


OUT_DIR_DEFAULT = Path("noise_ood_plots")
OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
NGUYEN_ID = 9


def import_main_module(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"main.py not found at: {p}")
    sha1 = hashlib.sha1(str(p).encode()).hexdigest()[:8]
    mod_name = f"nsr_main_{p.stem}_{sha1}"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

def safe_log1p(x):
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, -0.999999, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log1p(arr)
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out

FALLBACK_SAFE_FUNCS = {
    "id": lambda x: x,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "square": np.square,
    "tanh": np.tanh,
    "log1p": safe_log1p,
    "cube": lambda x: x**3
}

def build_fallback_library(X: np.ndarray, funcs=None, include_interactions=True, max_order=2):
    if funcs is None:
        funcs = list(FALLBACK_SAFE_FUNCS.keys())
    if X.ndim == 1:
        X = X[:, None]
    n_samples, n_features = X.shape
    terms = []
    names = []
    for fname in funcs:
        f = FALLBACK_SAFE_FUNCS[fname]
        for i in range(n_features):
            try:
                t = f(X[:, i])
            except Exception:
                t = np.zeros(n_samples)
            terms.append(np.asarray(t, dtype=np.float32))
            names.append(f"{fname}(x{i})")

    if include_interactions and n_features > 1 and max_order >= 2:
        from itertools import combinations, product
        idxs = range(n_features)
        func_idxs = range(len(funcs))
        for combo in combinations(idxs, 2):
            for fpair in product(func_idxs, repeat=2):
                term = np.ones(n_samples, dtype=np.float32)
                name_parts = []
                for idx, fidx in zip(combo, fpair):
                    f = FALLBACK_SAFE_FUNCS[funcs[fidx]]
                    term = term * np.asarray(f(X[:, idx]), dtype=np.float32)
                    name_parts.append(f"{funcs[fidx]}(x{idx})")
                terms.append(term)
                names.append("*".join(name_parts))
    Phi = np.stack(terms, axis=1) if terms else np.zeros((n_samples,0), dtype=np.float32)
    Phi = np.nan_to_num(Phi, nan=0.0, posinf=1e6, neginf=-1e6)
    return Phi.astype(np.float32), names


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, depth=2, out_dim=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(max(0, depth-1)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_simple_mlp(Phi_tr, y_tr, Phi_va=None, y_va=None, hidden=64, depth=2, lr=1e-3, batch_size=128, epochs=120, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model = SimpleMLP(Phi_tr.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xtr = torch.tensor(Phi_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    ds = torch.utils.data.TensorDataset(Xtr, Ytr)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        ypred_va = model(torch.tensor(Phi_va, dtype=torch.float32, device=device)).cpu().numpy() if Phi_va is not None else None
    return model, ypred_va


x = sp.symbols("x")
expr_ng4 = sp.sin(x) + sp.sin(x + x**2)
f_ng4 = sp.lambdify([x], expr_ng4, modules=["numpy"])


def prepare_data_for_noise(expr_func, domain: Tuple[float,float], n_samples:int, noise:float, seed:int=1337):
    rng = np.random.default_rng(seed)
    X = rng.uniform(domain[0], domain[1], size=(n_samples,1))
    y = np.squeeze(expr_func(X[:,0]))
    if noise > 0:
        y = y + rng.normal(0, noise, size=y.shape)
    return X, y

def compute_lasso_predict(Phi_tr, y_tr, Phi_va):
    try:
        model = Lasso(alpha=1e-3, max_iter=10000)
        model.fit(Phi_tr, y_tr)
        return model.predict(Phi_va), model.coef_
    except Exception:
        return np.zeros(Phi_va.shape[0]), np.zeros(Phi_tr.shape[1])
    
def compute_pysr_predict(Phi_tr, y_tr, Phi_va):
    try:
        model = PySRRegressor(
        niterations=1000,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        )
        model.fit(Phi_tr, y_tr)
        return model.predict(Phi_va), model.coef_
    except Exception:
        return np.zeros(Phi_va.shape[0]), np.zeros(Phi_tr.shape[1])



def run_noise_experiments(use_main_module: Optional[Any],
                          out_dir: Path,
                          noise_levels=None,
                          sample_n:int=1500,
                          quick:bool=False,
                          seed:int=1337):
    out_dir.mkdir(parents=True, exist_ok=True)
    if noise_levels is None:
        noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    if quick:
        noise_levels = noise_levels[:4]
        sample_n = min(sample_n, 800)
    records = []

    example_noise_levels = [noise_levels[0], noise_levels[len(noise_levels)//2], noise_levels[-1]]
    for noise in noise_levels:
        X, y = prepare_data_for_noise(f_ng4, (-1,1), sample_n, noise, seed=seed)

        rng = np.random.default_rng(seed)
        idx = np.arange(len(X)); rng.shuffle(idx)
        cut = int(0.8 * len(X))
        tr, va = idx[:cut], idx[cut:]
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        if use_main_module and hasattr(use_main_module, "FeatureLibrary "):
            try:
                LibCfg = getattr(use_main_module, "LibraryConfig")
                FeatureLibrary = getattr(use_main_module, "FeatureLibrary")
                lib_cfg = LibCfg(funcs=["id","sin","cos","exp","square","tanh","log1p","cube"],
                                 include_interactions=True, max_order=2, n_jobs=-1)
                library = FeatureLibrary(lib_cfg)
                Phi_tr, names = library.transform(Xtr)
                Phi_va, _ = library.transform(Xva)
                use_project_lib = True
            except Exception:
                Phi_tr, names = build_fallback_library(Xtr)
                Phi_va, _ = build_fallback_library(Xva)
                use_project_lib = False
        else:
            Phi_tr, names = build_fallback_library(Xtr)
            Phi_va, _ = build_fallback_library(Xva)
            use_project_lib = False

        if use_main_module and hasattr(use_main_module, "NeuralSRModel") and hasattr(use_main_module, "Trainer"):
            try:

                ModelCfg = getattr(use_main_module, "ModelConfig")
                TrainCfg = getattr(use_main_module, "TrainingConfig")
                NeuralSRModel = getattr(use_main_module, "NeuralSRModel")
                Trainer = getattr(use_main_module, "Trainer")
                mcfg = ModelCfg(hidden=128, depth=2, dropout=0.1, activation="gelu", weight_decay=1e-4)
                model = NeuralSRModel(mcfg)
                model.build(Phi_tr.shape[1])
                tcfg = TrainCfg(lr=1e-3, batch_size=128, epochs=(60 if quick else 160), early_stop_patience=30, scheduler="onecycle")
                trainer = Trainer(model, tcfg)
                trainer.fit(Phi_tr, ytr, Phi_va, yva, verbose=False)
                ypred_va = model.predict_from_phi(Phi_va)
                neural_model = model
                used_project_trainer = True
            except Exception:
                neural_model, ypred_va = train_simple_mlp(Phi_tr, ytr, Phi_va, y_va=yva, hidden=64, depth=2, epochs=(60 if quick else 120))
                used_project_trainer = False
        else:
            neural_model, ypred_va = train_simple_mlp(Phi_tr, ytr, Phi_va, y_va=yva, hidden=64, depth=2, epochs=(60 if quick else 120))
            used_project_trainer = False

        yhat_sindy, sindy_coefs = compute_lasso_predict(Phi_tr, ytr, Phi_va)
        yhat_pysr, pysr_coefs = compute_pysr_predict(Phi_tr, ytr, Phi_va)
        rmse_neu = math.sqrt(mean_squared_error(yva, ypred_va))
        rmse_sin = math.sqrt(mean_squared_error(yva, yhat_sindy))
        rmse_pysr = math.sqrt(mean_squared_error(yva, yhat_pysr))
        records.append({"noise": noise, "rmse_neural": rmse_neu, "rmse_sindy": rmse_sin, "rmse_pysr": rmse_pysr})

        if noise in example_noise_levels:
            x_grid = np.linspace(-1,1,1200).reshape(-1,1)
            y_true_grid = np.squeeze(f_ng4(x_grid[:,0]))
            if use_project_lib and 'library' in locals():
                try:
                    Phi_grid, _ = library.transform(x_grid)
                except Exception:
                    Phi_grid, _ = build_fallback_library(x_grid)
            else:
                Phi_grid, _ = build_fallback_library(x_grid)

            if used_project_trainer and hasattr(neural_model, "predict_from_phi"):
                y_grid_neu = neural_model.predict_from_phi(Phi_grid)
            else:
                neural_model.eval()
                with torch.no_grad():
                    y_grid_neu = neural_model(torch.tensor(Phi_grid, dtype=torch.float32)).cpu().numpy()

            lasso_tmp = Lasso(alpha=1e-3, max_iter=10000)
            lasso_tmp.fit(Phi_tr, ytr)
            y_grid_sindy = lasso_tmp.predict(Phi_grid)
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(x_grid[:,0], y_true_grid, label="True", linewidth=2)
            ax.plot(x_grid[:,0], y_grid_neu, label=f"Neural (noise={noise})", linewidth=1.5)
            ax.plot(x_grid[:,0], y_grid_sindy, label=f"SINDy (noise={noise})", linewidth=1.0, linestyle="--")
            ax.scatter(Xva[:,0], yva, s=10, alpha=0.4, label="val samples")
            ax.set_title(f"Noise example (Nguyen-4) noise={noise:.3f}")
            ax.legend()
            fig.savefig(out_dir / f"noise_example_curve_n{noise:.3f}.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

    import pandas as pd
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df['noise'], df['rmse_neural'], marker='o', label='Neural Baseline')
    ax.plot(df['noise'], df['rmse_sindy'], marker='s', label='SINDy (LASSO)')
    ax.set_xscale('log')
    ax.set_xlabel('Noise std (log scale)')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('RMSE vs Noise - Nguyen-4')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(out_dir / "rmse_vs_noise_nguyen4.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    df.to_csv(out_dir / "rmse_vs_noise_nguyen4.csv", index=False)
    return df


def run_ood_evaluation(use_main_module: Optional[Any], out_dir: Path, quick:bool=False, seed:int=1337):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_train = 1200 if quick else 2000
    n_test = 1200 if quick else 2000
    Xtr = rng.uniform(-1,1,size=(n_train,1))
    ytr = np.squeeze(f_ng4(Xtr[:,0])) + rng.normal(0, 0.01, size=n_train)
    Xtest = np.linspace(-3,3,n_test).reshape(-1,1)
    ytest = np.squeeze(f_ng4(Xtest[:,0]))

    if use_main_module and hasattr(use_main_module, "FeatureLibrary"):
        try:
            LibCfg = getattr(use_main_module, "LibraryConfig")
            FeatureLibrary = getattr(use_main_module, "FeatureLibrary")
            lib_cfg = LibCfg(funcs=["id","sin","cos","exp","square","tanh","log1p","cube"],
                             include_interactions=True, max_order=2, n_jobs=-1)
            library = FeatureLibrary(lib_cfg)
            Phi_tr, _ = library.transform(Xtr)
            Phi_test, _ = library.transform(Xtest)
            used_project_lib = True
        except Exception:
            Phi_tr, _ = build_fallback_library(Xtr)
            Phi_test, _ = build_fallback_library(Xtest)
            used_project_lib = False
    else:
        Phi_tr, _ = build_fallback_library(Xtr)
        Phi_test, _ = build_fallback_library(Xtest)
        used_project_lib = False

    if use_main_module and hasattr(use_main_module, "NeuralSRModel") and hasattr(use_main_module, "Trainer"):
        try:
            ModelCfg = getattr(use_main_module, "ModelConfig")
            TrainCfg = getattr(use_main_module, "TrainingConfig")
            NeuralSRModel = getattr(use_main_module, "NeuralSRModel")
            Trainer = getattr(use_main_module, "Trainer")
            mcfg = ModelCfg(hidden=128, depth=2, dropout=0.1, activation="gelu", weight_decay=1e-4)
            model = NeuralSRModel(mcfg)
            model.build(Phi_tr.shape[1])
            tcfg = TrainCfg(lr=1e-3, batch_size=128, epochs=(100 if quick else 200), early_stop_patience=30, scheduler="onecycle")
            trainer = Trainer(model, tcfg)
            trainer.fit(Phi_tr, ytr, Phi_val=Phi_test, y_val=ytest, verbose=False) if "Phi_val" in trainer.fit.__code__.co_varnames else trainer.fit(Phi_tr, ytr, Phi_test, ytest, verbose=False)
            ypred_test = model.predict_from_phi(Phi_test)
            neural_model = model
            used_project_trainer = True
        except Exception:
            neural_model, _ = train_simple_mlp(Phi_tr, ytr, Phi_va=Phi_test, y_va=ytest, hidden=128, depth=2, epochs=(100 if quick else 200))
            with torch.no_grad():
                neural_model.eval()
                ypred_test = neural_model(torch.tensor(Phi_test, dtype=torch.float32)).cpu().numpy()
            used_project_trainer = False
    else:
        neural_model, _ = train_simple_mlp(Phi_tr, ytr, Phi_va=Phi_test, y_va=ytest, hidden=128, depth=2, epochs=(100 if quick else 200))
        with torch.no_grad():
            neural_model.eval()
            ypred_test = neural_model(torch.tensor(Phi_test, dtype=torch.float32)).cpu().numpy()
        used_project_trainer = False

    yhat_sindy, _ = compute_lasso_predict(Phi_tr, ytr, Phi_test)
    rmse_neu = math.sqrt(mean_squared_error(ytest, ypred_test))
    rmse_sin = math.sqrt(mean_squared_error(ytest, yhat_sindy))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(Xtest[:,0], ytest, label='True', linewidth=2)
    ax.plot(Xtest[:,0], ypred_test, label=f'Neural Pred (RMSE={rmse_neu:.4f})', linewidth=1.5)
    ax.plot(Xtest[:,0], yhat_sindy, label=f'SINDy Pred (RMSE={rmse_sin:.4f})', linewidth=1.0, linestyle='--')
    ax.set_title('OOD Prediction: Nguyen-4 (train [-1,1], test [-3,3])')
    ax.legend()
    fig.savefig(out_dir / "ood_nguyen4_line.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(ytest, ypred_test, s=8, alpha=0.6, label='Neural')
    ax.scatter(ytest, yhat_sindy, s=6, alpha=0.6, label='SINDy')
    mn = min(ytest.min(), ypred_test.min(), yhat_sindy.min())
    mx = max(ytest.max(), ypred_test.max(), yhat_sindy.max())
    ax.plot([mn,mx],[mn,mx], color='k', linewidth=1)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('OOD Predicted vs True (Nguyen-4)')
    ax.legend()
    fig.savefig(out_dir / "ood_nguyen4_scatter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"rmse_neu": rmse_neu, "rmse_sin": rmse_sin, "ytest": ytest, "ypred_test": ypred_test, "yhat_sindy": yhat_sindy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default=None, help="Path to main.py to import (optional).")
    parser.add_argument("--out", type=str, default=str(OUT_DIR_DEFAULT), help="Output directory for plots.")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer samples & epochs).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--no-fallback", action="store_true", help="If set, fail if main import fails.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_module = None
    if args.main:
        try:
            main_module = import_main_module(args.main)
            print(f"Imported main.py from {args.main}; using project components where possible.")
        except Exception as e:
            print(f"Could not import main.py ({args.main}): {e}")
            if args.no_fallback:
                raise
            else:
                print("Falling back to internal implementations.")

    print("Running noise experiments...")
    df_rmse = run_noise_experiments(use_main_module=main_module, out_dir=out_dir, quick=args.quick, seed=args.seed)
    print("Saved RMSE vs noise CSV and plots to", out_dir)

    print("Running OOD evaluation (Nguyen-4)...")
    ood_results = run_ood_evaluation(use_main_module=main_module, out_dir=out_dir, quick=args.quick, seed=args.seed)
    print("OOD RMSE (Neural):", ood_results["rmse_neu"], "SINDy:", ood_results["rmse_sin"])

    print("All plots and CSV are in:", out_dir)
    print("Files:")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)

if __name__ == "__main__":
    main()
