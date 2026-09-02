# Neural Symbolic Regression (NSR)

A neural-symbolic framework for discovering **sparse, interpretable mathematical expressions** from data by combining neural representation learning with sparse regression.

The central idea is to use a neural network as a **functional preconditioner**: instead of directly searching the symbolic space, the method first learns a smooth approximation in a nonlinear feature space and subsequently performs sparse equation recovery using LASSO.

---

## Overview

Symbolic Regression (SR) aims to recover an analytical function

$$
y = f(x_1,x_2,\ldots,x_d)
$$

directly from observed data.

Traditional symbolic regression methods such as Genetic Programming can become computationally expensive as the search space grows, while sparse methods such as SINDy depend strongly on the choice of a predefined feature library.

This project investigates a **Neural Symbolic Regression (NSR)** pipeline:

$$
X
\rightarrow
\Phi(X)
\rightarrow
\text{Neural Network}
\rightarrow
\hat{y}
$$

followed by sparse equation recovery:

$$
\hat{y} \approx \Phi(X)\boldsymbol{\beta}
$$

where:

* \(X\) — input observations
* \(\Phi(X)\) — nonlinear feature library
* Neural Network — learns a robust functional representation
* \(\boldsymbol{\beta}\) — sparse symbolic coefficients

The final expression is reconstructed from the non-zero coefficients.

---

## Method

### 1. Nonlinear Feature Library

The input variables are transformed using a predefined library of nonlinear functions:

$$
\Phi(X) =
[
x,\sin(x),\cos(x),e^x,x^2,\tanh(x),
\log(1+x),x^3,\ldots
]
$$

The implementation supports interaction terms such as

$$
\sin(x_1)\cos(x_2)
$$

and higher-order combinations controlled by `max_order`.

The default library is:

```text
id
sin
cos
exp
square
tanh
log1p
cube
```

---

### 2. Neural Functional Preconditioning

The generated feature matrix is given to a multilayer perceptron:

$$
\hat{y}=f_\theta(\Phi(X))
$$

The network is trained by minimizing mean squared error:

$$
\mathcal{L}(\theta) =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

The implementation supports:

* ReLU, GELU and Tanh activations
* configurable depth and hidden dimensions
* dropout
* weight decay
* gradient clipping
* early stopping
* OneCycleLR / StepLR scheduling
* CUDA, Apple MPS and CPU execution
* automatic mixed precision on CUDA

The neural model therefore acts as a **smooth nonlinear preconditioner** before symbolic recovery.

---

### 3. Sparse Equation Recovery

After constructing the feature library, sparse coefficients are recovered using LASSO:

$$
\boldsymbol{\beta}^{*} =
\arg\min_{\boldsymbol{\beta}}
\left[
\frac{1}{2N}
\|y-\Phi\boldsymbol{\beta}\|_2^2
+
\alpha\|\boldsymbol{\beta}\|_1
\right]
$$

The \(L_1\) penalty encourages many coefficients to become exactly zero, producing a compact symbolic expression.

The implementation also supports Elastic Net:

$$
\mathcal{L}
=
\frac{1}{2N}\|y-\Phi\beta\|_2^2
+
\alpha
\left[
\rho\|\beta\|_1
+
\frac{1-\rho}{2}\|\beta\|_2^2
\right]
$$

The resulting coefficients are converted into a SymPy expression.

---

## Hyperparameter Optimization

The framework optionally integrates **Ray Tune** for distributed hyperparameter optimization.

The search space includes:

```text
hidden       : [64, 128, 256]
depth        : [1, 2, 3, 4]
dropout      : [0.0, 0.1, 0.2, 0.3]
activation   : [relu, gelu, tanh]
learning rate
weight decay
batch size   : [64, 128, 256]
epochs       : [100, 150, 200]
```

Configurations are evaluated using validation RMSE:

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

Ray Tune is used to identify a high-performing neural configuration before retraining the final model.

---

## Benchmark

Experiments use the **Nguyen symbolic regression benchmark suite**, containing functions involving:

* polynomial relationships
* trigonometric functions
* exponentials
* logarithms
* multivariate interactions
* mixed nonlinear structures

Example:

$$
f(x)=\sin(x)+\sin(x+x^2)
$$

and

$$
f(x,y)=xy+\sin(x)\cos(y)
$$

The benchmark generator is implemented directly using SymPy.

---

## Experimental Evaluation

The repository contains experiments for:

### Baseline comparison

Comparison between:

* SINDy/LASSO
* Baseline Neural Network
* Tuned Neural Network

### Symbolic recovery

The extracted equation is evaluated against the ground-truth symbolic support using:

* Precision
* Recall
* F1 score
* Number of recovered terms

### Noise robustness

The framework evaluates performance under increasing levels of Gaussian noise.

### Out-of-distribution evaluation

Models are additionally evaluated outside their training distribution to assess generalization.

### Ablation studies

The repository includes ablations over:

* feature-library configuration
* interaction terms
* maximum interaction order
* neural depth
* hidden dimension
* activation
* equation extractor
* learning-rate scheduler

---

# Repository Structure

```text
Neural-Symbollic-Regression-v1/
│
├── nsr_main.py                 # Main NSR pipeline
├── nsr_ablation.py             # Hyperparameter / architecture ablations
├── noise_ood_analysis.py       # Noise robustness and OOD experiments
│
├── requirements.txt            # Python dependencies
├── LICENSE
├── README.md
│
├── outputs/
│   └── <experiment_id>/
│       ├── checkpoint.pkl
│       └── hall_of_fame.csv
│
├── ablation_outputs/
│   ├── ablation_results_full.csv
│   ├── agg_rmse_by_config.png
│   ├── precision_recall_scatter.png
│   └── per_benchmark_json/
│
├── noise_ood_outputs/
│   ├── rmse_vs_noise_nguyen4.csv
│   ├── rmse_vs_noise_nguyen4.png
│   └── *.png
│
├── noise_ood_plots/
│   └── generated experiment plots
│
├── nsr_training_curves.png
├── nsr_rmse_bar.png
└── nsr_pred_scatter.png
```

---

# Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Neural-Symbollic-Regression-v1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For Ray Tune:

```bash
pip install "ray[tune]"
```

PyTorch should be installed according to your hardware configuration.

---

# Running the Main NSR Pipeline

The primary entry point is:

```bash
python nsr_main.py
```

> If you rename `nsr_main.py` to `nsr.py`, use:
>
> ```bash
> python nsr.py
> ```

The default pipeline performs the following steps:

```text
Nguyen-10 benchmark
        │
        ▼
Generate synthetic data
        │
        ▼
Train / validation split
        │
        ▼
Construct nonlinear feature library
        │
        ▼
Train baseline neural model
        │
        ▼
LASSO / SINDy baseline
        │
        ▼
Sparse equation extraction
        │
        ▼
Ray Tune hyperparameter search
        │
        ▼
Retrain tuned neural model
        │
        ▼
Generate evaluation plots
```

The default experiment uses:

```text
Benchmark       : Nguyen-10
Samples         : 2000
Noise           : 0.01
Validation      : 20%
Hidden units    : 128
Depth           : 2
Activation      : GELU
Dropout         : 0.1
Learning rate   : 1e-3
Batch size      : 128
Epochs          : 200
Scheduler       : OneCycleLR
```

---

# Expected Outputs

After running:

```bash
python nsr_main.py
```

the pipeline reports validation RMSE for the neural and sparse baselines and prints the recovered symbolic equation.

Example output structure:

```text
Validation RMSE (Baseline Neural): ...

Validation RMSE (SINDy/LASSO): ...

Extracted equation (approx):
...

Best tuning config:
...

Best val RMSE:
...

Validation RMSE (Tuned Neural):
...
```

The following figures are generated:

### Training curves

```text
nsr_training_curves.png
```

Shows training and validation MSE across epochs.

### RMSE comparison

```text
nsr_rmse_bar.png
```

Compares validation RMSE between SINDy, baseline NSR and tuned NSR.

### Prediction scatter

```text
nsr_pred_scatter.png
```

Compares predicted values against ground-truth values.

---

# Running Ablation Experiments

Ablation experiments are implemented in:

```bash
nsr_ablation.py
```

Run:

```bash
python nsr_ablation.py
```

A faster configuration can be used when supported by the script:

```bash
python nsr_ablation.py --quick
```

Results are stored under:

```text
ablation_outputs/
```

including aggregated RMSE, precision-recall analysis and per-benchmark results.

---

# Noise and OOD Analysis

Noise robustness and out-of-distribution experiments are implemented in:

```bash
noise_ood_analysis.py
```

Run:

```bash
python noise_ood_analysis.py --main nsr_main.py
```

For a faster experiment:

```bash
python noise_ood_analysis.py --main nsr_main.py --quick
```

Generated results are stored in:

```text
noise_ood_outputs/
```

including:

```text
rmse_vs_noise_nguyen4.csv
rmse_vs_noise_nguyen4.png
noise_example_curve_*.png
ood_nguyen4_scatter.png
ood_nguyen4_line.png
```

---

# Research Pipeline Summary

The complete framework can be summarized as:

$$
\boxed{
X
\rightarrow
\Phi(X)
\rightarrow
f_\theta(\Phi(X))
\rightarrow
\text{Sparse Regression}
\rightarrow
\hat{f}(X)
}
$$

where the neural network provides a learned nonlinear functional representation and sparse regression converts the learned representation into an interpretable mathematical equation.

The key research hypothesis is that **neural functional preconditioning can improve the robustness of symbolic discovery without sacrificing interpretability**.

---

## Limitations

The current implementation relies on a predefined nonlinear function library. Therefore, it does not perform completely unconstrained symbolic search.

The quality of symbolic recovery remains dependent on:

* feature-library completeness
* interaction order
* LASSO regularization
* neural architecture
* training stability

Future extensions can investigate adaptive library construction, search-space routing, and more computationally efficient symbolic search.

---

## Citation

If you use this implementation in academic work, please cite the associated research paper:

```bibtex
@article{neural_symbolic_regression,
  title   = {Neural Symbolic Regression through Functional Preconditioning and Sparse Equation Recovery},
  authors  = {Ravi Kumar U, Sumitra S},
  journal = {https://arxiv.org/abs/2609.01102},
  year    = {2026}
}
```

---

## License

This project is released under the license included in the repository.
