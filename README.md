# cv_unstable_correlations

**cv_unstable_correlations** is the official codebase accompanying our KDD 2025 paper:

> **Cross-Validation for Longitudinal Datasets with Unstable Correlations**  
> *Presented at the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2025*

This package implements a cross-validation-based model selection technique that helps identify hyperparameters leading to **models that rely on stable, rather than unstable, correlations** in longitudinal data.

Standard cross-validation approaches typically split data into folds either randomly or temporally and then average performance across these folds.  However, this averaging can **mask a model’s reliance on unstable correlations**.

> 🔍 **Key Insight**  
> If a model relies on features with a consistent relationship with the outcome across temporally sampled folds ($V = 0$), its average performance will be the same whether folds are sampled randomly or temporally.  
> However, if a model relies on features with inconsistent relationships with outcomes over time ($V > 0$), its average performance will differ depending on the sampling strategy—**revealing a reliance on unstable correlations**.

---

## 📦 Installation

```bash
# From the root of the repository
pip install -e .
````

---

## 🚀 Usage

```python
from cv_unstable_correlations.core import run_cv

# Example inputs:
# X: features (numpy array)
# y: binary or continuous labels (numpy array)
# ts: timestamps (numpy array)
# metric_fn: your metric function (e.g., log_loss or mean_squared_error)

results = run_cv(
    X,
    y,
    ts,
    metric_fn,
    model_type='logreg',  # or 'linreg'
    seed=0
)
```

---

## 🧠 Function Signature

```python
def run_cv(X, y, ts, metric_fn, model_type='logreg', seed=0):
    """
    Evaluate model performance using random CV, block CV, and a stability-aware CV approach.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels (binary for logistic regression, continuous for linear regression).
        ts (np.ndarray): Timestamps or temporal indices for each row of X.
        metric_fn (callable): Scoring function that takes (y_true, y_pred).
        model_type (str): 'logreg' (default) or 'linreg'.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (random_cv_score, block_cv_score, proposed_score)
    """
```

---

## 💻 Demo

A visual demo of our theoretical results is available at:
👉 [https://mld3.github.io/cv\_unstable\_correlations](https://mld3.github.io/cv_unstable_correlations)

This interactive site simulates how standard and proposed CV methods behave when feature–outcome relationships vary over time.

