import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

"""
Multiverse integration methods as implemented in Cantone & Tomaselli (2024).
The equations in the comments refer to the equations in the paper.

Reference:
Cantone, G. G., & Tomaselli, V. (2024). Characterisation and calibration of multiversal methods. 
Advances in Data Analysis and Classification. https://doi.org/10.1007/s11634-024-00610-9

Usage:
integrate(results, measure, method, type)
    - results: DataFrame with multiverse results (from mverse.get_results(as_df=True, expand_dec=True))
    - measure: name of the column to integrate (e.g., "beta", "accuracy", "cohen_d")
    - method:  uniform, bma, myh, myhn, mli       
    - type:    mean, median

compare_methods(data, measure, true_value)
- Compare all integration methods and return a summary table

plot_methods(data, measure, weights, true_value)
- Plot weighted density distributions for each method

Decision Column Types (for myh, myhn, mli):
- Binary: bool (True/False) - converted to 0/1
- Categorical: object/string ("a", "b", "1", "2") - dummy encoded 
- Numerical: int/float - kept as-is (ratio scale)

Sequence Handling
-Supports measures that are scalars (e.g, beta) or sequences (e.g, cross-validation scores).
-If a measure is a sequence, it is aggregated per universe using agg (mean/median/first/last) before integration.

Decision columns must start with '_' (exapmle: _X1, _X2, _method, _threshold).
"""
def _is_sequence(val) -> bool:
    # checking if value is a sequence (list or array), not a scalar
    return isinstance(val, (list, np.ndarray))


def reduce_sequence(series: pd.Series, method: str = "mean") -> np.ndarray:

    reducers = {
        "mean": lambda x: np.mean(x),
        "median": lambda x: np.median(x),
        "first": lambda x: x[0],
        "last": lambda x: x[-1],
    }
    
    if method not in reducers:
        raise ValueError(f"agg must be one of {list(reducers.keys())}")
    
    return series.apply(reducers[method]).to_numpy(dtype=float)


def get_measure_values(data: pd.DataFrame, measure: str, agg: str = "mean") -> np.ndarray:
    # get measure values, automatically handling sequences.
    #agg : str
    #Aggregation method if column contains sequences.
    
    series = data[measure]
    
    # check if first value is a sequence
    if _is_sequence(series.iloc[0]):
        return reduce_sequence(series, method=agg)
    else:
        return series.to_numpy(dtype=float)

def get_decision_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # find columns starting with '_'::covers both '_' and '__'
    decision_cols = [col for col in data.columns if col.startswith('_')]
    
    if not decision_cols:
        raise ValueError(
            "No decision columns found. Expected columns starting with '_' "
            "(e.g., _X1, _X2, _method). Use expand_dec=True when getting results."
        )
    
    
    decision_cols = sorted(decision_cols)
    
    result_parts = []
    col_types = []
    
    for col in decision_cols:
        if data[col].dtype == bool:
            # Binary: True/False - 0/1
            result_parts.append(data[[col]].astype(int))
            col_types.append('binary')
        elif data[col].dtype == object:
            # Categorical: dummy encoding 
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            result_parts.append(dummies)
            col_types.extend(['categorical'] * dummies.shape[1])
        else:
            # Numerical: ratio scale
            result_parts.append(data[[col]])
            col_types.append('numeric')
    
    matrix = pd.concat(result_parts, axis=1).to_numpy(dtype=float)
    
    return matrix, col_types


def gower_distance(Q: np.ndarray, col_types: list[str]) -> np.ndarray:
    #Binary/Categorical: Hamming distance (0 or 1 per feature)
    #Numeric: Normalized Manhattan distance (|x - y| / range)
    n = len(Q)
    dist = np.zeros((n, n))
    
    for k, ctype in enumerate(col_types):
        if ctype in ('binary', 'categorical'):
            # Hamming: 0 or 1
            dist += (Q[:, None, k] != Q[None, :, k]).astype(float)
        else:
            # Numeric: normalized absolute difference
            rng = Q[:, k].max() - Q[:, k].min()
            if rng > 0:
                dist += np.abs(Q[:, None, k] - Q[None, :, k]) / rng
    
    return dist


def gini(w: np.ndarray) -> float:
    # Gini coefficient of weights
    w = np.sort(w)
    n = len(w)
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * w) / (n * w.sum())) - (n + 1) / n)


def integrate(results, measure=None, method="uniform", type="median", agg="mean"):
    """
    Integrate the multiverse results.

    Parameters
    results : pd.DataFrame
        DataFrame containing the multiverse results.
    measure : str
        Name of the measure to integrate ( "beta", "accuracy", "cohen_d").
    method : str
        Method to use for integration. Options are:
            - "uniform" (default): Simple mean/median across all universes
            - "bma": Bayesian model averaging (requires 'bic' column)
            - "myh": Multiverse yield heterogeneity (positive weights)
            - "myhn": Multiverse yield negative heterogeneity (inverse weights)
            - "mli": Multiverse local instability
    type : str
        Type of (weighted) integration. Options are "median" (default) or "mean".
    """
    # work on a copy to avoid modifying original
    results = results.copy()
    
    # Convert columns to lowercase
    results.columns = results.columns.str.lower()
    measure = measure.lower() if measure else None

    # Initial checks
    if measure is None:
        raise ValueError("Please provide a measure to integrate.")
    if measure not in results.columns:
        raise ValueError(f"The measure '{measure}' was not found in the results.")
    if method == "bma" and "bic" not in results.columns:
        raise ValueError("BMA weights require a 'bic' column in the results.")
    
    x = get_measure_values(results, measure, agg=agg)
  

    # Compute weights based on method
    if method == "uniform":
        weights = uniform_weights(x)
    elif method == "bma":
        weights = bma_weights(results)
    elif method == "myh":
        weights = myh_weights(results, measure=measure, agg=agg, negative=False)
    elif method == "myhn":
        weights = myh_weights(results, measure=measure, agg=agg, negative=True)
    elif method == "mli":
        weights = mli_weights(results, measure=measure, agg=agg)
    else:
        raise ValueError("method must be 'uniform', 'bma', 'myh', 'myhn', or 'mli'")
    
    # Compute integrated estimate
    if type == "mean":
        integrated_estimate = weighted_mean(x, weights)
    elif type == "median":
        integrated_estimate = weighted_median(x, weights)
    else:
        raise ValueError("type must be 'mean' or 'median'")
    
    return integrated_estimate, weights


# Aggregation methods
def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x))


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]

    cw = np.cumsum(w_sorted)
    return float(x_sorted[np.searchsorted(cw, 0.5, side="left")])


# Weight functions
def uniform_weights(data) -> np.ndarray:
    #Compute uniform weights for all universes (w >= 0 and w.sum() == 1).
    n = len(data)
    return np.full(n, 1.0 / n, dtype=float)


def bma_weights(data: pd.DataFrame) -> np.ndarray:
  
    bic_col = next((c for c in data.columns if c.lower() == "bic"), None)
    if bic_col is None:
        raise KeyError("The DataFrame must contain a 'bic' column (case-insensitive).")

    bic = data[bic_col].to_numpy(float)
    delta = bic - np.min(bic)  # for numerical stability
    w = np.exp(-0.5 * delta)   
    return w / w.sum()         


def myh_weights(data: pd.DataFrame, measure: str, agg: str = "mean", negative: bool = False) -> np.ndarray:

    # Get measure values
    if measure not in data.columns:
        raise KeyError(f"The DataFrame must contain a '{measure}' column.")
    
    y = get_measure_values(data, measure, agg=agg)

    # Get decision matrix (handles binary, categorical, numeric)
    X_dec, _ = get_decision_matrix(data)  # (n, k)
    
    # OLS regression: measure ~ constant + decision_features
    X = sm.add_constant(X_dec, has_constant="add")
    fit = sm.OLS(y, X).fit()
    alpha = float(fit.params[0])
    beta_q = fit.params[1:]  # length k
    
    
    Cj = np.sqrt(alpha**2 + (X_dec * (beta_q**2)).sum(axis=1))
    
    # Compute weights
    num = (Cj.max() - Cj) if negative else Cj
    s = num.sum()
    n = len(num)
    return np.full(n, 1.0 / n) if s == 0 else num / s


def mli_weights(data: pd.DataFrame, measure: str, agg: str = "mean") -> np.ndarray:
    # Get measure values
    if measure not in data.columns:
        raise KeyError(f"The DataFrame must contain a '{measure}' column.")
    
    y = get_measure_values(data, measure, agg=agg)
    
    # Get decision matrix and column types
    Q, col_types = get_decision_matrix(data)  # (n, k)
    
    # Compute Gower distance matrix
    dist = gower_distance(Q, col_types)
    n = len(y)
    
    # Compute local instability
    # "neighbor" = minimum non-zero Gower distance
    err = np.zeros(n, dtype=float)
    for i in range(n):
        # Find minimum non-zero distance (closest neighbors)
        nonzero_dist = dist[i][dist[i] > 0]
        if nonzero_dist.size:
            min_dist = nonzero_dist.min()
            # Neighbors are those at minimum distance (with small tolerance)
            neigh = np.where(np.abs(dist[i] - min_dist) < 1e-9)[0]
            if neigh.size:
                err[i] = np.max(np.abs(y[i] - y[neigh]))
    
    # Inverse: lower instability = higher weight
    num = err.max() - err
    s = num.sum()
    return np.full(n, 1.0 / n) if s == 0 else num / s


# Comparison functions
def compare_methods(
    data: pd.DataFrame,
    measure: str,
    true_value: float | None = None,
    agg: str = "mean"
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compare all integration methods and return a summary table.
    
    Returns a table with both weighted median and weighted mean for each method.
    
    Columns:
    - Scheme: Name of the weighting method
    - median: Weighted median estimate
    - mean: Weighted mean estimate
    - gini_w: Gini coefficient of weights (0=uniform, 1=concentrated)
    - maxw_estimate: Estimate from the universe with maximum weight
    - err_median, err_mean, err_maxw: Errors vs true_value (if provided)
    """
    # Work on a copy with lowercase columns
    data = data.copy()
    data.columns = data.columns.str.lower()
    measure = measure.lower()
    
    y = get_measure_values(data, measure, agg=agg)
    
    schemes = [
        ("Uniform", "uniform"),
        ("BMA", "bma"),
        ("MYH", "myh"),
        ("MYHN", "myhn"),
        ("MLI", "mli"),
    ]
    
    rows = []
    weights = {}
    
    for name, method in schemes:
        try:
            # Get weights (type doesn't matter for weights, just use median)
            _, w = integrate(data, measure, method, type="median", agg=agg)
        except (KeyError, ValueError) as e:
            print(f"Method {method} could not be computed: {e}")
            continue
        
        # Compute both median and mean
        y_median = weighted_median(y, w)
        y_mean = weighted_mean(y, w)
        
        # Get estimate from max-weight universe
        j = int(np.argmax(w))
        y_maxw = float(y[j])
        
        row = {
            "Scheme": name,
            "median": y_median,
            "mean": y_mean,
            "gini_w": gini(w),
            "maxw_estimate": y_maxw,
        }
        
        if true_value is not None:
            row["err_median"] = abs(y_median - true_value)
            row["err_mean"] = abs(y_mean - true_value)
            row["err_maxw"] = abs(y_maxw - true_value)
        
        rows.append(row)
        weights[name] = w
    
    table = pd.DataFrame(rows)
    return table, weights

# Plotting function
def plot_methods(
    data: pd.DataFrame,
    measure: str,
    weights: Dict[str, np.ndarray],
    true_value: float | None = None,
    agg: str = "mean",
    xlim: tuple = None,
    figsize: tuple = (10, 4),
):

    
    data = data.copy()
    data.columns = data.columns.str.lower()
    measure = measure.lower()
    
    y = get_measure_values(data, measure, agg=agg)
    # Bandwidth adjustments per method
    bw_adjust = {"Uniform": 1, "BMA": 2, "MYH": 1, "MYHN": 1, "MLI": 1}
    colors = {"Uniform": "black", "BMA": "red", "MYH": "purple", "MYHN": "orange", "MLI": "green"}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Auto xlim if not provided
    if xlim is None:
        margin = (y.max() - y.min()) * 0.1
        xlim = (y.min() - margin, y.max() + margin)
    
    ax.set(xlabel=measure, ylabel="Density", xlim=xlim)
    
    if true_value is not None:
        ax.axvline(true_value, color="gray", linestyle="--", linewidth=1.5, label=f"True {measure}")
    
    for name, w in weights.items():
        sns.kdeplot(
            x=y, 
            weights=w, 
            label=name, 
            color=colors.get(name, "black"), 
            bw_adjust=bw_adjust.get(name, 1), 
            ax=ax
        )
    
    ax.legend()
    plt.tight_layout()
    
    return fig, ax