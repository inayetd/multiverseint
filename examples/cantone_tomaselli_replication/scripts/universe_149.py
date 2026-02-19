import comet
import pandas as pd
import statsmodels.formula.api as smf

data= pd.read_csv("/home/mibur/multiverse-analysis/micha_new/data/data.csv").iloc[:, 1:]

# Build composite predictor
data["pred"] = 0.0
if True:
    data["pred"] += data["BM1"]
if False:
    data["pred"] += data["BM2"]
if True:
    data["pred"] += data["BM3"]
if False:
    data["pred"] += data["BM4"]

# Build covariates
cov = ""
if False:
    cov += " + genotype"
if True:
    cov += " + pain"
if False:
    cov += " + fatigue"
if False:
    cov += " + age"

# Regression model
formula = "depression ~ pred" + cov
fit_res = smf.ols(formula=formula, data=data).fit()

beta_raw = float(fit_res.params["pred"])
sd_y = float(data["depression"].std())
sd_x = float(data["pred"].std())
beta_std = beta_raw * (sd_x / sd_y)

out = {
    "formula": formula,
    "beta": float(beta_std),
    "beta_raw": float(beta_raw),
    "p_value": float(fit_res.pvalues["pred"]),
    "bic": float(fit_res.bic),
}

comet.utils.save_universe_results(out)