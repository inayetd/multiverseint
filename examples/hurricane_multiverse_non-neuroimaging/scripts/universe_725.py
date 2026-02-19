import comet
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load hurricane data
df = comet.utils.load_example("hurricane")

# Create derived predictors
df["post1979"] = (df["elapsed_years"] > 1979).astype(int)
df["zcat"] = (df["category"] - df["category"].mean()) / df["category"].std()
df["zpressure"] = -((df["pressure"] - df["pressure"].mean()) / df["pressure"].std())
df["zwind"] = (df["wind"] - df["wind"].mean()) / df["wind"].std()
df["z3"] = (df["zpressure"] + df["zcat"] + df["zwind"]) / 3.0

# Decisions 1 & 2: Exclude outliers
top_deaths = df["deaths"].nlargest(1).unique()
top_damage = df["damages"].nlargest(2).unique()
df = df[~df["deaths"].isin(top_deaths) & ~df["damages"].isin(top_damage)]

# Decision 2: Choose femininity rating
df["femininity"] = df["fem_likert"]

# Decision 3: Transform damage variable
if "ln" == "ln":
    df["damages"] = np.log(df["damages"])

# Decision 4 & 5: Effect type & account for year
formula = "deaths ~ " + "femininity * damages" + " + year:damages"

# Decision 6: Model type and fitting
if "log_linear" == "log_linear":
    df["deaths"] = np.log(df["deaths"] + 1)
    fit = smf.ols(formula, data=df).fit()
else:
    fit = smf.glm(formula, data=df, family=sm.families.NegativeBinomial(alpha=1)).fit()

# Results: Set femininity for male vs female and predict deaths at the sample means
if "fem_likert" == "fem_likert":
    fem_male = df["femininity"][df["fem_binary"] == 0].mean()
    fem_fem = df["femininity"][df["fem_binary"] == 1].mean()
else:
    fem_male = 0
    fem_fem = 1

base = df.mean(numeric_only=True).to_frame().T

base_male = base.copy()
base_male["femininity"] = fem_male

base_fem = base.copy()
base_fem["femininity"] = fem_fem

if "log_linear" == "log_linear":
    # Predict and back-transform (slightly simplified)
    pred_m = np.exp(fit.predict(base_male)) - 1
    pred_f = np.exp(fit.predict(base_fem)) - 1
else:
    pred_m = fit.predict(base_male)
    pred_f = fit.predict(base_fem)

# Get the additional deaths as multiverse outcome measure
extra_deaths = pred_f.values - pred_m.values
# The p-values mix interaction with main effect only so its a bit hacky, but good enough for illustration
p_val = fit.pvalues["femininity:damages"] if "femininity:damages" in fit.pvalues.index else fit.pvalues["femininity"]
bic = fit.bic

comet.utils.save_universe_results({"formula": formula, 
                                   "extra_deaths": extra_deaths, 
                                   "p_val": p_val,
                                   "bic": bic})