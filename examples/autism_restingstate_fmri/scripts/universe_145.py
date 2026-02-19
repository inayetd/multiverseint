import comet
import numpy as np
from nilearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_1samp

# Get data (if available, it will be loaded from disk)
data = datasets.fetch_abide_pcp(data_dir="/home/mibur/multiverse-analysis/micha_new/data/abide", verbose=0, 
                                pipeline="niak",
                                derivatives="rois_aal",
                                band_pass_filtering=True,
                                global_signal_regression=True)

time_series = data["rois_aal"]
diagnosis = data["phenotypic"]["DX_GROUP"]

# Calculate FC
tri_ix = None
features = []

for ts in time_series:
    FC = comet.connectivity.Static_Pearson(ts).estimate()

    if tri_ix == None:
        tri_ix = np.triu_indices_from(FC, k=1)
    
    feat_vec = FC[tri_ix]
    features.append(feat_vec)

# Prepare features (FC estimates) and target (autism/control)
X = np.vstack(features)
X[np.isnan(X)] = 0.0
y = np.array(diagnosis)

# Classification model
model = Pipeline([('scaler', StandardScaler()), ('reg', LogisticRegression(penalty='l2', C=0.25, tol=1e-3))])
cv = StratifiedKFold(n_splits=5)
accuracies = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
t_stat, p_value = ttest_1samp(accuracies, 0.5)

# Save the results
comet.utils.save_universe_results({"accuracy": accuracies, "p_value": round(p_value, 6)})