## Weighted Integration in Multiverse Analyses

Analysis code for integrating multiverse results using several aggregation/calibration schemes:
**Uniform**, **BMA**, **MYH**, **MYHN**, and **MLI**.

> This repository was developed as part of the Practical Project course of the  
> **Neurocognitive Psychology Master's programme** at the University of Oldenburg.  
> **Author:** Inayet Dincer | **Supervisor:** Micha Burkhardt

*Project repository:* `multiverseint`  
*Last updated:* 2026-02-19

## Abstract
...

## How to Use

### Installation

Install the required package:
```bash
pip install git+https://github.com/mibur1/comet.git
```

### Requirements
- Python 3.10+
- comet
- pandas
- numpy
- statsmodels
- matplotlib
- seaborn

### Running the Analyses

Each example follows the same structure:
```
example_name/
├── create_mv.ipynb   # Creates and runs the multiverse
├── integrate.ipynb   # Integrates and visualizes results
└── example_mv/       # Multiverse folder (created by comet)
```

**Step 1:** Open `create_mv.ipynb` and run all cells.  
This defines the forking paths and runs all universe combinations.

**Step 2:** Open `integrate.ipynb` and run all cells.  
This loads the results and applies integration methods (Uniform, BMA, MYH, MYHN, MLI).

### Examples

| Example | Description |
|---------|-------------|
| `hurricane_multiverse_non_neuroimaging` | Replication of Jung et al. (2014) hurricane fatalities study |
| `cantone_tomaselli_replication` | Replication of Cantone & Tomaselli (2024) |
| `autism_restingstate_fmri` | Autism resting-state fMRI multiverse analysis |

## Methods Implemented

**Uniform:** equal weights across all universes.  
**BMA:** Bayesian Model Averaging weights computed from BIC (requires a bic column).  
**MYH / MYHN:** regression-based decision sensitivity weighting (positive vs inverse weighting).  
**MLI:** local-instability weighting using Gower distance over decision encodings.  

## Poster
...

## References

Burkhardt, M., & Gießing, C. (2026). The Comet Toolbox: Improving robustness in network 
neuroscience through multiverse analysis. *Imaging Neuroscience*. 
https://doi.org/10.1162/IMAG.a.1122

Cantone, G. G., & Tomaselli, V. (2024). Characterisation and calibration of multiversal methods. 
*Advances in Data Analysis and Classification*. 
https://doi.org/10.1007/s11634-024-00610-9
