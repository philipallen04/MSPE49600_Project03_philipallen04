# MSPE 49600 - Project 3: Radial Basis Function and Neural Network Modeling

**Student:** Philip Allen  
**Purdue username:** `philipallen04`  
**Course:** MSPE 49600 - Data Analytics for Motorsports  
**Semester:** Fall 2026

## Project overview

This repository contains the complete Project 3 analysis for MSPE 49600. It uses two separate course datasets and keeps their validation results separate.

1. `ims_setup_project02.csv` - repeated IMS setup runs from Project 2. Rear-wing angle and rear ride height are used to predict mean Sector 3 time with Gaussian radial basis function (RBF) and quadratic surrogate models.
2. `simulated_laptime_data.csv` - 100 simulated lap-time observations. Fuel load and tire age are used to compare fixed ReLU and tanh multilayer perceptrons (MLPs) with a quadratic reference model.

The notebook follows the Project 3 requirements for setup-level splitting, fixed normalization bounds, custom RBF functions, LOOCV hyperparameter selection, held-out test evaluation, response-surface analysis, and fixed MLP settings.

## RBF results

The IMS data produce 27 independent setup configurations. The required 75/25 split gives 20 training setups and 7 held-out test setups.

- Fixed normalization domain: 5-13 deg rear-wing angle and 35-45 mm rear ride height
- Default RBF: sigma = 0.20, lambda = 0.01
- Selected RBF width: **sigma = 0.652057229343**
- Selected RBF regularization: **lambda = 1e-8**
- Default RBF LOOCV RMSE: **5.505476 s**
- Tuned RBF LOOCV RMSE: **0.079407 s**
- Quadratic LOOCV RMSE: **0.026529 s**
- Tuned RBF held-out test RMSE: **0.121868 s**
- Quadratic held-out test RMSE: **0.025850 s**

The quadratic model is recommended for Sector 3 prediction because it has both the lowest cross-validation error and the lowest held-out test error. The selected RBF lambda is on the lower boundary of the required search grid, so the reported pair is the best tested combination rather than proof of an interior continuous optimum.

## Lap-time neural-network results

The lap-time CSV is split with `test_size=0.25` and `random_state=42`, giving 75 training and 25 test observations. Input normalization uses only training-set minima and maxima. Both MLPs use the required `(20, 20)` architecture, `solver="lbfgs"`, `alpha=0.1`, `max_iter=5000`, and `random_state=1`.

| Model | Test RMSE (s) | Test MAE (s) | Test R2 |
|---|---:|---:|---:|
| ReLU MLP | 0.678809 | 0.485218 | 0.533951 |
| tanh MLP | **0.410702** | **0.300425** | **0.829396** |
| Quadratic | 0.590557 | 0.431262 | 0.647255 |

For the specified split and fixed settings, the tanh MLP has the lowest held-out test RMSE. This is a result for this dataset and split, not a universal ranking of activation functions or model families.

## Required files

- `philipallen04_pr03_notebook.ipynb` - executed notebook with all required outputs and all six figures
- `philipallen04_pr03_notebook.pdf` - PDF export of the executed notebook
- `philipallen04_pr03_notebook.py` - matching Python export
- `philipallen04_pr03_grouped.csv` - grouped IMS setup data
- `philipallen04_pr03_laptime_metrics.csv` - lap-time model metrics
- `philipallen04_pr03_laptime_predictions.csv` - train/test labels and predictions for all 100 lap-time rows
- `philipallen04_pr03_laptime_validation.png` - held-out lap-time actual-vs-predicted panels
- `philipallen04_pr03_laptime_surfaces.png` - ReLU, tanh, and quadratic lap-time surfaces
- `ims_setup_project02.csv` - unchanged source data
- `simulated_laptime_data.csv` - unchanged source data
- `README.md`
- `requirements.txt`
- `.gitignore`

## Environment used for the executed notebook

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8
- scikit-learn 1.8.0

## How to run

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Keep both source CSV files in the same folder as the notebook and Python export. Then either open and run `philipallen04_pr03_notebook.ipynb` from top to bottom, or run:

```bash
python philipallen04_pr03_notebook.py
```

The code uses relative paths and regenerates the three required result CSV files and two required neural-network PNG files.

## Reuse, references, and assistance

The setup cleaning/grouping pattern is reused from Philip Allen's Project 2 course work. The Task VI neural-network workflow is adapted from Andres Tovar's MSPE 49600 Lecture 16 CSV MLP example, modified to use the Project 3-required split, training-only normalization, architecture, solver, regularization, and seed.

OpenAI/ChatGPT was used for code review, debugging, formatting, and verification. The submitted calculations are executed from the supplied course datasets, and the student is responsible for understanding and explaining the analysis.