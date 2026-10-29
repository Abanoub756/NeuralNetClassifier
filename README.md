# NeuralNetClassifier - RBF vs PNN

A complete, modular Python project that compares two neural-network classifiers implemented **from scratch** -
a **Radial Basis Function network (RBF)** and a **Probabilistic Neural Network (PNN)** - on two real datasets, with a Streamlit app.

## Research question
Under identical splits, preprocessing and evaluation, how do an RBF network and a PNN compare in predictive quality
and computational cost on (A) Cleveland Heart Disease and (B) Student Performance (Portuguese)?
Conclusions apply only to these datasets and this setup.

## Datasets and targets
| Dataset | File format | Target |
|---|---|---|
| Cleveland Heart Disease (`processed.cleveland.data`, 303 rows) | headerless CSV, `?` = missing | original 0 → **0** (no disease); 1-4 → **1** (disease) |
| Student Performance, Portuguese (`student-por.csv`, 649 rows) | `;`-separated with header | `G3 < 10` → **0** (fail); `G3 ≥ 10` → **1** (pass) |

Paths are set in **`config.py`** (`HEART_DATA_PATH`, `STUDENT_DATA_PATH`). The files are only read, never modified, and are not in this ZIP.

## Preprocessing (fitted on training data only)
* **Heart** - continuous: `age, trestbps, chol, thalach, oldpeak` → median imputation + standardisation.
  Categorical: `sex, cp, fbs, restecg, exang, slope, ca, thal` → most-frequent imputation + one-hot. `?` becomes NaN *before* numeric conversion; no rows are dropped. The original 0-4 target is kept for exploration only, never as a feature.
* **Student** - text columns (school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic) → one-hot.
  Integer-coded scales (age, Medu, Fedu, traveltime, studytime, failures, famrel, freetime, goout, Dalc, Walc, health, absences) have a meaningful *order*, so they stay numeric (median imputation + standardisation).
* The same feature representation feeds both models within a dataset. Unseen categories at test time are ignored safely.

## Data-leakage prevention
* `G3` only builds the target; **`G1` and `G2` are also excluded** (30 input features remain). The loader raises an error if any grade column reaches the feature matrix, and tests verify it.
* Imputers, scalers and encoders are fitted on the training split only; inside cross-validation they are re-fitted per fold.
* Hyper-parameters are chosen by CV on the training split; the test set is used once for final evaluation.

## The models
**RBF** (`src/rbf.py`): K-Means centres → Gaussian units `φ_j(x)=exp(-‖x-c_j‖²/(2σ_j²))` → ridge least-squares output layer on one-hot targets `W=(ΦᵀΦ+λI')⁻¹ΦᵀT` (bias unpenalised). Prediction = argmax of output scores (raw regression scores, not probabilities). Widths: `global` σ=scale·d_max/√(2m) or `local` (mean distance to nearest centres).

**PNN** (`src/pnn.py`): pattern layer (one Gaussian kernel per training sample), summation layer (class-conditional Parzen densities, normalised by `1/n_k` and `(2πh²)^{d/2}`), output layer: posterior ∝ prior × density. Priors: empirical class frequencies by default (`uniform` available). Computed with log-sum-exp for numerical stability.

## Experimental methodology
1. Load/validate and build the binary target; report counts, class balance and missing values.
2. Stratified 80/20 split, `random_state=42`; **identical indices for RBF and PNN**.
3. Fit preprocessing on the training split.
4. Stratified 5-fold CV on the training split: RBF grid (centres × width rule × width scale × ridge = 90 configs) and PNN bandwidth grid (8 values). Selection metric: balanced accuracy.
5. Refit the winners on the full training split; evaluate **once** on the test split.
6. Results are saved to `results/<dataset>_results.json`.

## Metrics
Accuracy, precision, recall, F1 (positive class = 1), balanced accuracy, confusion matrix, ROC-AUC (positive-class score), training time (**seconds**) and inference time (**milliseconds per sample**, model only, median of repeats). Undefined metrics are reported as *undefined* (`None`) with a note - never invented.

## Project structure
```
NeuralNetClassifier/
├── NeuralNetClassifier.pyproj   Visual Studio 2022 project
├── app.py                       Streamlit application (7 pages)
├── run_app.py                   F5 launcher: runs `python -m streamlit run app.py` (installs nothing)
├── config.py                    paths, seed, split, grids, CV, output settings
├── requirements.txt  pytest.ini  conftest.py  .gitignore
├── src/  data_loader.py preprocessing.py rbf.py pnn.py mathutils.py evaluation.py experiments.py visualizations.py
├── tests/  test_data_loader.py test_preprocessing.py test_rbf.py test_pnn.py test_evaluation.py test_experiments.py (+ conftest.py)
├── data/README.md   results/README.md
```

## Dependencies
`numpy, pandas, scipy, scikit-learn (≥1.2), streamlit, plotly, pytest` - see `requirements.txt`. Use your **existing** Python environment; nothing in the project installs packages.

## Configure the dataset paths
Open `config.py` and edit `HEART_DATA_PATH` and `STUDENT_DATA_PATH` (raw strings `r"..."` handle spaces, `+` and back-slashes). Alternatively set the environment variables `NNC_HEART_PATH` / `NNC_STUDENT_PATH`. If a path is wrong the app shows exactly which setting to fix.

## Open in Visual Studio 2022
1. Extract the ZIP. 2. *File → Open → Project/Solution…* → select `NeuralNetClassifier.pyproj`.
3. In *Solution Explorer → Python Environments*, select the environment that already has your libraries.
4. Edit the paths in `config.py`. 5. Set `run_app.py` as startup file (already set) and press **Ctrl+F5**, or use a terminal (below).

## Run the app (on the computer that has the dataset files)
Open a terminal (Cmder / VS terminal) **in the project folder** with your Python environment active:
```bash
streamlit run app.py
```
The first visit runs the experiments (tuning + training; typically seconds to about a minute) and caches them. Use *Re-run experiments* in the sidebar to clear the cache; changed files/settings never reuse stale models.

## Run the tests
```bash
pytest
```
Tests use small synthetic fixtures (software tests only, never presented as results) and do not need the real datasets.

## Interpreting results
Compare balanced accuracy, F1 and ROC-AUC rather than accuracy alone; check the confusion matrices. The test sets are small (≈61 heart, ≈130 student rows), so differences of a few percent may be noise - the Findings page shows 95 % Wilson intervals. RBF scores are regression outputs; only PNN posteriors are probabilities. In the Playground, held-out samples show correct/incorrect; custom edited samples never do (no ground truth).

## Limitations
Single split and seed; two datasets; restricted grids; one-hot encoding raises dimensionality for distance-based methods; student models exclude G1/G2 by design; timings are machine dependent; no formal feature-importance analysis is included.
