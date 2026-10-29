"""Streamlit application: RBF vs PNN on Cleveland Heart Disease and Student Performance (Portuguese).

Run from the project folder with:   streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import config
from src import visualizations as viz
from src.data_loader import DatasetError, load_dataset
from src.evaluation import wilson_interval
from src.experiments import MODEL_NAMES, ExperimentResult, best_model, predict_samples, run_experiment

st.set_page_config(page_title="RBF vs PNN Classifier", page_icon="🧠", layout="wide")
LABELS = {k: v["label"] for k, v in config.DATASET_INFO.items()}
METRIC_KEYS = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1",
               "Balanced accuracy": "balanced_accuracy", "ROC-AUC": "roc_auc"}


# ------------------------------------------------------------------ cached loaders
def _file_signature(path: Path) -> Tuple[int, int]:
    stt = path.stat()
    return stt.st_mtime_ns, stt.st_size


@st.cache_resource(show_spinner="Running experiments (tuning with cross-validation, training, evaluating)...")
def _cached_experiment(key: str, path_str: str, file_sig: Tuple[int, int], settings_json: str) -> ExperimentResult:
    # file_sig + settings_json are part of the cache key -> changed data/settings never reuse stale models.
    return run_experiment(key, path=Path(path_str), settings=json.loads(settings_json))


@st.cache_data(show_spinner="Loading dataset...")
def _cached_dataset(key: str, path_str: str, file_sig: Tuple[int, int]):
    return load_dataset(key, Path(path_str))


def get_experiment(key: str) -> Tuple[Optional[ExperimentResult], Optional[str]]:
    path = config.get_dataset_path(key)
    if not path.is_file():
        return None, _missing_msg(key, path)
    try:
        return _cached_experiment(key, str(path), _file_signature(path), json.dumps(config.default_settings(), sort_keys=True)), None
    except DatasetError as exc:
        return None, str(exc)


def get_dataset(key: str):
    path = config.get_dataset_path(key)
    if not path.is_file():
        return None, _missing_msg(key, path)
    try:
        return _cached_dataset(key, str(path), _file_signature(path)), None
    except DatasetError as exc:
        return None, str(exc)


def _missing_msg(key: str, path: Path) -> str:
    return (f"{LABELS[key]} file not found at:\n\n`{path}`\n\nUpdate `{config.CONFIG_PATH_NAMES[key]}` in "
            f"`config.py` (in `{config.PROJECT_ROOT}`) and reload. No results are shown until the real data has been loaded "
            "and the models have been trained.")


def all_experiments() -> Tuple[Dict[str, ExperimentResult], Dict[str, str]]:
    ok, errs = {}, {}
    for key in config.DATASET_KEYS:
        res, err = get_experiment(key)
        if res is not None:
            ok[key] = res
        else:
            errs[key] = err
    return ok, errs


def metrics_dataframe(exps: Dict[str, ExperimentResult]) -> pd.DataFrame:
    rows = []
    for exp in exps.values():
        for name in MODEL_NAMES:
            r = exp.results[name]
            row = {"Dataset": exp.dataset.label, "Model": name}
            for col, k in METRIC_KEYS.items():
                row[col] = np.nan if r.metrics[k] is None else r.metrics[k]
            row["Train time (s)"] = r.train_time_s
            row["Inference (ms/sample)"] = r.inference_ms_per_sample
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in list(METRIC_KEYS) + ["Train time (s)", "Inference (ms/sample)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def show_table(df: pd.DataFrame) -> None:
    st.dataframe(df.style.format(precision=4, na_rep="undefined", subset=[c for c in df.columns if c not in ("Dataset", "Model")]),
                 hide_index=True)


def show_errors(errs: Dict[str, str]) -> None:
    for key, msg in errs.items():
        st.error(msg)
    if errs:
        st.info("The experiments have not run yet because at least one real dataset could not be loaded. "
                "The explanatory pages (RBF Notes, PNN Notes) still work.")


# ------------------------------------------------------------------------- pages
def page_overview() -> None:
    st.title("🧠 RBF vs PNN: neural-network classification")
    st.markdown(
        "**Purpose.** Compare two neural-network classifiers implemented from scratch - a **Radial Basis Function "
        "network (RBF)** and a **Probabilistic Neural Network (PNN)** - on two real datasets, using identical splits, "
        "identical preprocessing and the same evaluation protocol.")
    c1, c2 = st.columns(2)
    c1.markdown("##### Cleveland Heart Disease\nPredict whether heart disease is present (original target 1-4) or absent (0). "
                "5 continuous + 8 categorical features.")
    c2.markdown("##### Student Performance (Portuguese)\nPredict pass (G3 ≥ 10) vs fail (G3 < 10) from 30 background/behaviour "
                "features. **G1, G2 and G3 are never used as inputs** (leakage).")
    c1, c2 = st.columns(2)
    c1.markdown("**RBF**: K-Means centres → Gaussian hidden units → ridge least-squares linear output layer.")
    c2.markdown("**PNN**: one Gaussian kernel per training sample → class-wise density estimates → Bayes-style decision.")

    exps, errs = all_experiments()
    show_errors(errs)
    if not exps:
        return
    st.subheader("Datasets (loaded from your files)")
    cols = st.columns(len(exps))
    for col, exp in zip(cols, exps.values()):
        rep = exp.dataset.report
        col.markdown(f"**{exp.dataset.label}**")
        col.metric("Records", rep["n_records"])
        col.metric("Raw input features", rep["n_features"])
        col.metric("Encoded features (after preprocessing)", len(exp.feature_names))
        col.caption(f"Train {len(exp.train_idx)} / Test {len(exp.test_idx)}")
        for w in exp.warnings:
            col.warning(w)
    st.subheader("Selected model settings (cross-validation on training split)")
    for exp in exps.values():
        with st.expander(exp.dataset.label, expanded=True):
            for name in MODEL_NAMES:
                r = exp.results[name]
                st.write(f"**{name}** - CV {config.CV_SCORING.replace('_', ' ')}: {r.cv_mean:.3f} ± {r.cv_std:.3f}")
                st.json(r.best_params, expanded=False)
    st.subheader("Held-out test results")
    df = metrics_dataframe(exps)
    show_table(df)
    st.plotly_chart(viz.metric_comparison_fig(df.assign(Model=df["Dataset"] + " - " + df["Model"]), title="All metrics"))
    for exp in exps.values():
        name, why = best_model(exp)
        st.success(f"{exp.dataset.label}: " + (f"best observed model = **{name}**. {why}" if name else why))
    st.subheader("Experimental procedure")
    st.markdown(
        "1. Load + validate file, build binary target.  2. Stratified 80/20 split (seed 42) shared by both models.  "
        "3. Fit imputers/scalers/encoders on the training split only.  4. Tune RBF and PNN with stratified "
        f"{config.CV_FOLDS}-fold CV on the training split (preprocessing re-fitted inside each fold).  "
        "5. Refit the chosen settings on the full training split.  6. Evaluate each model once on the held-out test set.")


def page_explorer() -> None:
    st.title("🔎 Dataset Explorer")
    key = st.selectbox("Dataset", config.DATASET_KEYS, format_func=LABELS.get)
    ds, err = get_dataset(key)
    if err:
        st.error(err)
        return
    for w in ds.warnings:
        st.warning(w)
    rep = ds.report
    c = st.columns(4)
    c[0].metric("Records", rep["n_records"]); c[1].metric("Input features", rep["n_features"])
    c[2].metric("Columns in file", rep["n_raw_columns"]); c[3].metric("Missing feature values", rep["total_missing_values"])
    if key == "student":
        st.info("The file's G1, G2 and G3 columns are shown here for exploration only; they are excluded from model inputs.")
    st.write("**Column names:**", ", ".join(map(str, ds.raw.columns)))
    st.write("**Preview**")
    st.dataframe(ds.raw.head(25))
    left, right = st.columns(2)
    with left:
        st.write("**Missing values per column**")
        miss = ds.raw.isna().sum().rename("missing").to_frame()
        st.dataframe(miss[miss["missing"] > 0] if (miss["missing"] > 0).any() else miss.head(0))
        if not (miss["missing"] > 0).any():
            st.caption("No missing values.")
    with right:
        st.plotly_chart(viz.target_distribution_fig(rep["target_counts"], ds.class_names, "Target distribution"))
    st.write("**Numerical feature summary**")
    st.dataframe(ds.X[ds.numeric_features].describe().T)
    st.write("**Categorical feature summary**")
    st.dataframe(pd.DataFrame({"unique values": ds.X[ds.categorical_features].nunique(),
                               "most frequent": ds.X[ds.categorical_features].mode().iloc[0],
                               "missing": ds.X[ds.categorical_features].isna().sum()}))
    t1, t2 = st.tabs(["Numeric feature", "Categorical feature"])
    with t1:
        col = st.selectbox("Numeric feature", ds.numeric_features)
        st.plotly_chart(viz.numeric_distribution_fig(ds.X, col, ds.y, ds.class_names))
    with t2:
        col = st.selectbox("Categorical feature", ds.categorical_features)
        st.plotly_chart(viz.categorical_distribution_fig(ds.X, col, ds.y, ds.class_names))


def page_comparison() -> None:
    st.title("📊 Model Comparison")
    exps, errs = all_experiments()
    show_errors(errs)
    if not exps:
        return
    df = metrics_dataframe(exps)
    st.subheader("Metrics table (held-out test set)")
    show_table(df)
    st.caption("Precision/recall/F1 use the positive class (1 = heart disease / pass). 'undefined' means the metric cannot be computed "
               "(e.g. no positive predictions) - no value is invented. Times: training in seconds, inference in milliseconds per sample "
               "(model only, preprocessing excluded).")
    st.subheader("Metric charts")
    for exp in exps.values():
        sub = df[df["Dataset"] == exp.dataset.label]
        st.plotly_chart(viz.metric_comparison_fig(sub, title=exp.dataset.label))
    st.subheader("Training and inference time")
    a, b = st.columns(2)
    a.plotly_chart(viz.timing_fig(df, "Train time (s)", "Training time", "seconds"))
    b.plotly_chart(viz.timing_fig(df, "Inference (ms/sample)", "Inference time per sample", "milliseconds / sample"))
    for exp in exps.values():
        st.markdown(f"### {exp.dataset.label}")
        names = [exp.dataset.class_names[0], exp.dataset.class_names[1]]
        cols = st.columns(len(MODEL_NAMES) + 1)
        for col, name in zip(cols, MODEL_NAMES):
            r = exp.results[name]
            col.plotly_chart(viz.confusion_matrix_fig(r.metrics["confusion_matrix"], names, f"{name} confusion matrix"))
        curves = {n: exp.results[n].roc for n in MODEL_NAMES}
        if any(v is not None for v in curves.values()):
            cols[-1].plotly_chart(viz.roc_fig(curves, {n: exp.results[n].metrics["roc_auc"] for n in MODEL_NAMES}, "ROC curves"))
        else:
            cols[-1].info("ROC curve unavailable for this test split.")
        h1, h2 = st.columns(2)
        for col, name in zip((h1, h2), MODEL_NAMES):
            col.write(f"**{name} selected hyper-parameters**")
            col.json(exp.results[name].best_params, expanded=True)
            with col.expander(f"{name}: top cross-validation configurations"):
                st.dataframe(pd.DataFrame(exp.results[name].cv_table))
        winner, why = best_model(exp)
        st.info((f"On this test split, **{winner}** performed better. " if winner else "On this test split the models tied. ") + why +
                " This conclusion applies only to this dataset, this split and this experimental setup - it is not a claim that either model is universally better.")


def _playground_inputs(exp: ExperimentResult, key: str, pos: int) -> pd.DataFrame:
    """Editable widgets for a custom sample, initialised from the selected test row."""
    base = exp.X_test.iloc[[pos]].copy()
    cols = st.columns(4)
    values = {}
    for i, col in enumerate(exp.dataset.X.columns):
        wkey, cur, holder = f"pg_edit_{key}_{pos}_{col}", base.iloc[0][col], cols[i % 4]
        if col in exp.dataset.numeric_features:
            default = float(cur) if pd.notna(cur) else float(exp.X_train[col].median())
            values[col] = holder.number_input(col, value=default, key=wkey)
        else:
            opts = sorted(exp.X_train[col].dropna().unique().tolist())
            labels = opts + ["(missing)"]
            idx = opts.index(cur) if pd.notna(cur) and cur in opts else len(opts)
            choice = holder.selectbox(col, labels, index=idx, key=wkey)
            values[col] = np.nan if choice == "(missing)" else choice
    return pd.DataFrame([values], columns=exp.dataset.X.columns)


def page_playground() -> None:
    st.title("🎮 Interactive Prediction Playground")
    key = st.selectbox("Dataset", config.DATASET_KEYS, format_func=LABELS.get)
    exp, err = get_experiment(key)
    if err:
        st.error(err)
        return
    ss, n_test = st.session_state, len(exp.test_idx)
    pos_key, shown_key, mode_key = f"pg_pos_{key}", f"pg_shown_{key}", f"pg_mode_{key}"
    ss.setdefault(pos_key, 0)
    ss.setdefault(shown_key, False)
    disagree = [i for i in range(n_test) if exp.results["RBF"].y_pred[i] != exp.results["PNN"].y_pred[i]]

    def _random():
        ss[pos_key] = int(np.random.default_rng().integers(n_test)); ss[shown_key] = False

    def _disagree():
        ss[pos_key] = int(np.random.default_rng().choice(disagree)); ss[shown_key] = False

    def _reset():
        ss[shown_key] = False

    b1, b2, b3 = st.columns([1, 1, 2])
    b1.button("🎲 Pick Random Test Sample", on_click=_random)
    b2.button("⚡ Pick Disagreement Sample", on_click=_disagree, disabled=not disagree,
              help=None if disagree else "RBF and PNN agree on every held-out test sample.")
    b3.number_input(f"Held-out test sample number (0 to {n_test - 1})", 0, n_test - 1, key=pos_key, on_change=_reset)
    pos = int(ss[pos_key])
    mode = st.radio("Input mode", ["Held-out test sample (known true label)", "Edit features / custom sample (no ground truth)"],
                    key=mode_key, horizontal=True, on_change=_reset)
    custom = mode.startswith("Edit")
    names = exp.dataset.class_names
    st.caption(f"Original dataset row index: {int(exp.test_idx[pos])}")
    if custom:
        st.warning("Custom sample: predictions will be shown, but they **cannot be judged correct or incorrect** - there is no independent ground truth.")
        row = _playground_inputs(exp, key, pos)
    else:
        row = exp.X_test.iloc[[pos]]
        st.write("**Selected sample features**")
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}).astype(str))
        true = int(exp.y_test[pos])
        st.write(f"**True label:** {names[true]} ({true})")
    if st.button("▶ Test Prediction", type="primary"):
        ss[shown_key] = True
    if ss[shown_key]:
        out = predict_samples(exp, row)
        cols = st.columns(2)
        for col, name in zip(cols, MODEL_NAMES):
            pred = int(out[name]["pred"][0])
            col.markdown(f"#### {name}")
            col.write(f"Predicted: **{names[pred]}** ({pred})")
            if not custom:
                (col.success if pred == true else col.error)("✅ Correct" if pred == true else "❌ Incorrect")
            else:
                col.info("No ground truth for a custom sample.")
            score_label = "Posterior probability" if name == "PNN" else "Raw output-layer score (not a probability)"
            col.plotly_chart(viz.score_bar_fig({name: out[name]["scores"][0]}, [names[0], names[1]], f"{name}: class scores", score_label))
        st.plotly_chart(viz.score_bar_fig({n: out[n]["proba"][0] for n in MODEL_NAMES}, [names[0], names[1]],
                                          "Side by side (RBF values are clipped/normalised pseudo-probabilities, for display only)", "Score"))
        if not custom and out["RBF"]["pred"][0] != out["PNN"]["pred"][0]:
            st.info("The two models disagree on this sample.")
    st.caption(f"{len(disagree)} of {n_test} test samples are ones where RBF and PNN disagree.")


def page_rbf() -> None:
    st.title("📘 RBF Notes")
    st.markdown("""
**What is an RBF network?** A small neural network with one hidden layer whose units respond to *how close* an input is to a
stored point (a **centre**). The output layer is a simple linear model on those responses.

**Architecture:** input → hidden layer of Gaussian units → linear output layer (one output per class) → pick the largest.""")
    st.graphviz_chart(viz.rbf_architecture_dot())
    st.markdown("**Gaussian basis function** (centre $c_j$, width $\\sigma_j$):")
    st.latex(r"\phi_j(x)=\exp\!\left(-\frac{\|x-c_j\|^2}{2\sigma_j^2}\right)")
    st.markdown("""
* **Centres** are found with **K-Means** on the training data - they summarise where the data lives.
* **Width σ** controls how far a unit's influence reaches: small σ → very local, wiggly; large σ → smooth, overlapping.
  Here: `global` rule σ = scale·d_max/√(2m), or `local` rule (distance to nearest centres).
* **Output layer** - one-hot targets $T$, design matrix $\\Phi$ (with a bias column), ridge least squares:""")
    st.latex(r"W=(\Phi^{\top}\Phi+\lambda I')^{-1}\Phi^{\top}T")
    st.markdown("""The class with the largest output score is predicted. Scores are regression outputs, **not probabilities**.

**Hyper-parameters:** number of centres, width rule and scale, ridge λ (all tuned with cross-validation on training data).

**Advantages:** compact model after training, fast prediction, smooth decision boundaries, closed-form output training.
**Disadvantages:** sensitive to centre count and width, K-Means depends on initialisation, struggles when many features are irrelevant.

**In this project:** `src/rbf.py` - `select_centers` (K-Means), `RBFNetwork.fit` (widths + ridge solve), `class_scores`, `predict`.""")


def page_pnn() -> None:
    st.title("📗 PNN Notes")
    st.markdown("""
**What is a PNN?** A neural-network form of a *Parzen-window / kernel density classifier*. It stores **every training sample**
and classifies a new point by asking which class's training samples are "densest" around it.""")
    st.graphviz_chart(viz.pnn_architecture_dot())
    st.markdown("""
* **Pattern layer:** one Gaussian kernel per training sample, $K(x,x_i)=\\exp(-\\|x-x_i\\|^2/2h^2)$.
* **Summation layer:** averages the kernels within each class → class-conditional density estimate:""")
    st.latex(r"f_k(x)=\frac{1}{n_k(2\pi h^2)^{d/2}}\sum_{i\in k}\exp\!\left(-\frac{\|x-x_i\|^2}{2h^2}\right)")
    st.markdown("""
* **Output layer:** posterior $P(k\\mid x)\\propto \\pi_k f_k(x)$ with priors $\\pi_k$ (empirical class frequencies by default); predict the largest.
* **Bandwidth h:** small h → memorises (overfits); large h → over-smooths. It is the key hyper-parameter and is tuned by cross-validation.

**Advantages:** no iterative training, few hyper-parameters, outputs real probabilities. **Disadvantages:** stores all training data, prediction cost grows
with training-set size, sensitive to the single bandwidth and to irrelevant features.

**Versus RBF:** a PNN has one unit per training sample and no learned output weights; an RBF has a few centres and *learns* a linear output layer.

**In this project:** `src/pnn.py` - `PNN.fit` (pattern layer), `log_class_densities` (log-sum-exp for stability), `predict_proba`, `predict`.""")


def page_findings() -> None:
    st.title("📝 Final Findings")
    exps, errs = all_experiments()
    if errs:
        show_errors(errs)
        st.warning("Findings are generated from real experiment results and will be available once the datasets are loaded.")
    if not exps:
        return
    for exp in exps.values():
        rep, (r, p) = exp.dataset.report, (exp.results["RBF"], exp.results["PNN"])
        n_test = len(exp.test_idx)
        st.header(exp.dataset.label)
        tc = rep["target_counts"]
        st.write(f"{rep['n_records']} records, {rep['n_features']} input features ({len(exp.feature_names)} after encoding). "
                 f"Class counts: {exp.dataset.class_names[0]} = {tc[0]}, {exp.dataset.class_names[1]} = {tc[1]}. "
                 f"Train/test = {len(exp.train_idx)}/{n_test}.")
        winner, why = best_model(exp)
        f = lambda v: "undefined" if v is None else f"{v:.3f}"
        st.write(f"**RBF** - accuracy {f(r.metrics['accuracy'])}, balanced accuracy {f(r.metrics['balanced_accuracy'])}, F1 {f(r.metrics['f1'])}, ROC-AUC {f(r.metrics['roc_auc'])}.  \n"
                 f"**PNN** - accuracy {f(p.metrics['accuracy'])}, balanced accuracy {f(p.metrics['balanced_accuracy'])}, F1 {f(p.metrics['f1'])}, ROC-AUC {f(p.metrics['roc_auc'])}.")
        st.write((f"Best observed model: **{winner}**. " if winner else "No clear winner. ") + why)
        for m in (r, p):
            lo, hi = wilson_interval(round(m.metrics["accuracy"] * n_test), n_test)
            st.caption(f"{m.name} test accuracy 95% Wilson interval: {lo:.3f} - {hi:.3f} (n = {n_test}).")
        st.write(f"**Efficiency:** training RBF {r.train_time_s:.4f} s vs PNN {p.train_time_s:.4f} s; inference RBF {r.inference_ms_per_sample:.4f} ms/sample vs "
                 f"PNN {p.inference_ms_per_sample:.4f} ms/sample (timings depend on this computer).")
        if exp.results_path:
            st.caption(f"Structured results saved to: {exp.results_path}")
    st.header("Limitations")
    st.markdown("""
* A single stratified 80/20 split: small test sets give noisy estimates, and differences between models may be within chance (see intervals).
* Only two datasets, one random seed, and the hyper-parameter grids in `config.py`; results may change with other settings.
* Student-performance models use no prior grades (G1/G2), so accuracy is lower than in studies that include them - by design, to avoid leakage.
* One-hot encoding increases dimensionality, which affects distance-based methods such as RBF and PNN.
* Timings are wall-clock measurements on one machine.""")
    st.header("Conclusion")
    st.write("Within this experimental setup, the table above reports which model scored better on each held-out test set. "
             "These findings should be discussed as observations about these datasets and this protocol, not as a general ranking of RBF versus PNN.")


PAGES = {"Overview Dashboard": page_overview, "Dataset Explorer": page_explorer, "Model Comparison": page_comparison,
         "Prediction Playground": page_playground, "RBF Notes": page_rbf, "PNN Notes": page_pnn, "Final Findings": page_findings}


def main() -> None:
    st.sidebar.title("NeuralNetClassifier")
    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.markdown("**Dataset paths** (edit in `config.py`)")
    for k in config.DATASET_KEYS:
        st.sidebar.caption(f"{LABELS[k]}:\n`{config.get_dataset_path(k)}`")
    if st.sidebar.button("Re-run experiments (clear cache)"):
        st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    PAGES[page]()


main()
