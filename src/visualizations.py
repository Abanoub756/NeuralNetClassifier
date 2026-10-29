"""Reusable Plotly figures and diagram sources. All functions take real data/results as input."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "Balanced accuracy", "ROC-AUC"]
MODEL_COLORS = {"RBF": "#1f77b4", "PNN": "#ff7f0e"}


def target_distribution_fig(counts: Dict[int, int], class_names: Dict[int, str], title: str) -> go.Figure:
    df = pd.DataFrame({"Class": [class_names[k] for k in sorted(counts)], "Records": [counts[k] for k in sorted(counts)]})
    fig = px.bar(df, x="Class", y="Records", text="Records", title=title, color="Class")
    fig.update_layout(showlegend=False)
    return fig


def numeric_distribution_fig(df: pd.DataFrame, column: str, y: pd.Series, class_names: Dict[int, str]) -> go.Figure:
    data = pd.DataFrame({column: df[column], "Class": y.map(class_names)}).dropna()
    return px.histogram(data, x=column, color="Class", barmode="overlay", opacity=0.65, nbins=25,
                        title=f"Distribution of '{column}' by class")


def categorical_distribution_fig(df: pd.DataFrame, column: str, y: pd.Series, class_names: Dict[int, str]) -> go.Figure:
    data = pd.DataFrame({column: df[column].astype(str), "Class": y.map(class_names)})
    counts = data.groupby([column, "Class"]).size().reset_index(name="Records")
    return px.bar(counts, x=column, y="Records", color="Class", barmode="group",
                  title=f"'{column}' categories by class")


def metric_comparison_fig(table: pd.DataFrame, metrics: Sequence[str] = METRIC_COLUMNS, title: str = "") -> go.Figure:
    """Grouped bars: one group per metric, one bar per model. Undefined values are simply not drawn."""
    fig = go.Figure()
    for _, row in table.iterrows():
        vals = [None if pd.isna(row[m]) else float(row[m]) for m in metrics]
        fig.add_bar(name=str(row["Model"]), x=list(metrics), y=vals, marker_color=MODEL_COLORS.get(row["Model"]),
                    text=[("" if v is None else f"{v:.3f}") for v in vals], textposition="outside")
    fig.update_layout(barmode="group", title=title, yaxis=dict(range=[0, 1.1], title="Score"))
    return fig


def confusion_matrix_fig(cm: Sequence[Sequence[int]], class_names: Sequence[str], title: str) -> go.Figure:
    cm = np.asarray(cm)
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred: {c}" for c in class_names], y=[f"True: {c}" for c in class_names],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return fig


def roc_fig(curves: Dict[str, Optional[Dict[str, List[float]]]], aucs: Dict[str, Optional[float]], title: str) -> go.Figure:
    fig = go.Figure()
    for name, c in curves.items():
        if c is None:
            continue
        auc = aucs.get(name)
        label = f"{name} (AUC = {auc:.3f})" if auc is not None else name
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=label, line=dict(color=MODEL_COLORS.get(name)))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray"))
    fig.update_layout(title=title, xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def timing_fig(table: pd.DataFrame, column: str, title: str, y_title: str) -> go.Figure:
    fig = px.bar(table, x="Dataset", y=column, color="Model", barmode="group", title=title,
                 color_discrete_map=MODEL_COLORS, text_auto=".3g")
    fig.update_layout(yaxis_title=y_title)
    return fig


def score_bar_fig(scores: Dict[str, Sequence[float]], class_names: Sequence[str], title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    for name, vals in scores.items():
        fig.add_bar(name=name, x=list(class_names), y=[float(v) for v in vals], marker_color=MODEL_COLORS.get(name))
    fig.update_layout(barmode="group", title=title, yaxis_title=y_title)
    return fig


def rbf_architecture_dot() -> str:
    return """digraph G { rankdir=LR; node [shape=circle, fontsize=10];
    subgraph cluster_in {label="Input layer"; x1; x2; x3;}
    subgraph cluster_h {label="Hidden layer (Gaussian units, centres c_j)"; h1 [label="φ1"]; h2 [label="φ2"]; h3 [label="φ3"]; b [label="bias=1", shape=box];}
    subgraph cluster_o {label="Output layer (linear)"; o0 [label="score\\nclass 0"]; o1 [label="score\\nclass 1"];}
    x1->h1; x1->h2; x1->h3; x2->h1; x2->h2; x2->h3; x3->h1; x3->h2; x3->h3;
    h1->o0; h1->o1; h2->o0; h2->o1; h3->o0; h3->o1; b->o0; b->o1;
    o0->pred; o1->pred; pred [label="argmax", shape=box]; }"""


def pnn_architecture_dot() -> str:
    return """digraph G { rankdir=LR; node [shape=circle, fontsize=10];
    subgraph cluster_in {label="Input layer"; x1; x2;}
    subgraph cluster_p {label="Pattern layer (one Gaussian kernel per training sample)"; a1 [label="K(x,x₁)\\nclass 0"]; a2 [label="K(x,x₂)\\nclass 0"]; b1 [label="K(x,x₃)\\nclass 1"]; b2 [label="K(x,x₄)\\nclass 1"];}
    subgraph cluster_s {label="Summation layer (density per class)"; s0 [label="f₀(x)"]; s1 [label="f₁(x)"];}
    x1->a1; x1->a2; x1->b1; x1->b2; x2->a1; x2->a2; x2->b1; x2->b2;
    a1->s0; a2->s0; b1->s1; b2->s1;
    s0->out; s1->out; out [label="argmax\\nprior × density", shape=box]; }"""
