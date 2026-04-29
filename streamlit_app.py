"""
Streamlit web app for single-file and batch EIS fitting.

Place this file in the same repo folder as eis_fit.py, then run:
    streamlit run streamlit_app.py

Expected repo structure:
    battery-data-analysis/
    ├── streamlit_app.py
    ├── eis_fit.py
    └── requirements.txt
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from eis_fit import (
    PARAM_ORDER,
    arc_metrics,
    circuit_z,
    fit_eis,
    pack_params,
    read_biologic_mpr_eis,
    read_csv_eis,
    read_zfit_xml,
    unpack_params,
)


st.set_page_config(page_title="EIS Data Fitting", layout="wide")

st.title("EIS Data Fitting")
st.caption("Single-file and batch fitting with R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4).")


@st.cache_data(show_spinner=False)
def load_eis_from_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        if suffix == ".mpr":
            return read_biologic_mpr_eis(tmp_path)
        return read_csv_eis(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def load_xml_params_from_bytes(file_name: str, file_bytes: bytes) -> dict[str, float]:
    suffix = Path(file_name).suffix.lower() or ".xml"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return read_zfit_xml(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def safe_stem(file_name: str) -> str:
    return Path(file_name).stem.replace(" ", "_").replace("/", "_")


def make_nyquist_figure(df: pd.DataFrame, p_fit: np.ndarray, title: str | None = None):
    freq = df["freq_hz"].to_numpy(float)
    z_fit = circuit_z(p_fit, freq)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(df["z_real_ohm"], df["minus_z_imag_ohm"], label="data")
    ax.plot(np.real(z_fit), -np.imag(z_fit), label="fit")
    ax.set_xlabel("Z' / Ω")
    ax.set_ylabel("-Z'' / Ω")
    if title:
        ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    return fig


def make_residual_figure(df: pd.DataFrame, p_fit: np.ndarray, title: str | None = None):
    freq = df["freq_hz"].to_numpy(float)
    z_fit = circuit_z(p_fit, freq)
    residual_y = -np.imag(z_fit) - df["minus_z_imag_ohm"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.axhline(0, linewidth=1)
    ax.scatter(freq, residual_y)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Fit - data in -Z'' / Ω")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fit_one_file(file_name: str, file_bytes: bytes, p0: np.ndarray, weight: str) -> dict:
    df = load_eis_from_bytes(file_name, file_bytes)
    p_fit, result = fit_eis(df, p0, weight=weight)

    params_df = pd.DataFrame({
        "file": file_name,
        "parameter": PARAM_ORDER,
        "initial": p0,
        "fit": p_fit,
    })
    params_df.loc[len(params_df)] = [file_name, "cost", np.nan, result.cost]
    params_df.loc[len(params_df)] = [file_name, "nfev", np.nan, result.nfev]

    fmin, fmax = float(df["freq_hz"].min()), float(df["freq_hz"].max())
    metrics_df, fusion_df = arc_metrics(p_fit, fmin, fmax)
    metrics_df.insert(0, "file", file_name)
    fusion_df.insert(0, "file", file_name)

    z_fit = circuit_z(p_fit, df["freq_hz"].to_numpy(float))
    curve_df = df.copy()
    curve_df.insert(0, "file", file_name)
    curve_df["fit_z_real_ohm"] = np.real(z_fit)
    curve_df["fit_z_imag_ohm"] = np.imag(z_fit)
    curve_df["fit_minus_z_imag_ohm"] = -np.imag(z_fit)
    curve_df["residual_real_ohm"] = curve_df["fit_z_real_ohm"] - curve_df["z_real_ohm"]
    curve_df["residual_minus_imag_ohm"] = curve_df["fit_minus_z_imag_ohm"] - curve_df["minus_z_imag_ohm"]

    low_freq_mask = df["freq_hz"].to_numpy(float) <= 0.1
    residual_low = curve_df.loc[low_freq_mask, "residual_minus_imag_ohm"]
    mean_low_residual = float(residual_low.mean()) if len(residual_low) else np.nan
    max_low_residual = float(residual_low.max()) if len(residual_low) else np.nan

    summary = {
        "file": file_name,
        "success": True,
        "n_points": len(df),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "mean_low_freq_residual_ohm": mean_low_residual,
        "max_low_freq_residual_ohm": max_low_residual,
    }
    summary.update({k: float(v) for k, v in unpack_params(p_fit).items()})

    return {
        "summary": summary,
        "df": df,
        "p_fit": p_fit,
        "params_df": params_df,
        "metrics_df": metrics_df,
        "fusion_df": fusion_df,
        "curve_df": curve_df,
        "nyquist_png": fig_to_png_bytes(make_nyquist_figure(df, p_fit, file_name)),
        "residual_png": fig_to_png_bytes(make_residual_figure(df, p_fit, file_name)),
    }


def make_results_zip(results: list[dict]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            stem = safe_stem(r["summary"]["file"])
            zf.writestr(f"{stem}/fit_params.csv", dataframe_to_csv_bytes(r["params_df"]))
            zf.writestr(f"{stem}/arc_metrics.csv", dataframe_to_csv_bytes(r["metrics_df"]))
            zf.writestr(f"{stem}/fusion_metrics.csv", dataframe_to_csv_bytes(r["fusion_df"]))
            zf.writestr(f"{stem}/fit_curve.csv", dataframe_to_csv_bytes(r["curve_df"]))
            zf.writestr(f"{stem}/nyquist_fit.png", r["nyquist_png"])
            zf.writestr(f"{stem}/vertical_residual.png", r["residual_png"])
    return buf.getvalue()


with st.sidebar:
    st.header("Inputs")
    mode = st.radio("Mode", ["Single file", "Batch"], horizontal=True)

    if mode == "Single file":
        data_files = st.file_uploader("Upload EIS data", type=["mpr", "csv", "txt"], accept_multiple_files=False)
        if data_files is not None:
            data_files = [data_files]
    else:
        data_files = st.file_uploader("Upload multiple EIS files", type=["mpr", "csv", "txt"], accept_multiple_files=True)

    xml_file = st.file_uploader("Optional: upload ZFit XML initial model", type=["xml"])

    weight = st.selectbox(
        "Residual weighting",
        options=["unit", "sqrt_modulus", "modulus"],
        index=0,
        help="unit usually fits large arcs strongly; modulus gives relatively more weight to small/low-frequency features.",
    )

    run_fit = st.button("Fit EIS data", type="primary")

if not data_files:
    st.info("Upload one or more `.mpr`, `.csv`, or `.txt` EIS files to begin.")
    st.stop()

p0_dict: dict[str, float] = {}
if xml_file is not None:
    try:
        p0_dict = load_xml_params_from_bytes(xml_file.name, xml_file.getvalue())
        st.sidebar.success("Loaded XML initial parameters.")
    except Exception as exc:
        st.sidebar.warning(f"Could not read XML parameters; using defaults: {exc}")

p0 = pack_params(p0_dict)

st.subheader("Uploaded files")
st.write([f.name for f in data_files])

if not run_fit:
    st.warning("Click **Fit EIS data** in the sidebar to run the fitting.")
    st.stop()

results: list[dict] = []
failures: list[dict] = []
progress = st.progress(0)
status = st.empty()

for i, uploaded in enumerate(data_files, start=1):
    status.write(f"Fitting {uploaded.name} ({i}/{len(data_files)})...")
    try:
        results.append(fit_one_file(uploaded.name, uploaded.getvalue(), p0, weight))
    except Exception as exc:
        failures.append({"file": uploaded.name, "error": str(exc)})
    progress.progress(i / len(data_files))

status.empty()

if failures:
    st.error("Some files failed to fit.")
    st.dataframe(pd.DataFrame(failures), use_container_width=True)

if not results:
    st.stop()

st.success(f"Fit completed for {len(results)} file(s).")

summary_df = pd.DataFrame([r["summary"] for r in results])
params_all = pd.concat([r["params_df"] for r in results], ignore_index=True)
metrics_all = pd.concat([r["metrics_df"] for r in results], ignore_index=True)
fusion_all = pd.concat([r["fusion_df"] for r in results], ignore_index=True)
curves_all = pd.concat([r["curve_df"] for r in results], ignore_index=True)

st.subheader("Batch summary")
st.dataframe(summary_df, use_container_width=True)

selected_file = st.selectbox("Preview one fitted file", [r["summary"]["file"] for r in results])
selected = next(r for r in results if r["summary"]["file"] == selected_file)

left_col, right_col = st.columns([1.15, 1])
with left_col:
    st.subheader("Nyquist fit")
    st.image(selected["nyquist_png"])
with right_col:
    st.subheader("Vertical residual")
    st.image(selected["residual_png"])

st.subheader("Fit parameters")
st.dataframe(params_all, use_container_width=True)

st.subheader("Arc geometry metrics")
st.dataframe(metrics_all, use_container_width=True)

st.subheader("Fusion metrics")
st.dataframe(fusion_all, use_container_width=True)

st.subheader("Downloads")
d1, d2, d3, d4, d5 = st.columns(5)
d1.download_button(
    "Summary CSV",
    data=dataframe_to_csv_bytes(summary_df),
    file_name="batch_summary.csv",
    mime="text/csv",
)
d2.download_button(
    "Fit parameters CSV",
    data=dataframe_to_csv_bytes(params_all),
    file_name="batch_fit_params.csv",
    mime="text/csv",
)
d3.download_button(
    "Arc metrics CSV",
    data=dataframe_to_csv_bytes(metrics_all),
    file_name="batch_arc_metrics.csv",
    mime="text/csv",
)
d4.download_button(
    "Fit curves CSV",
    data=dataframe_to_csv_bytes(curves_all),
    file_name="batch_fit_curves.csv",
    mime="text/csv",
)
d5.download_button(
    "All results ZIP",
    data=make_results_zip(results),
    file_name="eis_batch_results.zip",
    mime="application/zip",
)
