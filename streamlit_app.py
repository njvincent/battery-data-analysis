"""
Streamlit web app for EIS fitting.

Put this file in the same folder as eis_fit.py, then run:
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
st.caption("Fit BioLogic/EC-Lab EIS data with R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4).")


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


def make_nyquist_figure(df: pd.DataFrame, p_fit: np.ndarray):
    freq = df["freq_hz"].to_numpy(float)
    z_fit = circuit_z(p_fit, freq)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(df["z_real_ohm"], df["minus_z_imag_ohm"], label="data")
    ax.plot(np.real(z_fit), -np.imag(z_fit), label="fit")
    ax.set_xlabel("Z' / Ω")
    ax.set_ylabel("-Z'' / Ω")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    return fig


def make_residual_figure(df: pd.DataFrame, p_fit: np.ndarray):
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
    fig.tight_layout()
    return fig


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


with st.sidebar:
    st.header("Inputs")
    data_file = st.file_uploader("Upload EIS data", type=["mpr", "csv", "txt"])
    xml_file = st.file_uploader("Optional: upload ZFit XML initial model", type=["xml"])

    weight = st.selectbox(
        "Residual weighting",
        options=["unit", "sqrt_modulus", "modulus"],
        index=0,
        help="unit usually fits large arcs strongly; modulus gives relatively more weight to small/low-frequency features.",
    )

    run_fit = st.button("Fit EIS data", type="primary")

if data_file is None:
    st.info("Upload a `.mpr`, `.csv`, or `.txt` EIS file to begin.")
    st.stop()

try:
    df = load_eis_from_bytes(data_file.name, data_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the EIS data file: {exc}")
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head(20), use_container_width=True)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Number of points", len(df))
col_b.metric("Max frequency / Hz", f"{df['freq_hz'].max():.4g}")
col_c.metric("Min frequency / Hz", f"{df['freq_hz'].min():.4g}")

if not run_fit:
    st.warning("Click **Fit EIS data** in the sidebar to run the fitting.")
    st.stop()

p0_dict: dict[str, float] = {}
if xml_file is not None:
    try:
        p0_dict = load_xml_params_from_bytes(xml_file.name, xml_file.getvalue())
    except Exception as exc:
        st.warning(f"Could not read XML parameters, using default initial values instead: {exc}")

p0 = pack_params(p0_dict)

with st.spinner("Fitting EIS data..."):
    try:
        p_fit, result = fit_eis(df, p0, weight=weight)
    except Exception as exc:
        st.error(f"Fit failed: {exc}")
        st.stop()

params_df = pd.DataFrame({
    "parameter": PARAM_ORDER,
    "initial": p0,
    "fit": p_fit,
})
params_df.loc[len(params_df)] = ["cost", np.nan, result.cost]
params_df.loc[len(params_df)] = ["nfev", np.nan, result.nfev]

fmin, fmax = float(df["freq_hz"].min()), float(df["freq_hz"].max())
metrics_df, fusion_df = arc_metrics(p_fit, fmin, fmax)

z_fit = circuit_z(p_fit, df["freq_hz"].to_numpy(float))
curve_df = df.copy()
curve_df["fit_z_real_ohm"] = np.real(z_fit)
curve_df["fit_z_imag_ohm"] = np.imag(z_fit)
curve_df["fit_minus_z_imag_ohm"] = -np.imag(z_fit)
curve_df["residual_real_ohm"] = curve_df["fit_z_real_ohm"] - curve_df["z_real_ohm"]
curve_df["residual_minus_imag_ohm"] = curve_df["fit_minus_z_imag_ohm"] - curve_df["minus_z_imag_ohm"]

st.success("Fit completed.")

left_col, right_col = st.columns([1.15, 1])
with left_col:
    st.subheader("Nyquist fit")
    st.pyplot(make_nyquist_figure(df, p_fit), clear_figure=True)

with right_col:
    st.subheader("Vertical residual")
    st.pyplot(make_residual_figure(df, p_fit), clear_figure=True)

st.subheader("Fit parameters")
st.dataframe(params_df, use_container_width=True)

st.subheader("Arc geometry metrics")
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Fusion metrics")
st.dataframe(fusion_df, use_container_width=True)

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "Download fit parameters CSV",
    data=dataframe_to_csv_bytes(params_df),
    file_name="fit_params.csv",
    mime="text/csv",
)
d2.download_button(
    "Download arc metrics CSV",
    data=dataframe_to_csv_bytes(metrics_df),
    file_name="arc_metrics.csv",
    mime="text/csv",
)
d3.download_button(
    "Download fit curve CSV",
    data=dataframe_to_csv_bytes(curve_df),
    file_name="fit_curve.csv",
    mime="text/csv",
)

with st.expander("Fitted parameter dictionary"):
    st.json(unpack_params(p_fit))
