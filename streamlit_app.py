#!/usr/bin/env python3
"""
Battery Data Analysis Streamlit dashboard.

This version only depends on files you already have in the repo:
    - eis_fit.py
    - eis_fit_batch.py can stay untouched for now
    - requirements.txt
    - streamlit_app.py

It imports the fitting logic from eis_fit.py and keeps Streamlit UI code here.
Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import io
import json
import html
import importlib.util
import subprocess
import tempfile
import zipfile
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.ticker import AutoMinorLocator

# Use the files that actually exist in your repo.
from eis_fit import (
    PARAM_ORDER,
    arc_metrics,
    circuit_z,
    fit_eis,
    pack_params,
    read_biologic_mpr_eis,
    read_csv_eis,
    read_zfit_xml,
)

import stripping_batch as stripping
import dqdv_batch as dqdv


# -----------------------------------------------------------------------------
# Data containers and wrappers around eis_fit.py
# -----------------------------------------------------------------------------


@dataclass
class FitResultBundle:
    name: str
    weight: str
    p0: np.ndarray
    p_fit: np.ndarray
    result: object
    curve_df: pd.DataFrame
    params_df: pd.DataFrame
    arc_df: pd.DataFrame
    fusion_df: pd.DataFrame


def _write_uploaded_to_temp(uploaded_file, suffix: str | None = None) -> Path:
    suffix = suffix or Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def read_uploaded_eis(uploaded_file) -> pd.DataFrame:
    """Read an uploaded .mpr/.csv/.txt by reusing path-based readers in eis_fit.py."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp_path = _write_uploaded_to_temp(uploaded_file, suffix=suffix)
    try:
        if suffix == ".mpr":
            return read_biologic_mpr_eis(tmp_path)
        return read_csv_eis(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def read_zfit_xml_uploaded(xml_file) -> dict[str, float]:
    if xml_file is None:
        return {}
    tmp_path = _write_uploaded_to_temp(xml_file, suffix=".xml")
    try:
        return read_zfit_xml(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def make_fit_bundle(name: str, df: pd.DataFrame, p0: np.ndarray, weight: str) -> FitResultBundle:
    p_fit, result = fit_eis(df, p0, weight=weight)
    z_fit = circuit_z(p_fit, df["freq_hz"].to_numpy(float))

    curve_df = df.copy()
    curve_df["fit_z_real_ohm"] = np.real(z_fit)
    curve_df["fit_z_imag_ohm"] = np.imag(z_fit)
    curve_df["fit_minus_z_imag_ohm"] = -np.imag(z_fit)
    curve_df["residual_real_ohm"] = curve_df["fit_z_real_ohm"] - curve_df["z_real_ohm"]
    curve_df["residual_minus_imag_ohm"] = curve_df["fit_minus_z_imag_ohm"] - curve_df["minus_z_imag_ohm"]

    params_df = pd.DataFrame({"parameter": PARAM_ORDER, "initial": p0, "fit": p_fit})
    params_df.loc[len(params_df)] = ["cost", np.nan, getattr(result, "cost", np.nan)]
    params_df.loc[len(params_df)] = ["nfev", np.nan, getattr(result, "nfev", np.nan)]

    fmin = float(df["freq_hz"].min())
    fmax = float(df["freq_hz"].max())
    arc_df, fusion_df = arc_metrics(p_fit, fmin, fmax)

    return FitResultBundle(
        name=name,
        weight=weight,
        p0=p0,
        p_fit=p_fit,
        result=result,
        curve_df=curve_df,
        params_df=params_df,
        arc_df=arc_df,
        fusion_df=fusion_df,
    )


# -----------------------------------------------------------------------------
# Summary and plotting helpers
# -----------------------------------------------------------------------------


def _fmt_num(x: float, unit: str = "", sig: int = 3) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        x = float(x)
    except Exception:
        return "—"
    return f"{x:.{sig}g}{(' ' + unit) if unit else ''}"


def fit_quality_summary(bundle: FitResultBundle, low_freq_cutoff: float) -> dict[str, object]:
    df = bundle.curve_df
    re = df["residual_real_ohm"].to_numpy(float)
    im = df["residual_minus_imag_ohm"].to_numpy(float)
    rmse_z = float(np.sqrt(np.nanmean(re**2 + im**2)))

    low = df[df["freq_hz"] <= float(low_freq_cutoff)]
    if len(low):
        low_bias = float(low["residual_minus_imag_ohm"].mean())
        low_max_abs = float(low["residual_minus_imag_ohm"].abs().max())
        low_points = int(len(low))
    else:
        low_bias = np.nan
        low_max_abs = np.nan
        low_points = 0

    fit_params = dict(zip(bundle.params_df["parameter"], bundle.params_df["fit"]))
    arc_df = bundle.arc_df
    fusion_df = bundle.fusion_df

    right_intercept = float(arc_df["right_intercept_ohm"].iloc[-1]) if len(arc_df) else np.nan
    arc3 = arc_df[arc_df["arc"] == 3]
    arc3_height = float(arc3["max_height_minus_im_ohm"].iloc[0]) if len(arc3) else np.nan
    arc3_depression = float(arc3["depression_ratio_height_over_radius"].iloc[0]) if len(arc3) else np.nan

    fusion23 = fusion_df[fusion_df["arc_pair"] == "2-3"]
    fusion23_val = (
        float(fusion23["fusion_index_overlap_over_narrower_FWHM"].iloc[0])
        if len(fusion23)
        else np.nan
    )

    note = "OK"
    if np.isfinite(low_bias) and np.isfinite(arc3_height):
        threshold = max(5.0, 0.05 * max(1.0, abs(arc3_height)))
        if abs(low_bias) > threshold:
            note = "fit high at low frequency" if low_bias > 0 else "fit low at low frequency"

    s4 = float(fit_params.get("s4", np.nan))
    a4 = float(fit_params.get("a4", np.nan))
    if np.isfinite(s4) and abs(s4) < 1e-8:
        note = "Warburg inactive; arc 3 is effective"
    if np.isfinite(a4) and (a4 > 0.995 or a4 < 0.055):
        note = "a4 near bound; check arc 3"

    return {
        "success": bool(getattr(bundle.result, "success", False)),
        "cost": float(getattr(bundle.result, "cost", np.nan)),
        "nfev": int(getattr(bundle.result, "nfev", -1)),
        "rmse_z_ohm": rmse_z,
        "low_f_points": low_points,
        "low_f_bias_ohm": low_bias,
        "low_f_max_abs_ohm": low_max_abs,
        "R1_ohm": float(fit_params.get("R1", np.nan)),
        "R2_ohm": float(fit_params.get("R2", np.nan)),
        "R3_ohm": float(fit_params.get("R3", np.nan)),
        "R4_ohm": float(fit_params.get("R4", np.nan)),
        "s4": s4,
        "a4": a4,
        "right_intercept_final_ohm": right_intercept,
        "arc3_height_ohm": arc3_height,
        "arc3_depression_ratio": arc3_depression,
        "fusion_2_3": fusion23_val,
        "note": note,
    }


def summary_row(file_name: str, df: pd.DataFrame, bundle: FitResultBundle, low_freq_cutoff: float) -> dict[str, object]:
    q = fit_quality_summary(bundle, low_freq_cutoff)
    return {
        "file": file_name,
        "weight": bundle.weight,
        "success": q["success"],
        "points": len(df),
        "f_min_Hz": float(df["freq_hz"].min()),
        "f_max_Hz": float(df["freq_hz"].max()),
        "RMSE_Z_ohm": q["rmse_z_ohm"],
        "low_f_bias_ohm": q["low_f_bias_ohm"],
        "low_f_max_abs_ohm": q["low_f_max_abs_ohm"],
        "R1_ohm": q["R1_ohm"],
        "R2_ohm": q["R2_ohm"],
        "R3_ohm": q["R3_ohm"],
        "R4_ohm": q["R4_ohm"],
        "R_total_span_ohm": q["right_intercept_final_ohm"],
        "arc3_height_ohm": q["arc3_height_ohm"],
        "arc3_depression": q["arc3_depression_ratio"],
        "fusion_2_3": q["fusion_2_3"],
        "note": q["note"],
    }


def display_metric_strip(bundle: FitResultBundle, df: pd.DataFrame, low_freq_cutoff: float) -> None:
    q = fit_quality_summary(bundle, low_freq_cutoff)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Points", f"{len(df)}")
    c2.metric("Freq. range", f"{_fmt_num(df['freq_hz'].min())}–{_fmt_num(df['freq_hz'].max())} Hz")
    c3.metric("RMSE |Z|", _fmt_num(q["rmse_z_ohm"], "Ω"))
    c4.metric("Low-f bias", _fmt_num(q["low_f_bias_ohm"], "Ω"))
    c5.metric("Final intercept", _fmt_num(q["right_intercept_final_ohm"], "Ω"))
    c6.metric("Arc 2–3 fusion", _fmt_num(q["fusion_2_3"]))

    if q["note"] != "OK":
        st.warning(f"Low-frequency check: {q['note']}")
    else:
        st.success("Fit completed. No major low-frequency warning from compact checks.")


def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in "f":
            out[col] = out[col].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.5g}"))
    return out


def make_nyquist_figure(df: pd.DataFrame, bundles: list[FitResultBundle], show_low_freq_labels: bool = False):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(df["z_real_ohm"], df["minus_z_imag_ohm"], s=24, label="data")
    for b in bundles:
        ax.plot(b.curve_df["fit_z_real_ohm"], b.curve_df["fit_minus_z_imag_ohm"], label=f"fit: {b.weight}")

    if show_low_freq_labels:
        low = df.nsmallest(min(8, len(df)), "freq_hz")
        for _, row in low.iterrows():
            ax.annotate(f"{row['freq_hz']:.2g} Hz", (row["z_real_ohm"], row["minus_z_imag_ohm"]), fontsize=8)

    ax.set_title("Nyquist plot")
    ax.set_xlabel("Z' / Ω")
    ax.set_ylabel("-Z'' / Ω")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_low_freq_figure(df: pd.DataFrame, bundles: list[FitResultBundle], cutoff_hz: float):
    mask = df["freq_hz"] <= float(cutoff_hz)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(df.loc[mask, "z_real_ohm"], df.loc[mask, "minus_z_imag_ohm"], s=28, label="data")
    for b in bundles:
        cdf = b.curve_df[b.curve_df["freq_hz"] <= float(cutoff_hz)]
        ax.plot(cdf["fit_z_real_ohm"], cdf["fit_minus_z_imag_ohm"], label=f"fit: {b.weight}")
    ax.set_title(f"Low-frequency zoom: f ≤ {cutoff_hz:g} Hz")
    ax.set_xlabel("Z' / Ω")
    ax.set_ylabel("-Z'' / Ω")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def make_zip_download(bundles: list[FitResultBundle]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for b in bundles:
            stem = Path(b.name).stem.replace(" ", "_") + f"_{b.weight}"
            zf.writestr(f"{stem}_fit_params.csv", b.params_df.to_csv(index=False))
            zf.writestr(f"{stem}_arc_metrics.csv", b.arc_df.to_csv(index=False))
            zf.writestr(f"{stem}_fusion_metrics.csv", b.fusion_df.to_csv(index=False))
            zf.writestr(f"{stem}_fit_curve.csv", b.curve_df.to_csv(index=False))
    return mem.getvalue()


# -----------------------------------------------------------------------------
# Sidebar widgets
# -----------------------------------------------------------------------------


def sidebar_fit_options(key_prefix: str):
    st.sidebar.header("Fit options")
    weight = st.sidebar.selectbox(
        "Primary weighting",
        ["unit", "sqrt_modulus", "modulus"],
        index=0,
        key=f"{key_prefix}_weight",
    )
    compare_weights = st.sidebar.checkbox(
        "Compare all three weightings",
        value=False,
        key=f"{key_prefix}_compare_weights",
    )
    show_low_freq_labels = st.sidebar.checkbox(
        "Label lowest-frequency points",
        value=False,
        key=f"{key_prefix}_labels",
    )
    low_freq_cutoff = st.sidebar.number_input(
        "Low-frequency cutoff / Hz",
        min_value=1e-9,
        max_value=1e9,
        value=0.1,
        format="%.6g",
        key=f"{key_prefix}_low_cutoff",
    )
    return weight, compare_weights, show_low_freq_labels, float(low_freq_cutoff)


def sidebar_initial_params(xml_file, key_prefix: str) -> np.ndarray:
    xml_params = {}
    if xml_file is not None:
        try:
            xml_params = read_zfit_xml_uploaded(xml_file)
            st.sidebar.success(f"Loaded {len(xml_params)} XML initial parameters.")
        except Exception as exc:
            st.sidebar.warning(f"Could not read XML: {exc}")

    st.sidebar.header("Initial parameters")
    default_p0 = pack_params(xml_params)
    editable = pd.DataFrame({"parameter": PARAM_ORDER, "initial_value": default_p0})
    edited = st.sidebar.data_editor(
        editable,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        key=f"{key_prefix}_p0_editor",
    )
    try:
        return np.array(edited["initial_value"].astype(float).to_list(), dtype=float)
    except Exception:
        st.sidebar.error("Initial parameter table contains non-numeric values; using defaults/XML values.")
        return default_p0


def weights_to_run(primary_weight: str, compare_weights: bool) -> list[str]:
    return ["unit", "sqrt_modulus", "modulus"] if compare_weights else [primary_weight]


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------


def render_eis_fit_page() -> None:
    st.title("EIS Fit")
    st.caption("Single-file EIS fitting using the model in `eis_fit.py`: R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4).")

    with st.sidebar:
        st.header("Input")
        data_file = st.file_uploader(
            "EIS data file (.mpr, .csv, .txt)",
            type=["mpr", "csv", "txt"],
            accept_multiple_files=False,
            key="single_data_file",
        )
        xml_file = st.file_uploader(
            "Optional EC-Lab ZFit XML for initial values",
            type=["xml"],
            accept_multiple_files=False,
            key="single_xml_file",
        )
        weight, compare_weights, show_low_freq_labels, low_freq_cutoff = sidebar_fit_options("single")
        p0 = sidebar_initial_params(xml_file, "single")

    if data_file is None:
        st.info("Upload one `.mpr`, `.csv`, or `.txt` EIS file to start.")
        st.markdown(
            """
            **Current model**

            `R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4)`

            The third arc should be treated as an effective low-frequency descriptor when the Warburg term is weak or poorly constrained.
            """
        )
        return

    try:
        df = read_uploaded_eis(data_file).sort_values("freq_hz", ascending=False).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not read `{data_file.name}`: {exc}")
        return

    weights = weights_to_run(weight, compare_weights)
    bundles: list[FitResultBundle] = []
    with st.spinner(f"Fitting {data_file.name}..."):
        for w in weights:
            try:
                bundles.append(make_fit_bundle(data_file.name, df, p0, w))
            except Exception as exc:
                st.error(f"Fit failed with weight={w}: {exc}")

    if not bundles:
        return

    primary = next((b for b in bundles if b.weight == weight), bundles[0])
    display_metric_strip(primary, df, low_freq_cutoff)

    tab_plot, tab_params, tab_metrics, tab_data = st.tabs(["Preview", "Fit parameters", "Arc/fusion metrics", "Data & downloads"])

    with tab_plot:
        c1, c2 = st.columns([1.25, 1.0])
        with c1:
            st.pyplot(make_nyquist_figure(df, bundles, show_low_freq_labels), clear_figure=True)
        with c2:
            st.pyplot(make_low_freq_figure(df, bundles, low_freq_cutoff), clear_figure=True)

    with tab_params:
        selected_weight = st.selectbox("Select weighting", [b.weight for b in bundles], index=[b.weight for b in bundles].index(primary.weight))
        b = next(x for x in bundles if x.weight == selected_weight)
        st.dataframe(b.params_df, use_container_width=True)

    with tab_metrics:
        selected_weight = st.selectbox("Select weighting for metrics", [b.weight for b in bundles], index=[b.weight for b in bundles].index(primary.weight))
        b = next(x for x in bundles if x.weight == selected_weight)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Arc geometry descriptors")
            st.dataframe(b.arc_df, use_container_width=True)
        with c2:
            st.caption("Fusion descriptors")
            st.dataframe(b.fusion_df, use_container_width=True)

    with tab_data:
        st.dataframe(primary.curve_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download primary fit curve CSV",
                data=primary.curve_df.to_csv(index=False),
                file_name=f"{Path(data_file.name).stem}_{primary.weight}_fit_curve.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                "Download all outputs ZIP",
                data=make_zip_download(bundles),
                file_name=f"{Path(data_file.name).stem}_eis_fit_outputs.zip",
                mime="application/zip",
            )


def render_eis_fit_batch_page() -> None:
    st.title("EIS Fit Batch")
    st.caption("Upload multiple EIS files. The page fits each file and shows a compact summary first, then lets you preview one selected file.")

    with st.sidebar:
        st.header("Input")
        data_files = st.file_uploader(
            "EIS data files (.mpr, .csv, .txt)",
            type=["mpr", "csv", "txt"],
            accept_multiple_files=True,
            key="batch_data_files",
        )
        xml_file = st.file_uploader(
            "Optional EC-Lab ZFit XML for initial values",
            type=["xml"],
            accept_multiple_files=False,
            key="batch_xml_file",
        )
        weight, compare_weights, show_low_freq_labels, low_freq_cutoff = sidebar_fit_options("batch")
        p0 = sidebar_initial_params(xml_file, "batch")

    if not data_files:
        st.info("Upload multiple `.mpr`, `.csv`, or `.txt` files to run batch fitting.")
        return

    weights = weights_to_run(weight, compare_weights)
    all_bundles: list[FitResultBundle] = []
    data_by_name: dict[str, pd.DataFrame] = {}
    bundles_by_name: dict[str, list[FitResultBundle]] = {}
    summary_rows: list[dict[str, object]] = []

    progress = st.progress(0)
    status = st.empty()

    for i, uploaded in enumerate(data_files, start=1):
        status.write(f"Fitting {uploaded.name} ({i}/{len(data_files)})...")
        try:
            df = read_uploaded_eis(uploaded).sort_values("freq_hz", ascending=False).reset_index(drop=True)
            data_by_name[uploaded.name] = df
            file_bundles: list[FitResultBundle] = []
            for w in weights:
                b = make_fit_bundle(uploaded.name, df, p0, w)
                file_bundles.append(b)
                all_bundles.append(b)
                summary_rows.append(summary_row(uploaded.name, df, b, low_freq_cutoff))
            bundles_by_name[uploaded.name] = file_bundles
        except Exception as exc:
            summary_rows.append(
                {
                    "file": uploaded.name,
                    "weight": ",".join(weights),
                    "success": False,
                    "points": np.nan,
                    "f_min_Hz": np.nan,
                    "f_max_Hz": np.nan,
                    "RMSE_Z_ohm": np.nan,
                    "low_f_bias_ohm": np.nan,
                    "low_f_max_abs_ohm": np.nan,
                    "R1_ohm": np.nan,
                    "R2_ohm": np.nan,
                    "R3_ohm": np.nan,
                    "R4_ohm": np.nan,
                    "R_total_span_ohm": np.nan,
                    "arc3_height_ohm": np.nan,
                    "arc3_depression": np.nan,
                    "fusion_2_3": np.nan,
                    "note": f"read/fit failed: {exc}",
                }
            )
        progress.progress(i / len(data_files))

    status.empty()
    progress.empty()

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Batch summary")
    st.dataframe(format_summary_table(summary_df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download batch summary CSV",
            data=summary_df.to_csv(index=False),
            file_name="eis_batch_summary.csv",
            mime="text/csv",
        )
    with c2:
        if all_bundles:
            st.download_button(
                "Download all fit outputs ZIP",
                data=make_zip_download(all_bundles),
                file_name="eis_batch_fit_outputs.zip",
                mime="application/zip",
            )

    if not bundles_by_name:
        return

    st.divider()
    st.subheader("Detailed preview")
    selected_file = st.selectbox("Select file to preview", list(bundles_by_name.keys()))
    file_bundles = bundles_by_name[selected_file]
    df = data_by_name[selected_file]
    primary = next((b for b in file_bundles if b.weight == weight), file_bundles[0])
    display_metric_strip(primary, df, low_freq_cutoff)

    tab_plot, tab_params, tab_metrics, tab_data = st.tabs(["Preview", "Fit parameters", "Arc/fusion metrics", "Data"])
    with tab_plot:
        c1, c2 = st.columns([1.25, 1.0])
        with c1:
            st.pyplot(make_nyquist_figure(df, file_bundles, show_low_freq_labels), clear_figure=True)
        with c2:
            st.pyplot(make_low_freq_figure(df, file_bundles, low_freq_cutoff), clear_figure=True)

    with tab_params:
        selected_weight = st.selectbox("Select weighting", [b.weight for b in file_bundles], key="batch_preview_params_weight")
        b = next(x for x in file_bundles if x.weight == selected_weight)
        st.dataframe(b.params_df, use_container_width=True)

    with tab_metrics:
        selected_weight = st.selectbox("Select weighting for metrics", [b.weight for b in file_bundles], key="batch_preview_metrics_weight")
        b = next(x for x in file_bundles if x.weight == selected_weight)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Arc geometry descriptors")
            st.dataframe(b.arc_df, use_container_width=True)
        with c2:
            st.caption("Fusion descriptors")
            st.dataframe(b.fusion_df, use_container_width=True)

    with tab_data:
        st.dataframe(primary.curve_df, use_container_width=True)


# -----------------------------------------------------------------------------
# Cycling capacity batch analysis
# -----------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """Convert a sample name into a safe filename."""
    name = str(name).strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name.strip("_") or "sample"


def shorten_label(text: str, max_len: int = 24) -> str:
    """
    Shorten long file/sample labels so legends and widgets do not squeeze plots.
    """
    text = str(text)
    max_len = int(max_len)

    if max_len <= 3 or len(text) <= max_len:
        return text

    keep_left = max(6, int(max_len * 0.62))
    keep_right = max(3, max_len - keep_left - 3)

    return f"{text[:keep_left]}...{text[-keep_right:]}"


def compact_widget_label(prefix: str, index: int, full_name: str, max_len: int = 28) -> str:
    """Create a short Streamlit widget label while keeping the full name in help text."""
    return f"{prefix} {index}: {shorten_label(full_name, max_len=max_len)}"


def find_capacity_sample_folders(root_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """
    Treat each direct subfolder under root_dir as one sample.
    """
    folders = []

    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.lower() == "ignore":
            continue

        if output_dir is not None:
            try:
                if item.resolve() == output_dir.resolve():
                    continue
            except Exception:
                pass

        folders.append(item)

    return folders


def find_capacity_excel_files(sample_dir: Path) -> list[Path]:
    """
    Recursively find Excel files under one sample folder.
    """
    files = []

    for path in sample_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in [".xlsx", ".xls"]:
            files.append(path)

    return sorted(files)


def infer_repeat_from_relative_path(sample_name: str, relative_path: str) -> str:
    """Infer a repeat label from root-relative path data/sample/repeat/file.xlsx."""
    parts = Path(str(relative_path)).parts
    if parts and parts[0] == sample_name:
        rest = parts[1:]
    else:
        rest = parts

    if len(rest) >= 2:
        return str(rest[0])
    if rest:
        return Path(rest[-1]).stem
    return Path(str(relative_path)).stem or "repeat"


def detect_cycle_column(df: pd.DataFrame) -> str | None:
    """
    Detect a cycle index column if it exists.
    If not found, cycle index will be generated automatically.
    """
    possible_names = [
        "Cycle Index",
        "Cycle",
        "Cycle No.",
        "Cycle Number",
        "Cycle Num",
        "Cycle_Index",
    ]

    for name in possible_names:
        if name in df.columns:
            return name

    return None


def read_one_capacity_file(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_capacity_retention: float | None,
) -> pd.DataFrame | None:
    """
    Read one cycling Excel file and return tidy plotting data.

    Returned columns:
        sample
        source_file
        relative_path
        cycle_index
        discharge_capacity_mAh
        capacity_retention_percent
        coulombic_efficiency_percent
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        st.warning(f"Could not read `{file_path.name}`: {exc}")
        return None

    missing_cols = [
        col for col in [capacity_col, efficiency_col] if col not in df.columns
    ]

    if missing_cols:
        st.warning(f"Skipping `{file_path.name}`: missing columns {missing_cols}")
        return None

    cycle_col = detect_cycle_column(df)

    data = df.iloc[int(skip_initial_rows):].copy()

    capacity = pd.to_numeric(data[capacity_col], errors="coerce")
    efficiency = pd.to_numeric(data[efficiency_col], errors="coerce")

    if cycle_col is not None:
        cycle_index = pd.to_numeric(data[cycle_col], errors="coerce")
    else:
        cycle_index = pd.Series(np.arange(len(data)), index=data.index)

    valid_mask = capacity.notna() & efficiency.notna() & cycle_index.notna()

    capacity = capacity[valid_mask].reset_index(drop=True)
    efficiency = efficiency[valid_mask].reset_index(drop=True)
    cycle_index = cycle_index[valid_mask].reset_index(drop=True)

    if len(capacity) == 0:
        st.warning(f"Skipping `{file_path.name}`: no valid numeric data.")
        return None

    initial_capacity = capacity.iloc[0]

    if pd.isna(initial_capacity) or initial_capacity == 0:
        st.warning(f"Skipping `{file_path.name}`: invalid initial capacity.")
        return None

    capacity_retention = capacity / initial_capacity * 100

    out = pd.DataFrame(
        {
            "sample": sample_name,
            "repeat": infer_repeat_from_relative_path(sample_name, str(file_path.relative_to(root_dir))),
            "source_file": file_path.name,
            "relative_path": str(file_path.relative_to(root_dir)),
            "cycle_index": cycle_index,
            "discharge_capacity_mAh": capacity,
            "capacity_retention_percent": capacity_retention,
            "coulombic_efficiency_percent": efficiency,
        }
    )

    if min_capacity_retention is not None:
        out = out[
            out["capacity_retention_percent"] >= float(min_capacity_retention)
        ].reset_index(drop=True)

    if out.empty:
        st.warning(f"Skipping `{file_path.name}`: no data after filtering.")
        return None

    return out



def read_one_capacity_file_silent(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_capacity_retention: float | None,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Pure reader for one cycling Excel file.

    This is the same parsing logic as read_one_capacity_file, but it returns an
    error string instead of writing Streamlit warnings. It is useful for cached
    previews and file-selection diagnostics.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        return None, f"Could not read file: {exc}"

    missing_cols = [col for col in [capacity_col, efficiency_col] if col not in df.columns]
    if missing_cols:
        return None, f"Missing columns: {missing_cols}"

    cycle_col = detect_cycle_column(df)
    data = df.iloc[int(skip_initial_rows):].copy()

    capacity = pd.to_numeric(data[capacity_col], errors="coerce")
    efficiency = pd.to_numeric(data[efficiency_col], errors="coerce")

    if cycle_col is not None:
        cycle_index = pd.to_numeric(data[cycle_col], errors="coerce")
    else:
        cycle_index = pd.Series(np.arange(len(data)), index=data.index)

    valid_mask = capacity.notna() & efficiency.notna() & cycle_index.notna()

    capacity = capacity[valid_mask].reset_index(drop=True)
    efficiency = efficiency[valid_mask].reset_index(drop=True)
    cycle_index = cycle_index[valid_mask].reset_index(drop=True)

    if len(capacity) == 0:
        return None, "No valid numeric data."

    initial_capacity = capacity.iloc[0]
    if pd.isna(initial_capacity) or initial_capacity == 0:
        return None, "Invalid initial capacity."

    capacity_retention = capacity / initial_capacity * 100

    out = pd.DataFrame(
        {
            "sample": sample_name,
            "repeat": infer_repeat_from_relative_path(sample_name, str(file_path.relative_to(root_dir))),
            "source_file": file_path.name,
            "relative_path": str(file_path.relative_to(root_dir)),
            "cycle_index": cycle_index,
            "discharge_capacity_mAh": capacity,
            "capacity_retention_percent": capacity_retention,
            "coulombic_efficiency_percent": efficiency,
        }
    )

    if min_capacity_retention is not None:
        out = out[out["capacity_retention_percent"] >= float(min_capacity_retention)].reset_index(drop=True)

    if out.empty:
        return None, "No data after filtering."

    return out, None


def parsed_excel_cache_dir(output_dir: Path, analysis_kind: str) -> Path:
    return output_dir / ".streamlit_parsed_excel_cache" / analysis_kind


def persistent_cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe_value(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_cache_paths(cache_dir: Path, cache_key: str) -> dict[str, Path]:
    return {
        "meta": cache_dir / f"{cache_key}.json",
        "parquet": cache_dir / f"{cache_key}.parquet",
        "csv": cache_dir / f"{cache_key}.csv",
    }


def parquet_cache_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def read_persistent_dataframe_cache(
    cache_dir: Path,
    cache_key: str,
) -> tuple[bool, pd.DataFrame | None, dict[str, object]]:
    paths = dataframe_cache_paths(cache_dir, cache_key)
    if not paths["meta"].exists():
        return False, None, {}
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except Exception:
        return False, None, {}

    if not bool(meta.get("ok", False)):
        return True, None, meta

    cache_format = str(meta.get("format") or "")
    read_order = [cache_format] if cache_format in {"parquet", "csv"} else ["parquet", "csv"]
    for fmt in read_order:
        path = paths.get(fmt)
        if path is None or not path.exists():
            continue
        try:
            if fmt == "parquet":
                return True, pd.read_parquet(path), meta
            return True, pd.read_csv(path), meta
        except Exception:
            continue
    return False, None, {}


def write_persistent_dataframe_cache(
    cache_dir: Path,
    cache_key: str,
    df: pd.DataFrame | None,
    error: str | None,
    extra_meta: dict[str, object] | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = dataframe_cache_paths(cache_dir, cache_key)
    meta: dict[str, object] = {
        "ok": df is not None,
        "error": error,
        "format": None,
    }
    if extra_meta:
        meta.update(json_safe_value(extra_meta))

    if df is not None:
        if parquet_cache_available():
            try:
                df.to_parquet(paths["parquet"], index=False)
                meta["format"] = "parquet"
            except Exception:
                df.to_csv(paths["csv"], index=False)
                meta["format"] = "csv"
        else:
            df.to_csv(paths["csv"], index=False)
            meta["format"] = "csv"

    paths["meta"].write_text(json.dumps(json_safe_value(meta), indent=2, sort_keys=True), encoding="utf-8")


def normalize_capacity_cache_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    for col in [
        "cycle_index",
        "discharge_capacity_mAh",
        "capacity_retention_percent",
        "coulombic_efficiency_percent",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def filter_capacity_retention(
    raw_df: pd.DataFrame | None,
    min_capacity_retention: float | None,
) -> tuple[pd.DataFrame | None, str | None]:
    if raw_df is None:
        return None, "No valid numeric data."
    out = raw_df.copy()
    if min_capacity_retention is not None:
        out = out[
            pd.to_numeric(out["capacity_retention_percent"], errors="coerce") >= float(min_capacity_retention)
        ].reset_index(drop=True)
    if out.empty:
        return None, "No data after filtering."
    return out, None


def capacity_raw_cache_key(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
) -> str:
    stat = file_path.stat()
    try:
        relative_path = str(file_path.relative_to(root_dir))
    except ValueError:
        relative_path = file_path.name
    return persistent_cache_key(
        {
            "kind": "cycling_raw_v1",
            "file_path": str(file_path),
            "relative_path": relative_path,
            "sample_name": sample_name,
            "root_dir": str(root_dir),
            "sheet_name": sheet_name,
            "capacity_col": capacity_col,
            "efficiency_col": efficiency_col,
            "skip_initial_rows": int(skip_initial_rows),
            "file_size": int(stat.st_size),
            "file_mtime": float(stat.st_mtime),
        }
    )


def load_persistent_capacity_raw_file(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame | None, str | None]:
    cache_key = capacity_raw_cache_key(
        file_path=file_path,
        sample_name=sample_name,
        root_dir=root_dir,
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
    )
    hit, cached_df, meta = read_persistent_dataframe_cache(cache_dir, cache_key)
    if hit:
        return normalize_capacity_cache_df(cached_df), meta.get("error") if isinstance(meta, dict) else None

    raw_df, error = read_one_capacity_file_silent(
        file_path=file_path,
        sample_name=sample_name,
        root_dir=root_dir,
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
        min_capacity_retention=None,
    )
    raw_df = normalize_capacity_cache_df(raw_df)
    write_persistent_dataframe_cache(cache_dir, cache_key, raw_df, error)
    return raw_df, error


def load_persistent_capacity_file(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_retention: float | None,
    cache_dir: Path,
) -> tuple[pd.DataFrame | None, str | None]:
    raw_df, raw_error = load_persistent_capacity_raw_file(
        file_path=file_path,
        sample_name=sample_name,
        root_dir=root_dir,
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
        cache_dir=cache_dir,
    )
    if raw_df is None:
        return None, raw_error
    filtered_df, filter_error = filter_capacity_retention(raw_df, min_retention)
    return filtered_df, filter_error


@st.cache_data(show_spinner=False)
def cached_read_capacity_file(
    file_path_str: str,
    sample_name: str,
    root_dir_str: str,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_capacity_retention: float | None,
    file_size: int,
    file_mtime: float,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Cached one-file reader.

    file_size and file_mtime are intentionally included in the cache key, so if
    an Excel file changes on disk the preview cache is invalidated automatically.
    """
    _ = file_size, file_mtime
    return read_one_capacity_file_silent(
        file_path=Path(file_path_str),
        sample_name=sample_name,
        root_dir=Path(root_dir_str),
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
        min_capacity_retention=min_capacity_retention,
    )


def load_cached_capacity_file(
    file_path: Path,
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_retention: float | None,
    persistent_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    if persistent_cache_dir is not None:
        return load_persistent_capacity_file(
            file_path=file_path,
            sample_name=sample_name,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_retention=min_retention,
            cache_dir=persistent_cache_dir,
        )

    stat = file_path.stat()
    return cached_read_capacity_file(
        file_path_str=str(file_path),
        sample_name=sample_name,
        root_dir_str=str(root_dir),
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
        min_capacity_retention=min_retention,
        file_size=int(stat.st_size),
        file_mtime=float(stat.st_mtime),
    )


def stable_key_part(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


def file_record_signature(record: dict[str, object]) -> dict[str, object]:
    """Return stable file identity for bulk-preview cache invalidation."""
    path = Path(record["path"])
    try:
        stat = path.stat()
        size = int(stat.st_size)
        mtime = float(stat.st_mtime)
    except Exception:
        size = -1
        mtime = -1.0
    return {
        "relative_path": str(record.get("relative_path", path.name)),
        "size": size,
        "mtime": mtime,
    }


def cycling_file_include_key(sample_name: str, relative_path: str) -> str:
    return f"cycling_include_{stable_key_part(sample_name)}_{stable_key_part(relative_path)}"


def capacity_file_records(sample_name: str, sample_dir: Path, root_dir: Path) -> list[dict[str, object]]:
    """Fast file index: list Excel files without reading them."""
    records = []
    for file_path in find_capacity_excel_files(sample_dir):
        relative_path = str(file_path.relative_to(root_dir))
        records.append(
            {
                "sample": sample_name,
                "repeat": infer_repeat_from_relative_path(sample_name, relative_path),
                "source_file": file_path.name,
                "relative_path": relative_path,
                "path": file_path,
            }
        )
    return records


def selected_relative_paths_for_sample(
    sample_name: str,
    sample_dir: Path,
    root_dir: Path,
    manual_selection: bool,
) -> list[str] | None:
    """
    Return selected relative paths for one sample.

    None means use all files. A list means use exactly those files.

    Saved selections are treated as the source of truth. This prevents final
    processing from falling back to all files after the user has reviewed and
    saved a sample. If a sample has not been saved yet, the live checkbox state
    is used as a fallback.
    """
    if not manual_selection:
        return None

    ensure_cycling_selection_store()
    records = capacity_file_records(sample_name, sample_dir, root_dir)
    saved = st.session_state["cycling_saved_selection"].get(sample_name)

    selected = []
    for record in records:
        rel = str(record["relative_path"])
        if saved is not None:
            include = bool(saved.get(rel, True))
        else:
            key = cycling_file_include_key(sample_name, rel)
            include = bool(st.session_state.get(key, True))

        if include:
            selected.append(rel)

    return selected


def ensure_cycling_selection_store() -> None:
    """Ensure the persistent selection containers exist in Streamlit session state."""
    if "cycling_saved_selection" not in st.session_state:
        st.session_state["cycling_saved_selection"] = {}


def save_selection_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> None:
    """
    Save the current checkbox values for one sample.

    The saved copy is separate from the live checkbox state, so users can switch
    between samples and still restore or confirm earlier choices.
    """
    ensure_cycling_selection_store()
    saved = {}

    for record in file_records:
        rel = str(record["relative_path"])
        key = cycling_file_include_key(sample_name, rel)
        saved[rel] = bool(st.session_state.get(key, True))

    st.session_state["cycling_saved_selection"][sample_name] = saved


def sync_cycling_checkbox_to_saved(sample_name: str, relative_path: str, checkbox_key: str) -> None:
    """Persist one cycling checkbox immediately so page navigation does not lose it."""
    ensure_cycling_selection_store()
    all_saved = dict(st.session_state["cycling_saved_selection"])
    saved = dict(all_saved.get(sample_name, {}))
    saved[relative_path] = bool(st.session_state.get(checkbox_key, True))
    all_saved[sample_name] = saved
    st.session_state["cycling_saved_selection"] = all_saved


def restore_saved_selection_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> bool:
    """Restore saved checkbox values for one sample. Returns True if restored."""
    ensure_cycling_selection_store()
    saved = st.session_state["cycling_saved_selection"].get(sample_name)

    if not saved:
        return False

    for record in file_records:
        rel = str(record["relative_path"])
        key = cycling_file_include_key(sample_name, rel)
        st.session_state[key] = bool(saved.get(rel, True))

    return True


def saved_selection_summary(sample_name: str, file_records: list[dict[str, object]]) -> str:
    """Return a compact saved-selection status string for one sample."""
    ensure_cycling_selection_store()
    saved = st.session_state["cycling_saved_selection"].get(sample_name)

    if not saved:
        return "No saved selection yet."

    total = len(file_records)
    selected = sum(bool(saved.get(str(record["relative_path"]), True)) for record in file_records)
    return f"Saved: {selected} / {total} files included."


CYCLING_PLOT_MODE_SINGLE = "Single sample repeat overlay"
CYCLING_PLOT_MODE_COMPARE = "Compare all selected samples"
CYCLING_COMPARE_MODE_ALIASES = {
    CYCLING_PLOT_MODE_COMPARE,
    "Compare samples by one repeat",
}


def is_cycling_compare_mode(plot_mode: object) -> bool:
    return str(plot_mode) in CYCLING_COMPARE_MODE_ALIASES


def normalize_cycling_plot_mode(plot_mode: object) -> str:
    return CYCLING_PLOT_MODE_COMPARE if is_cycling_compare_mode(plot_mode) else CYCLING_PLOT_MODE_SINGLE


def save_current_cycling_selection_and_advance(
    current_sample: str,
    selected_samples: list[str],
    folder_map: dict[str, Path],
    root_dir: Path,
) -> None:
    """Save the current sample selection and advance the review flow.

    The button at the bottom of the file-preview page uses this as a callback.
    It saves only the sample the user just reviewed. If more selected samples
    remain, the app opens the next sample for review. If the current sample is
    the last selected sample, the app moves to the style-preview step.
    """
    ensure_cycling_selection_store()

    if current_sample not in selected_samples:
        return

    records = capacity_file_records(current_sample, folder_map[current_sample], root_dir)
    save_selection_for_sample(current_sample, records)

    current_index = selected_samples.index(current_sample)
    if current_index < len(selected_samples) - 1:
        st.session_state["cycling_inspect_sample"] = selected_samples[current_index + 1]
        st.session_state["cycling_workflow_step"] = "1. Data preview & file selection"
    else:
        apply_cycling_style_defaults_for_preview(selected_samples)
        st.session_state["cycling_workflow_step"] = "2. Style preview"


def set_cycling_workflow_step(step: str) -> None:
    """Set the cycling workflow step from a button callback."""
    st.session_state["cycling_workflow_step"] = step


def save_all_cycling_selections_and_go_style(
    selected_samples: list[str],
    folder_map: dict[str, Path],
    root_dir: Path,
) -> None:
    """Save all currently visible cycling selections and advance to style preview."""
    ensure_cycling_selection_store()
    for sample in selected_samples:
        if sample in folder_map:
            save_selection_for_sample(sample, capacity_file_records(sample, folder_map[sample], root_dir))
    apply_cycling_style_defaults_for_preview(selected_samples)
    st.session_state["cycling_workflow_step"] = "2. Style preview"


def reset_cycling_visual_style_defaults() -> None:
    """Keep cycling plot-mode changes from carrying stale visual styling."""
    defaults = {
        "cycling_plot_title": "{sample}",
        "cycling_x_label": "Cycle Index",
        "cycling_cap_y_label": "Capacity Retention (%)",
        "cycling_ce_y_label": "Coulombic Efficiency (%)",
        "cycling_show_legend": True,
        "cycling_legend_position": "Top",
        "cycling_legend_title": "Files",
        "cycling_legend_label_max_len": 24,
        "cycling_legend_columns": 3,
        "cycling_auto_x_range": True,
        "cycling_x_min": 0.0,
        "cycling_x_max": 500.0,
        "cycling_cap_y_min": 75.0,
        "cycling_cap_y_max": 110.0,
        "cycling_ce_y_min": 90.0,
        "cycling_ce_y_max": 100.5,
        "cycling_palette_name": "Set2 + Dark2 + tab20",
        "cycling_marker_size": 80,
        "cycling_fig_width": 9.5,
        "cycling_fig_height": 5.8,
        "cycling_dpi": 300,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def cycling_style_defaults_signature(selected_samples: list[str]) -> str:
    return hashlib.sha1(
        repr(
            {
                "selected_samples": list(selected_samples),
                "defaults_version": "cycling_visual_defaults_v3",
            }
        ).encode("utf-8")
    ).hexdigest()


def apply_cycling_style_defaults_for_preview(selected_samples: list[str]) -> None:
    reset_cycling_visual_style_defaults()
    st.session_state["cycling_style_defaults_signature"] = cycling_style_defaults_signature(selected_samples)


def applescript_string(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def send_system_notification(title: str, body: str) -> bool:
    """Best-effort OS notification on the machine running Streamlit."""
    system = platform.system()
    try:
        if system == "Darwin":
            script = f"display notification {applescript_string(body)} with title {applescript_string(title)}"
            subprocess.run(["osascript", "-e", script], check=False, timeout=5)
            return True
        if system == "Linux" and shutil.which("notify-send"):
            subprocess.run(["notify-send", title, body], check=False, timeout=5)
            return True
    except Exception:
        return False
    return False


def notify_load_all_complete(
    notification_id: str,
    title: str,
    body: str,
    enabled: bool = True,
) -> None:
    """Show a one-time Streamlit toast plus best-effort OS/browser notifications."""
    if not enabled or st.session_state.get(notification_id):
        return
    st.session_state[notification_id] = True

    if hasattr(st, "toast"):
        st.toast(body, icon="✅")

    send_system_notification(title, body)

    title_json = json.dumps(title)
    body_json = json.dumps(body)
    notification_id_json = json.dumps(notification_id)
    components.html(
        f"""
        <script>
        (function() {{
          const id = {notification_id_json};
          const title = {title_json};
          const body = {body_json};
          const storageKey = "streamlit_load_notification_" + id;
          if (window.sessionStorage && window.sessionStorage.getItem(storageKey)) {{
            return;
          }}
          if (window.sessionStorage) {{
            window.sessionStorage.setItem(storageKey, "1");
          }}
          if (!("Notification" in window)) {{
            return;
          }}
          function showNotification() {{
            try {{
              new Notification(title, {{ body: body }});
            }} catch (err) {{}}
          }}
          if (Notification.permission === "granted") {{
            showNotification();
          }} else if (Notification.permission !== "denied") {{
            Notification.requestPermission().then(function(permission) {{
              if (permission === "granted") {{
                showNotification();
              }}
            }});
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def rerun_streamlit_app() -> None:
    """Rerun the app across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def save_cycling_style_and_go_final(selected_samples: list[str], all_sample_names: list[str]) -> None:
    """Snapshot current style controls, then switch to Final output.

    Without an explicit snapshot, the first render of the final-output page can
    sometimes use stale widget values from the previous rerun, which leads to
    incorrect style or axis ranges that hide the points.
    """
    style_snapshot = {
        "plot_title": st.session_state.get("cycling_plot_title", "{sample}"),
        "x_label": st.session_state.get("cycling_x_label", "Cycle Index"),
        "cap_y_label": st.session_state.get("cycling_cap_y_label", "Capacity Retention (%)"),
        "ce_y_label": st.session_state.get("cycling_ce_y_label", "Coulombic Efficiency (%)"),
        "show_legend": bool(st.session_state.get("cycling_show_legend", True)),
        "legend_position": st.session_state.get("cycling_legend_position", "Top"),
        "legend_title": st.session_state.get("cycling_legend_title", "Files"),
        "legend_label_max_len": int(st.session_state.get("cycling_legend_label_max_len", 24)),
        "legend_columns": int(st.session_state.get("cycling_legend_columns", 3)),
        "auto_x_range": bool(st.session_state.get("cycling_auto_x_range", True)),
        "x_min": float(st.session_state.get("cycling_x_min", 0.0)),
        "x_max": float(st.session_state.get("cycling_x_max", 500.0)),
        "cap_y_min": float(st.session_state.get("cycling_cap_y_min", 75.0)),
        "cap_y_max": float(st.session_state.get("cycling_cap_y_max", 110.0)),
        "ce_y_min": float(st.session_state.get("cycling_ce_y_min", 90.0)),
        "ce_y_max": float(st.session_state.get("cycling_ce_y_max", 100.5)),
        "palette_name": st.session_state.get("cycling_palette_name", "Set2 + Dark2 + tab20"),
        "plot_mode": normalize_cycling_plot_mode(st.session_state.get("cycling_plot_mode", CYCLING_PLOT_MODE_SINGLE)),
        "compare_repeat": st.session_state.get("cycling_compare_repeat", ""),
        "compare_samples": list(st.session_state.get("cycling_compare_samples", selected_samples)),
        "marker_size": int(st.session_state.get("cycling_marker_size", 80)),
        "fig_width": float(st.session_state.get("cycling_fig_width", 9.5)),
        "fig_height": float(st.session_state.get("cycling_fig_height", 5.8)),
        "dpi": int(st.session_state.get("cycling_dpi", 300)),
    }

    palette_colors = palette_to_hex_colors(str(style_snapshot["palette_name"]), len(all_sample_names))
    palette_color_map = {sample: palette_colors[i] for i, sample in enumerate(all_sample_names)}
    style_snapshot["sample_colors"] = {
        sample: st.session_state.get(
            f"cycling_color_{safe_filename(sample)}",
            palette_color_map[sample],
        )
        for sample in selected_samples
    }

    st.session_state["cycling_saved_style"] = style_snapshot
    st.session_state["cycling_workflow_step"] = "3. Final output"


def selected_relative_paths_for_sample_saved_first(
    sample_name: str,
    sample_dir: Path,
    root_dir: Path,
    manual_selection: bool,
) -> list[str] | None:
    """Return selected file paths, preferring saved selections over live checkbox values.

    None means all files are allowed. A list means use exactly those root-relative
    paths. Invalid files should already have been saved as False during preview.
    """
    if not manual_selection:
        return None

    ensure_cycling_selection_store()
    records = capacity_file_records(sample_name, sample_dir, root_dir)
    saved = st.session_state["cycling_saved_selection"].get(sample_name)

    selected: list[str] = []
    for record in records:
        rel = str(record["relative_path"])
        key = cycling_file_include_key(sample_name, rel)
        if saved is not None:
            include = bool(saved.get(rel, st.session_state.get(key, True)))
        else:
            include = bool(st.session_state.get(key, True))
        if include:
            selected.append(rel)

    return selected



def capacity_file_summary_row(
    file_df: pd.DataFrame | None,
    record: dict[str, object],
    selected: bool,
    error: str | None = None,
) -> dict[str, object]:
    if file_df is None or file_df.empty:
        return {
            "include": selected,
            "file": record["source_file"],
            "relative_path": record["relative_path"],
            "points": 0,
            "max_cycle": np.nan,
            "final_retention_%": np.nan,
            "min_retention_%": np.nan,
            "mean_CE_%": np.nan,
            "status": error or "not loaded",
        }

    ordered = file_df.sort_values("cycle_index")
    return {
        "include": selected,
        "file": record["source_file"],
        "relative_path": record["relative_path"],
        "points": int(len(file_df)),
        "max_cycle": float(ordered["cycle_index"].max()),
        "final_retention_%": float(ordered["capacity_retention_percent"].iloc[-1]),
        "min_retention_%": float(ordered["capacity_retention_percent"].min()),
        "mean_CE_%": float(ordered["coulombic_efficiency_percent"].mean()),
        "status": "OK",
    }


def infer_operator_from_path(file_path: Path) -> str:
    """Best-effort extraction of operator name from a file path.

    The app cannot know the operator unless it is encoded in the file/folder
    name. This function looks for patterns such as ``operator_Alice`` or
    ``op-Alice``. If nothing is found, it leaves the field blank.
    """
    path_text = str(file_path)
    patterns = [
        r"(?:operator|oper|op)[-_\s:=]+([A-Za-z0-9]+)",
        r"([A-Za-z]+)[-_\s]*(?:operator|oper)",
    ]
    for pat in patterns:
        match = re.search(pat, path_text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1))
    return ""


def file_modified_time_string(file_path: Path) -> str:
    """Return the file modification time as a compact timestamp string."""
    try:
        return pd.Timestamp.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def compute_capacity_selection_metrics(
    file_df: pd.DataFrame | None,
    record: dict[str, object],
    target_retention: float = 80.0,
    tolerance: float = 2.0,
) -> dict[str, object]:
    """Compute compact per-file metrics for preview cards and summary CSV.

    Definitions used here:
    - ICE (%): coulombic efficiency at the first valid cycle.
    - Cycle Life: the first cycle whose capacity retention is within
      target_retention ± tolerance, i.e. 80 ± 2% by default.
    - ACE (%): average CE from the first valid cycle through the ACE cycle.
    - ACE cycle: the cycle used as the upper bound for ACE. If Cycle Life is
      available, ACE cycle equals Cycle Life. If the file has not reached
      80 ± 2%, ACE cycle is the last available cycle and Note explains this.
    """
    file_path = Path(record.get("path", ""))
    base = {
        "Sample": record.get("sample", ""),
        "Repeat": record.get("repeat", ""),
        "ICE (%)": np.nan,
        "Cycle Life": np.nan,
        "ACE (%)": np.nan,
        "ACE cycle": np.nan,
        "Time": file_modified_time_string(file_path) if file_path else "",
        "Operator": infer_operator_from_path(file_path) if file_path else "",
        "File name": record.get("source_file", ""),
        "Relative path": record.get("relative_path", ""),
        "Note": "",
    }

    if file_df is None or file_df.empty:
        base["Note"] = "invalid"
        return base

    df = file_df.sort_values("cycle_index").copy()
    retention = pd.to_numeric(df["capacity_retention_percent"], errors="coerce")
    ce = pd.to_numeric(df["coulombic_efficiency_percent"], errors="coerce")
    cycles = pd.to_numeric(df["cycle_index"], errors="coerce")
    valid = retention.notna() & ce.notna() & cycles.notna()
    df = df.loc[valid].copy()
    if df.empty:
        base["Note"] = "invalid"
        return base

    df = df.sort_values("cycle_index")
    base["ICE (%)"] = float(df["coulombic_efficiency_percent"].iloc[0])

    lower = float(target_retention) - float(tolerance)
    upper = float(target_retention) + float(tolerance)
    in_window = df[
        (df["capacity_retention_percent"] >= lower)
        & (df["capacity_retention_percent"] <= upper)
    ].sort_values("cycle_index")

    if len(in_window):
        cycle_life = float(in_window["cycle_index"].iloc[0])
        base["Cycle Life"] = cycle_life
        ace_cycle = cycle_life
    else:
        max_cycle = float(df["cycle_index"].max())
        ace_cycle = max_cycle
        min_retention = float(df["capacity_retention_percent"].min())
        if min_retention > upper:
            base["Note"] = "running"
        elif min_retention < lower:
            base["Note"] = "not finished"
        else:
            base["Note"] = "not finished"

    ace_rows = df[df["cycle_index"] <= ace_cycle]
    if ace_rows.empty:
        ace_rows = df
    base["ACE (%)"] = float(ace_rows["coulombic_efficiency_percent"].mean())
    base["ACE cycle"] = float(ace_cycle)
    return base


def format_metric_value(value: object, digits: int = 3) -> str:
    """Compact formatting for preview-card metrics."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}g}"
    except Exception:
        return "—" if value in [None, ""] else str(value)


def preview_metric_text(value: object, suffix: str = "", digits: int = 3) -> str:
    """Format one preview-card metric with the same compact style everywhere."""
    text = format_metric_value(value, digits=digits)
    return f"{text}{suffix}" if text != "—" else text


def render_preview_metric_grid(metrics: list[tuple[str, str]]) -> None:
    """Render the shared two-column metric grid used by compact preview cards."""
    cells = "\n".join(
        f"<span style='white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'><b>{label}</b>: {value}</span>"
        for label, value in metrics
    )
    st.markdown(
        f"""
        <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size:0.74rem; line-height:1.28; height:42px; overflow:hidden; margin-top:-0.25rem;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; column-gap:0.55rem;">
                {cells}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview_note(note_text: str) -> None:
    """Render the shared note line below compact preview-card metrics."""
    note_text = str(note_text or "").strip()
    if note_text:
        safe_note = html.escape(note_text)
        st.markdown(
            f"<div title='{safe_note}' style='font-size:0.72rem; color:rgba(120,120,120,0.95); height:18px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;'>Note: {safe_note}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)


def format_selected_file_summary_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Make the selected-file summary easier to read in Streamlit."""
    out = summary_df.copy()
    for col in ["ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else float(f"{float(x):.5g}"))
    return out


def load_raw_capacity_metrics_df(
    record: dict[str, object],
    sample_name: str,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    persistent_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    """Load unfiltered data for metrics such as 80±2% cycle life."""
    return load_cached_capacity_file(
        file_path=record["path"],
        sample_name=sample_name,
        root_dir=root_dir,
        sheet_name=sheet_name,
        capacity_col=capacity_col,
        efficiency_col=efficiency_col,
        skip_initial_rows=int(skip_initial_rows),
        min_retention=None,
        persistent_cache_dir=persistent_cache_dir,
    )


def load_capacity_sample_file_summary(
    sample_name: str,
    sample_dir: Path,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    selected_relative_paths: list[str] | None,
    persistent_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a per-file metric summary for selected files in one sample."""
    records = capacity_file_records(sample_name, sample_dir, root_dir)
    if selected_relative_paths is not None:
        selected_set = set(selected_relative_paths)
        records = [r for r in records if str(r["relative_path"]) in selected_set]

    rows = []
    for record in records:
        raw_df, error = load_raw_capacity_metrics_df(
            record=record,
            sample_name=sample_name,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            persistent_cache_dir=persistent_cache_dir,
        )
        row = compute_capacity_selection_metrics(raw_df, record)
        if error and not row.get("Note"):
            row["Note"] = error
        rows.append(row)

    return pd.DataFrame(rows)


def load_capacity_current_sample_summary_from_entries(
    entries: list[tuple[dict[str, object], pd.DataFrame | None, pd.DataFrame | None, str | None]],
) -> pd.DataFrame:
    """Build a summary table for selected files already loaded on the preview page."""
    rows = []
    for record, _plot_df, raw_df, error in entries:
        rel = str(record["relative_path"])
        key = cycling_file_include_key(str(record["sample"]), rel)
        if not bool(st.session_state.get(key, False)):
            continue
        row = compute_capacity_selection_metrics(raw_df, record)
        if error and not row.get("Note"):
            row["Note"] = error
        rows.append(row)
    return pd.DataFrame(rows)


def load_capacity_all_selected_file_summary(
    selected_samples: list[str],
    folder_map: dict[str, Path],
    root_dir: Path,
    manual_selection: bool,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    persistent_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a summary table for every selected file across selected samples."""
    frames = []
    for sample_name in selected_samples:
        selected_paths = selected_relative_paths_for_sample_saved_first(
            sample_name,
            folder_map[sample_name],
            root_dir,
            manual_selection=manual_selection,
        )
        frame = load_capacity_sample_file_summary(
            sample_name=sample_name,
            sample_dir=folder_map[sample_name],
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            selected_relative_paths=selected_paths,
            persistent_cache_dir=persistent_cache_dir,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Sample", "Repeat", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Relative path", "Note"])
    return pd.concat(frames, ignore_index=True)


def load_capacity_sample_plot_data(
    sample_name: str,
    sample_dir: Path,
    root_dir: Path,
    sheet_name: str,
    capacity_col: str,
    efficiency_col: str,
    skip_initial_rows: int,
    min_retention: float | None,
    top_n_value: int | None,
    selected_relative_paths: list[str] | None = None,
    persistent_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """
    Load and optionally filter Excel cycling files for one sample.

    selected_relative_paths:
        None -> use all detected files.
        list -> use only files whose root-relative paths are in the list.
    """
    excel_files = find_capacity_excel_files(sample_dir)
    excel_file_count = len(excel_files)

    if not excel_files:
        return None, 0

    if selected_relative_paths is not None:
        selected_set = set(selected_relative_paths)
        excel_files = [p for p in excel_files if str(p.relative_to(root_dir)) in selected_set]

    if not excel_files:
        return None, excel_file_count

    sample_dfs = []
    for file_path in excel_files:
        one_df, error = load_cached_capacity_file(
            file_path=file_path,
            sample_name=sample_name,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_retention=min_retention,
            persistent_cache_dir=persistent_cache_dir,
        )

        if one_df is not None:
            sample_dfs.append(one_df)
        elif error:
            st.warning(f"Skipping `{file_path.name}`: {error}")

    if not sample_dfs:
        return None, excel_file_count

    plot_df = pd.concat(sample_dfs, ignore_index=True)

    if top_n_value is not None:
        scores = (
            plot_df.groupby("relative_path")["capacity_retention_percent"]
            .sum()
            .sort_values(ascending=False)
        )
        keep_paths = scores.head(top_n_value).index
        plot_df = plot_df[plot_df["relative_path"].isin(keep_paths)].copy()

    if plot_df.empty:
        return None, excel_file_count

    return plot_df, excel_file_count


def read_cycling_preview_job_worker(
    args: tuple[
        str,
        dict[str, object],
        Path,
        str,
        str,
        str,
        int,
        float | None,
        Path | None,
    ],
) -> tuple[str, dict[str, object], pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Pickle-safe cycling preview worker for both threads and processes."""
    (
        sample,
        record,
        root_dir,
        sheet_name,
        capacity_col,
        efficiency_col,
        skip_initial_rows,
        preview_min_retention,
        persistent_cache_dir,
    ) = args

    if persistent_cache_dir is None:
        file_df, error = read_one_capacity_file_silent(
            file_path=Path(record["path"]),
            sample_name=sample,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_capacity_retention=preview_min_retention,
        )
        raw_df, raw_error = read_one_capacity_file_silent(
            file_path=Path(record["path"]),
            sample_name=sample,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_capacity_retention=None,
        )
    else:
        file_df, error = load_cached_capacity_file(
            file_path=Path(record["path"]),
            sample_name=sample,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_retention=preview_min_retention,
            persistent_cache_dir=persistent_cache_dir,
        )
        raw_df, raw_error = load_raw_capacity_metrics_df(
            record=record,
            sample_name=sample,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            persistent_cache_dir=persistent_cache_dir,
        )

    return sample, record, file_df, raw_df, error or raw_error


def capacity_auto_x_limit(max_cycle: float) -> int:
    """
    Choose a clean x-axis upper limit.
    """
    if max_cycle <= 100:
        return 100
    if max_cycle <= 200:
        return 200
    if max_cycle <= 300:
        return 300
    if max_cycle <= 500:
        return 500
    if max_cycle <= 1000:
        return 1000

    return int(np.ceil(max_cycle / 500) * 500)


def palette_to_hex_colors(palette_name: str, n: int) -> list[str]:
    """
    Create default hex colors from a Matplotlib qualitative palette.
    """
    if palette_name == "Set2":
        colors = list(plt.cm.Set2.colors)
    elif palette_name == "Dark2":
        colors = list(plt.cm.Dark2.colors)
    elif palette_name == "tab10":
        colors = list(plt.cm.tab10.colors)
    elif palette_name == "tab20":
        colors = list(plt.cm.tab20.colors)
    elif palette_name == "tab20 + tab20b":
        colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    else:
        colors = list(plt.cm.Set2.colors) + list(plt.cm.Dark2.colors) + list(plt.cm.tab20.colors)

    hex_colors = []

    for i in range(n):
        rgb = colors[i % len(colors)]
        rgb255 = tuple(int(round(c * 255)) for c in rgb[:3])
        hex_colors.append("#{:02x}{:02x}{:02x}".format(*rgb255))

    return hex_colors


def hex_to_rgb_tuple(hex_color: str) -> tuple[float, float, float]:
    """
    Convert #RRGGBB to Matplotlib RGB tuple.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


COMMON_PLOT_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
}


def apply_common_plot_style() -> None:
    """Reset Matplotlib defaults before every rendered figure."""
    plt.rcdefaults()
    plt.rcParams.update(COMMON_PLOT_RCPARAMS)


def keep_twin_axes_points_visible(ax1, ax2) -> None:
    """Keep twin-axis backgrounds from covering scatter points."""
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)


def clean_capacity_plot_df(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric rows that can actually be plotted."""
    numeric_plot_df = plot_df.copy()
    for col in ["cycle_index", "capacity_retention_percent", "coulombic_efficiency_percent"]:
        numeric_plot_df[col] = pd.to_numeric(numeric_plot_df[col], errors="coerce")
    return numeric_plot_df.dropna(
        subset=["cycle_index", "capacity_retention_percent", "coulombic_efficiency_percent"]
    )


def padded_limits(values: pd.Series, fallback_min: float, fallback_max: float, min_pad: float) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return float(fallback_min), float(fallback_max)
    lo = float(vals.min())
    hi = float(vals.max())
    pad = max(float(min_pad), 0.08 * max(1.0, hi - lo))
    return lo - pad, hi + pad


def capacity_figure_limits(plot_df: pd.DataFrame, style: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, bool]:
    """Use style limits unless they would hide all points in the final figure."""
    numeric_plot_df = clean_capacity_plot_df(plot_df)
    if numeric_plot_df.empty:
        return {
            "auto_x_range": bool(style["auto_x_range"]),
            "x_min": float(style["x_min"]),
            "x_max": float(style["x_max"]),
            "cap_y_min": float(style["cap_y_min"]),
            "cap_y_max": float(style["cap_y_max"]),
            "ce_y_min": float(style["ce_y_min"]),
            "ce_y_max": float(style["ce_y_max"]),
        }, numeric_plot_df, False

    adjusted = False
    auto_x_range = bool(style["auto_x_range"])
    if auto_x_range:
        x_min = 0.0
        x_max = float(capacity_auto_x_limit(float(numeric_plot_df["cycle_index"].max())))
    else:
        x_min = float(style["x_min"])
        x_max = float(style["x_max"])

    cap_y_min = float(style["cap_y_min"])
    cap_y_max = float(style["cap_y_max"])
    ce_y_min = float(style["ce_y_min"])
    ce_y_max = float(style["ce_y_max"])

    x_visible = numeric_plot_df["cycle_index"].between(x_min, x_max)
    if not x_visible.any():
        auto_x_range = False
        x_min = 0.0
        x_max = float(capacity_auto_x_limit(float(numeric_plot_df["cycle_index"].max())))
        x_visible = numeric_plot_df["cycle_index"].between(x_min, x_max)
        adjusted = True

    if not (x_visible & numeric_plot_df["capacity_retention_percent"].between(cap_y_min, cap_y_max)).any():
        cap_y_min, cap_y_max = padded_limits(numeric_plot_df.loc[x_visible, "capacity_retention_percent"], 75.0, 110.0, 2.0)
        adjusted = True

    if not (x_visible & numeric_plot_df["coulombic_efficiency_percent"].between(ce_y_min, ce_y_max)).any():
        ce_y_min, ce_y_max = padded_limits(numeric_plot_df.loc[x_visible, "coulombic_efficiency_percent"], 90.0, 100.5, 0.2)
        adjusted = True

    return {
        "auto_x_range": auto_x_range,
        "x_min": x_min,
        "x_max": x_max,
        "cap_y_min": cap_y_min,
        "cap_y_max": cap_y_max,
        "ce_y_min": ce_y_min,
        "ce_y_max": ce_y_max,
    }, numeric_plot_df, adjusted



def make_capacity_placeholder_plot_data(
    sample_name: str,
    n_files: int = 3,
    max_cycle: int = 500,
) -> pd.DataFrame:
    """
    Create lightweight placeholder cycling data for style preview.

    This avoids repeatedly reading Excel files while the user is only adjusting
    labels, legend placement, axis ranges, figure size, or colors.
    """
    cycles = np.arange(0, int(max_cycle) + 1)
    rows = []

    # Long file names are intentional: they make legend spacing problems visible
    # even before real Excel data are loaded.
    for i in range(1, int(n_files) + 1):
        decay = 0.018 + 0.006 * i
        ripple = 0.9 * np.sin(cycles / 34.0 + i)
        retention = 100.0 - decay * cycles + ripple
        efficiency = 99.35 + 0.35 * (1 - np.exp(-cycles / 55.0)) + 0.05 * np.sin(cycles / 17.0 + i)

        source_file = (
            f"{sample_name}_placeholder_long_filename_cell_{i:02d}_"
            f"capacity_retention_and_CE_preview.xlsx"
        )

        rows.append(
            pd.DataFrame(
                {
                    "sample": sample_name,
                    "source_file": source_file,
                    "relative_path": source_file,
                    "cycle_index": cycles,
                    "discharge_capacity_mAh": retention,
                    "capacity_retention_percent": retention,
                    "coulombic_efficiency_percent": efficiency,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def make_capacity_figure(
    plot_df: pd.DataFrame,
    sample_name: str,
    color_hex: str,
    plot_title: str,
    x_label: str,
    cap_y_label: str,
    ce_y_label: str,
    legend_title: str,
    show_legend: bool,
    auto_x_range: bool,
    x_min: float,
    x_max: float,
    cap_y_min: float,
    cap_y_max: float,
    ce_y_min: float,
    ce_y_max: float,
    marker_size: int,
    fig_width: float,
    fig_height: float,
    legend_position: str = "Top",
    legend_label_max_len: int = 24,
    legend_columns: int = 3,
):
    """
    Make one capacity-retention / coulombic-efficiency plot for one sample.

    Layout notes:
    - Top legend uses a figure-level legend and reserves a dedicated top band.
    - Right legend uses a figure-level legend and reserves a dedicated right band.
    - This avoids overlap with the title and the right y-axis.
    """
    apply_common_plot_style()

    # Make plotting robust across files whose Excel columns may have been read
    # as object/string dtype. If these values are not coerced here, Matplotlib
    # can silently draw an empty-looking figure in some Streamlit reruns.
    plot_df = plot_df.copy()
    for numeric_col in [
        "cycle_index",
        "capacity_retention_percent",
        "coulombic_efficiency_percent",
    ]:
        plot_df[numeric_col] = pd.to_numeric(plot_df[numeric_col], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["cycle_index", "capacity_retention_percent", "coulombic_efficiency_percent"]
    )

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax2 = ax1.twinx()
    keep_twin_axes_points_visible(ax1, ax2)

    color_rgb = hex_to_rgb_tuple(color_hex)

    group_key = "relative_path" if "relative_path" in plot_df.columns else "source_file"

    if plot_df.empty:
        ax1.text(
            0.5,
            0.5,
            "No valid numeric plotting data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )

    for group_id, group in plot_df.groupby(group_key, sort=True):
        group = group.sort_values("cycle_index")
        if "source_file" in group.columns and len(group):
            file_label_raw = Path(str(group["source_file"].iloc[0])).stem
        else:
            file_label_raw = Path(str(group_id)).stem
        if "repeat" in group.columns and len(group):
            repeat_label = str(group["repeat"].iloc[0])
            full_label = repeat_label if repeat_label == file_label_raw else f"{repeat_label} | {file_label_raw}"
        else:
            full_label = file_label_raw
        file_label = shorten_label(full_label, legend_label_max_len)

        ax1.scatter(
            group["cycle_index"].to_numpy(float),
            group["capacity_retention_percent"].to_numpy(float),
            color=color_rgb,
            marker="o",
            s=marker_size,
            alpha=1,
            zorder=3,
            label=file_label,
        )

        ax2.scatter(
            group["cycle_index"].to_numpy(float),
            group["coulombic_efficiency_percent"].to_numpy(float),
            facecolors="none",
            edgecolors=color_rgb,
            marker="o",
            s=marker_size,
            linewidths=1.5,
            alpha=1,
            zorder=3,
        )

    if auto_x_range:
        x_min_final = 0
        if plot_df.empty:
            x_max_final = 100
        else:
            x_max_final = capacity_auto_x_limit(float(plot_df["cycle_index"].max()))
    else:
        x_min_final = float(x_min)
        x_max_final = float(x_max)

    ax1.set_xlim(x_min_final, x_max_final)
    ax1.set_ylim(float(cap_y_min), float(cap_y_max))
    ax2.set_ylim(float(ce_y_min), float(ce_y_max))

    title = plot_title.replace("{sample}", sample_name)
    if title.strip():
        ax1.set_title(title, fontsize=18, pad=14)

    ax1.set_xlabel(x_label, fontsize=18, labelpad=8)
    ax1.set_ylabel(cap_y_label, fontsize=18, labelpad=8)
    ax2.set_ylabel(ce_y_label, fontsize=18, labelpad=12)

    for ax in [ax1, ax2]:
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=15,
            length=6,
            width=1.5,
            pad=6,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            length=4,
            width=1,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    legend_position = legend_position if show_legend else "Hide"

    handles, labels = ax1.get_legend_handles_labels()
    n_labels = max(1, len(labels))
    ncol = max(1, min(int(legend_columns), n_labels))

    # Avoid tight_layout here. It often cannot correctly reserve space for a
    # figure-level legend together with a twinx right y-axis.
    if legend_position == "Top":
        top = 0.74 if title.strip() else 0.80
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=top)

        # Put title inside the axes area and legend above it. This creates a
        # predictable separation between the two.
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=ncol,
            title=legend_title,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )

    elif legend_position == "Right":
        # Reserve a wider right band so the right y-axis label/ticks and legend
        # do not compete for the same physical space.
        fig.subplots_adjust(left=0.11, right=0.68, bottom=0.14, top=0.90)

        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(0.985, 0.53),
            ncol=1,
            title=legend_title,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    elif legend_position == "Inside":
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.88)
        ax1.legend(
            handles,
            labels,
            loc="best",
            title=legend_title,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
        )

    else:
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.90)

    return fig


def make_capacity_sample_comparison_figure(
    plot_df: pd.DataFrame,
    repeat_name: str,
    sample_colors: dict[str, str],
    plot_title: str,
    x_label: str,
    cap_y_label: str,
    ce_y_label: str,
    legend_title: str,
    show_legend: bool,
    auto_x_range: bool,
    x_min: float,
    x_max: float,
    cap_y_min: float,
    cap_y_max: float,
    ce_y_min: float,
    ce_y_max: float,
    marker_size: int,
    fig_width: float,
    fig_height: float,
    legend_position: str = "Top",
    legend_label_max_len: int = 24,
    legend_columns: int = 3,
):
    """Make one cycling comparison figure for the same repeat across samples."""
    apply_common_plot_style()

    plot_df = plot_df.copy()
    for numeric_col in [
        "cycle_index",
        "capacity_retention_percent",
        "coulombic_efficiency_percent",
    ]:
        plot_df[numeric_col] = pd.to_numeric(plot_df[numeric_col], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["cycle_index", "capacity_retention_percent", "coulombic_efficiency_percent"]
    )

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax2 = ax1.twinx()
    keep_twin_axes_points_visible(ax1, ax2)

    if plot_df.empty:
        ax1.text(
            0.5,
            0.5,
            "No valid numeric plotting data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
        )

    handles_seen: set[str] = set()
    for sample, sample_df in plot_df.groupby("sample", sort=True):
        color_rgb = hex_to_rgb_tuple(sample_colors.get(str(sample), "#4E79A7"))
        label = shorten_label(str(sample), legend_label_max_len)
        for _group_id, group in sample_df.groupby("relative_path", sort=True):
            group = group.sort_values("cycle_index")
            legend_label = label if str(sample) not in handles_seen else "_nolegend_"
            handles_seen.add(str(sample))
            ax1.scatter(
                group["cycle_index"].to_numpy(float),
                group["capacity_retention_percent"].to_numpy(float),
                color=color_rgb,
                marker="o",
                s=marker_size,
                alpha=1,
                zorder=3,
                label=legend_label,
            )
            ax2.scatter(
                group["cycle_index"].to_numpy(float),
                group["coulombic_efficiency_percent"].to_numpy(float),
                facecolors="none",
                edgecolors=color_rgb,
                marker="o",
                s=marker_size,
                linewidths=1.5,
                alpha=1,
                zorder=3,
            )

    if auto_x_range:
        x_min_final = 0
        x_max_final = 100 if plot_df.empty else capacity_auto_x_limit(float(plot_df["cycle_index"].max()))
    else:
        x_min_final = float(x_min)
        x_max_final = float(x_max)

    ax1.set_xlim(x_min_final, x_max_final)
    ax1.set_ylim(float(cap_y_min), float(cap_y_max))
    ax2.set_ylim(float(ce_y_min), float(ce_y_max))

    title = plot_title.replace("{repeat}", repeat_name).replace("{sample}", repeat_name)
    if title.strip():
        ax1.set_title(title, fontsize=18, pad=14)

    ax1.set_xlabel(x_label, fontsize=18, labelpad=8)
    ax1.set_ylabel(cap_y_label, fontsize=18, labelpad=8)
    ax2.set_ylabel(ce_y_label, fontsize=18, labelpad=12)

    for ax in [ax1, ax2]:
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=15,
            length=6,
            width=1.5,
            pad=6,
        )
        ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    legend_position = legend_position if show_legend else "Hide"
    handles, labels = ax1.get_legend_handles_labels()
    n_labels = max(1, len(labels))
    ncol = max(1, min(int(legend_columns), n_labels))

    if legend_position == "Top":
        top = 0.74 if title.strip() else 0.80
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=top)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=ncol,
            title=legend_title,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
    elif legend_position == "Right":
        fig.subplots_adjust(left=0.11, right=0.68, bottom=0.14, top=0.90)
        fig.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(0.985, 0.53),
            ncol=1,
            title=legend_title,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
    elif legend_position == "Inside":
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.88)
        ax1.legend(handles, labels, loc="best", title=legend_title, fontsize=10, title_fontsize=12, frameon=False)
    else:
        fig.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.90)

    return fig


def render_capacity_output_figure(item: dict[str, object]):
    if item.get("plot_kind") == "sample_comparison":
        return make_capacity_sample_comparison_figure(**item["figure_kwargs"])
    return make_capacity_figure(**item["figure_kwargs"])




def make_single_file_capacity_preview_figure(
    file_df: pd.DataFrame,
    title: str,
    color_hex: str,
    axis_mode: str = "Fixed common range",
    fixed_cap_y_min: float = 75.0,
    fixed_cap_y_max: float = 110.0,
    fixed_ce_y_min: float = 90.0,
    fixed_ce_y_max: float = 100.5,
    fig_width: float = 5.2,
    fig_height: float = 3.25,
):
    """
    Make a compact file-level cycling preview figure.

    This figure is meant for quality control. It deliberately uses a fixed,
    compact layout so that a long list of file previews looks aligned in
    Streamlit.
    """
    apply_common_plot_style()

    df = file_df.sort_values("cycle_index").copy()
    color_rgb = hex_to_rgb_tuple(color_hex)

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax2 = ax1.twinx()
    keep_twin_axes_points_visible(ax1, ax2)

    # Keep the file-level preview visually consistent with the final
    # sample-level figure: filled circles for capacity retention and open
    # circles for coulombic efficiency. No connecting lines are drawn.
    ax1.scatter(
        df["cycle_index"],
        df["capacity_retention_percent"],
        color=color_rgb,
        marker="o",
        s=18,
        alpha=1,
        zorder=3,
    )

    ax2.scatter(
        df["cycle_index"],
        df["coulombic_efficiency_percent"],
        facecolors="none",
        edgecolors=color_rgb,
        marker="o",
        s=18,
        linewidths=1.0,
        alpha=1,
        zorder=3,
    )

    max_cycle = float(df["cycle_index"].max()) if len(df) else 100.0
    ax1.set_xlim(0, capacity_auto_x_limit(max_cycle))

    if axis_mode == "Auto per file":
        cap = df["capacity_retention_percent"].to_numpy(float)
        ce = df["coulombic_efficiency_percent"].to_numpy(float)

        cap_min = np.nanmin(cap) if len(cap) else fixed_cap_y_min
        cap_max = np.nanmax(cap) if len(cap) else fixed_cap_y_max
        ce_min = np.nanmin(ce) if len(ce) else fixed_ce_y_min
        ce_max = np.nanmax(ce) if len(ce) else fixed_ce_y_max

        cap_pad = max(1.5, 0.08 * max(1.0, cap_max - cap_min))
        ce_pad = max(0.15, 0.12 * max(0.1, ce_max - ce_min))

        ax1.set_ylim(
            np.floor((cap_min - cap_pad) / 5) * 5,
            np.ceil((cap_max + cap_pad) / 5) * 5,
        )
        ax2.set_ylim(
            np.floor((ce_min - ce_pad) * 2) / 2,
            np.ceil((ce_max + ce_pad) * 2) / 2,
        )
    else:
        ax1.set_ylim(float(fixed_cap_y_min), float(fixed_cap_y_max))
        ax2.set_ylim(float(fixed_ce_y_min), float(fixed_ce_y_max))

    if str(title).strip():
        ax1.set_title(shorten_label(title, 54), fontsize=10, pad=7)
    ax1.set_xlabel("Cycle Index", fontsize=9, labelpad=4)
    ax1.set_ylabel("Retention (%)", fontsize=9, labelpad=4)
    ax2.set_ylabel("CE (%)", fontsize=9, labelpad=6)

    for ax in [ax1, ax2]:
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=8,
            length=4,
            width=1.0,
            pad=3,
        )
        ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    top_margin = 0.84 if str(title).strip() else 0.93
    fig.subplots_adjust(left=0.13, right=0.84, bottom=0.18, top=top_margin)
    return fig


def make_empty_single_file_capacity_preview_figure(
    message: str = "No valid preview",
    fig_width: float = 3.7,
    fig_height: float = 2.25,
):
    """Create an empty preview figure with the same footprint as valid file previews."""
    apply_common_plot_style()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=9,
        color="0.55",
        transform=ax.transAxes,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def render_file_preview_card(
    record: dict[str, object],
    file_df: pd.DataFrame | None,
    raw_df: pd.DataFrame | None,
    error: str | None,
    checkbox_key: str,
    color_hex: str,
    preview_axis_mode: str,
    cap_y_min: float = 75.0,
    cap_y_max: float = 110.0,
    ce_y_min: float = 90.0,
    ce_y_max: float = 100.5,
) -> None:
    """
    Render a compact, equal-height file-level preview card.

    The card intentionally keeps every file in the same visual footprint so the
    four-column preview grid stays aligned even when a file is invalid.
    """
    rel = str(record["relative_path"])
    file_stem = Path(rel).stem
    metrics = compute_capacity_selection_metrics(raw_df, record)

    ice_text = preview_metric_text(metrics.get("ICE (%)"), "%")
    life_text = preview_metric_text(metrics.get("Cycle Life"), "", digits=4)
    ace_text = preview_metric_text(metrics.get("ACE (%)"), "%")
    ace_cycle_text = preview_metric_text(metrics.get("ACE cycle"), "", digits=4)
    note_text = str(metrics.get("Note") or (error or ""))
    note_text = note_text.strip()

    try:
        card_ctx = st.container(border=True)
    except TypeError:
        card_ctx = st.container()

    with card_ctx:
        st.checkbox(
            shorten_label(file_stem, 30),
            key=checkbox_key,
            help=rel,
        )
        sync_cycling_checkbox_to_saved(str(record["sample"]), rel, checkbox_key)

        if file_df is not None and not file_df.empty:
            fig = make_single_file_capacity_preview_figure(
                file_df=file_df,
                title="",
                color_hex=color_hex,
                axis_mode=preview_axis_mode,
                fixed_cap_y_min=cap_y_min,
                fixed_cap_y_max=cap_y_max,
                fixed_ce_y_min=ce_y_min,
                fixed_ce_y_max=ce_y_max,
                fig_width=3.7,
                fig_height=2.25,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            # Use a Matplotlib placeholder with the same figsize as valid previews.
            # This keeps the 4-column grid aligned better than an HTML-only box.
            empty_fig = make_empty_single_file_capacity_preview_figure(
                message="No valid preview",
                fig_width=3.7,
                fig_height=2.25,
            )
            st.pyplot(empty_fig, clear_figure=True)
            plt.close(empty_fig)

        render_preview_metric_grid(
            [
                ("ICE", ice_text),
                ("Life", life_text),
                ("ACE", ace_text),
                ("ACE cyc.", ace_cycle_text),
            ]
        )
        render_preview_note(note_text)


def safe_extract_zip_to_dir(uploaded_zip, extract_dir: Path) -> None:
    """
    Safely extract an uploaded ZIP file into extract_dir.

    This mode is intended for small demo datasets. It is not recommended for
    large private research datasets because browser upload and cloud memory
    limits become the bottleneck long before the analysis logic does.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue()), "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP path detected: {member.filename}")
            target = (extract_dir / member_path).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Unsafe ZIP path detected: {member.filename}")
        zf.extractall(extract_dir)


def infer_cycling_root_after_unzip(extract_dir: Path) -> Path:
    """
    Infer the actual cycling root after unzipping.

    Handles both structures:
        zip_root/sample_1/*.xlsx
        zip_root/outer_folder/sample_1/*.xlsx
    """
    visible_children = [p for p in extract_dir.iterdir() if not p.name.startswith("__MACOSX")]
    dirs = [p for p in visible_children if p.is_dir() and not p.name.startswith(".")]
    files = [p for p in visible_children if p.is_file() and not p.name.startswith(".")]

    if len(dirs) == 1 and not files:
        inner = dirs[0]
        inner_dirs = [p for p in inner.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if inner_dirs:
            return inner

    return extract_dir


def get_or_create_uploaded_zip_root(uploaded_zip) -> Path:
    """
    Extract an uploaded demo ZIP once per session and return the inferred root.
    """
    upload_sig = hashlib.sha1(uploaded_zip.getvalue()).hexdigest()[:16]
    state_key = "cycling_uploaded_zip_state"
    current = st.session_state.get(state_key)

    if current and current.get("signature") == upload_sig:
        root = Path(current["root_dir"])
        if root.exists():
            return root

    extract_dir = Path(tempfile.mkdtemp(prefix="battery_cycling_zip_"))
    safe_extract_zip_to_dir(uploaded_zip, extract_dir)
    root_dir = infer_cycling_root_after_unzip(extract_dir)

    st.session_state[state_key] = {
        "signature": upload_sig,
        "extract_dir": str(extract_dir),
        "root_dir": str(root_dir),
    }
    return root_dir


def get_or_create_uploaded_stripping_zip_root(uploaded_zip) -> Path:
    """
    Extract an uploaded stripping ZIP once per session and return the inferred root.
    """
    upload_sig = hashlib.sha1(uploaded_zip.getvalue()).hexdigest()[:16]
    state_key = "stripping_uploaded_zip_state"
    current = st.session_state.get(state_key)

    if current and current.get("signature") == upload_sig:
        root = Path(current["root_dir"])
        if root.exists():
            return root

    extract_dir = Path(tempfile.mkdtemp(prefix="battery_stripping_zip_"))
    safe_extract_zip_to_dir(uploaded_zip, extract_dir)
    root_dir = infer_cycling_root_after_unzip(extract_dir)

    st.session_state[state_key] = {
        "signature": upload_sig,
        "extract_dir": str(extract_dir),
        "root_dir": str(root_dir),
    }
    return root_dir


def resolve_cycling_input_root_from_sidebar() -> tuple[Path | None, Path | None, str]:
    """
    Resolve the cycling root directory and output directory.

    For very large datasets, use Local/server folder path. The Streamlit process
    must run on the same machine/server where the data directory exists, or on a
    server where the data is mounted as a filesystem path.
    """
    st.header("Cycling input")

    input_mode = st.radio(
        "Data access mode",
        ["Local/server folder path", "Demo ZIP upload only"],
        index=0,
        help=(
            "Use Local/server folder path for real 20–50 GB datasets. "
            "ZIP upload is only for small demo datasets and is not suitable for large private data."
        ),
    )

    if input_mode == "Local/server folder path":
        st.info(
            "For large datasets, do not upload files through the browser. Run Streamlit on the machine/server that can already access the data, then point this field to that folder."
        )

        configured_base = os.environ.get("BATTERY_DATA_ROOT", "").strip()
        use_configured_base = False

        if configured_base:
            base_path = Path(configured_base).expanduser().resolve()
            st.caption(f"Configured server data root: `{base_path}`")
            use_configured_base = st.checkbox(
                "Use configured BATTERY_DATA_ROOT",
                value=True,
                help="Set BATTERY_DATA_ROOT on the server to restrict users to a known data folder.",
            )
        else:
            base_path = None

        if configured_base and use_configured_base and base_path is not None:
            relative_data_dir = st.text_input(
                "Dataset folder relative to BATTERY_DATA_ROOT",
                value="",
                help="Example: Li Vendor/data/test. Leave empty to use BATTERY_DATA_ROOT itself.",
            )
            root_dir = (base_path / relative_data_dir).expanduser().resolve()
        else:
            root_dir_str = st.text_input(
                "Root data directory on this machine/server",
                value="",
                help=(
                    "Folder containing sample subfolders. This path must exist on the machine/server running Streamlit, "
                    "not necessarily on the browser user's computer."
                ),
            )
            if not root_dir_str.strip():
                return None, None, "Enter a root data directory to start."
            root_dir = Path(root_dir_str).expanduser().resolve()

        output_dir_str = st.text_input(
            "Output directory",
            value="",
            help="Leave empty to save to <root_dir>/capacity_batch_results.",
        )
        output_dir = Path(output_dir_str).expanduser().resolve() if output_dir_str.strip() else root_dir / "capacity_batch_results"
        return root_dir, output_dir, input_mode

    st.warning(
        "ZIP upload is for small demo data only. For 20–50 GB raw datasets, use Local/server folder path on a machine/server where the data is already mounted."
    )
    uploaded_zip = st.file_uploader(
        "Upload a small demo ZIP containing sample folders",
        type=["zip"],
        accept_multiple_files=False,
        help="Do not use this for 20–50 GB datasets.",
    )
    if uploaded_zip is None:
        return None, None, "Upload a small demo ZIP to start, or switch to Local/server folder path for real data."

    try:
        root_dir = get_or_create_uploaded_zip_root(uploaded_zip)
    except Exception as exc:
        st.error(f"Could not extract ZIP: {exc}")
        return None, None, "ZIP extraction failed."

    output_dir = root_dir / "capacity_batch_results"
    st.caption(f"Temporary extracted root: `{root_dir}`")
    return root_dir, output_dir, input_mode

def render_cycling_analysis_page() -> None:
    st.title("Cycling Analysis")
    st.caption("Batch capacity retention and coulombic efficiency plotting.")

    st.markdown(
        """
        This tool treats each direct subfolder under the root directory as one sample.

        Expected structure:

        ```text
        root_directory/
            sample_1/
                file_1.xlsx
                file_2.xlsx
            sample_2/
                file_3.xlsx
        ```
        """
    )

    with st.sidebar:
        root_dir, output_dir, input_mode = resolve_cycling_input_root_from_sidebar()
        if not bool(st.session_state.get("cycling_bulk_preview_default_migrated", False)):
            st.session_state["cycling_bulk_preview"] = True
            st.session_state["cycling_bulk_preview_default_migrated"] = True
        cycling_bulk_preview = st.checkbox(
            "Load all selected samples at once in data preview",
            value=True,
            key="cycling_bulk_preview",
            help="Read all selected samples in one pass, then review and select files without stepping sample-by-sample.",
        )
        cycling_parallel_load = st.checkbox(
            "Parallel file loading (experimental)",
            value=False,
            key="cycling_parallel_load",
            disabled=not cycling_bulk_preview,
            help=(
                "Experimental. Reads multiple Excel files at the same time during load-all preview. "
                "This can use much more CPU, memory, and disk/network bandwidth, and may be less stable with very large files, cloud drives, or openpyxl."
            ),
        )
        cycling_parallel_backend = st.selectbox(
            "Parallel backend",
            ["Threads", "Processes"],
            key="cycling_parallel_backend",
            disabled=not cycling_parallel_load,
            help=(
                "Threads are lighter but Python Excel parsing may not fully use all CPU cores. "
                "Processes use true multi-core parallelism, but use more memory and can stress disk/cloud storage."
            ),
        )
        cycling_parallel_workers = int(
            st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=64,
                value=12,
                step=1,
                key="cycling_parallel_workers",
                disabled=not cycling_parallel_load,
                help="Maximum Excel files to parse at the same time when parallel loading is enabled. Higher values can increase CPU, RAM, and disk/cloud-drive pressure.",
            )
        )
        cycling_use_parsed_cache = st.checkbox(
            "Cache parsed Excel data",
            value=True,
            key="cycling_use_parsed_excel_cache",
            help="Store parsed per-file data as parquet when available, otherwise CSV. Preview, style preview, and final output reuse this cache until the Excel file or read settings change.",
        )
        cycling_notify_load_complete = st.checkbox(
            "Notify when load-all preview finishes",
            value=True,
            key="cycling_notify_load_complete",
            disabled=not cycling_bulk_preview,
            help="Shows a Streamlit toast and attempts a browser notification after the load-all data preview finishes. Browser notifications may require permission.",
        )

        st.header("Data settings")

        sheet_name = st.text_input("Excel sheet name", value="cycle")

        capacity_col = st.text_input(
            "Discharge capacity column",
            value="DChg. Cap.(mAh)",
        )

        efficiency_col = st.text_input(
            "Coulombic efficiency column",
            value="Chg.-DChg. Eff(%)",
        )

        skip_initial_rows = st.number_input(
            "Rows to skip at beginning",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
        )

        use_retention_filter = st.checkbox(
            "Filter by minimum capacity retention",
            value=True,
        )

        min_capacity_retention = st.number_input(
            "Minimum capacity retention (%)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            step=1.0,
            disabled=not use_retention_filter,
        )

    if root_dir is None or output_dir is None:
        st.info(input_mode)
        return

    cycling_persistent_cache_dir = (
        parsed_excel_cache_dir(output_dir, "cycling")
        if bool(cycling_use_parsed_cache)
        else None
    )

    if not root_dir.exists():
        st.error(
            f"Root directory does not exist on the Streamlit runtime machine/server: `{root_dir}`\n\n"
            "If you are using a deployed web app, a path like `/Users/...` refers to the remote server, not your Mac. "
            "For 20–50 GB datasets, run this app locally or on a server where the data folder is mounted."
        )
        return

    if not root_dir.is_dir():
        st.error(f"Root path is not a directory: `{root_dir}`")
        return

    sample_folders = find_capacity_sample_folders(root_dir, output_dir)

    if not sample_folders:
        st.warning("No sample folders found under the root directory.")
        return

    sample_names = [folder.name for folder in sample_folders]
    folder_map = {folder.name: folder for folder in sample_folders}
    default_colors = palette_to_hex_colors("Set2 + Dark2 + tab20", len(sample_names))
    default_color_map = {sample: default_colors[i] for i, sample in enumerate(sample_names)}

    min_retention = float(min_capacity_retention) if use_retention_filter else None
    top_n_value = None

    st.subheader("Cycling workflow")
    workflow_options = [
        "1. Data preview & file selection",
        "2. Style preview",
        "3. Final output",
    ]
    if st.session_state.get("cycling_workflow_step") not in workflow_options:
        st.session_state["cycling_workflow_step"] = workflow_options[0]

    workflow_view = st.radio(
        "Choose workflow step",
        workflow_options,
        horizontal=True,
        key="cycling_workflow_step",
        help="Review files sample-by-sample, tune figure style, then generate final outputs.",
    )

    selected_samples = st.multiselect(
        "Samples to process",
        options=sample_names,
        default=sample_names,
        help="Each selected sample will produce one final figure.",
    )

    if not selected_samples:
        st.warning("Select at least one sample.")
        return

    # Manual file selection is always enabled for the cycling workflow.
    # The UI no longer exposes this as an option because final plots should
    # always respect the per-file include/exclude choices made in preview.
    manual_selection = True

    ensure_cycling_selection_store()

    if st.session_state.get("cycling_inspect_sample") not in selected_samples:
        st.session_state["cycling_inspect_sample"] = selected_samples[0]

    # Initialize checkbox values once. If a saved selection exists for a sample,
    # use it as the initial value; otherwise include every file until we learn a
    # file is unreadable/invalid, in which case the preview step changes it to False.
    for sample in selected_samples:
        saved_for_sample = st.session_state["cycling_saved_selection"].get(sample, {})
        for record in capacity_file_records(sample, folder_map[sample], root_dir):
            rel = str(record["relative_path"])
            key = cycling_file_include_key(sample, rel)
            if key not in st.session_state:
                st.session_state[key] = bool(saved_for_sample.get(rel, True))

    # Defaults for plot styling. The widgets in Style preview write to these keys;
    # Final output reads from the same keys, so output can be a separate step.
    style_defaults = {
        "cycling_plot_title": "{sample}",
        "cycling_x_label": "Cycle Index",
        "cycling_cap_y_label": "Capacity Retention (%)",
        "cycling_ce_y_label": "Coulombic Efficiency (%)",
        "cycling_show_legend": True,
        "cycling_legend_position": "Top",
        "cycling_legend_title": "Files",
        "cycling_legend_label_max_len": 24,
        "cycling_legend_columns": 3,
        "cycling_auto_x_range": True,
        "cycling_x_min": 0.0,
        "cycling_x_max": 500.0,
        "cycling_cap_y_min": 75.0,
        "cycling_cap_y_max": 110.0,
        "cycling_ce_y_min": 90.0,
        "cycling_ce_y_max": 100.5,
        "cycling_palette_name": "Set2 + Dark2 + tab20",
        "cycling_plot_mode": CYCLING_PLOT_MODE_SINGLE,
        "cycling_compare_repeat": "",
        "cycling_compare_samples": selected_samples,
        "cycling_marker_size": 80,
        "cycling_fig_width": 9.5,
        "cycling_fig_height": 5.8,
        "cycling_dpi": 300,
    }
    for key, value in style_defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state["cycling_plot_mode"] = normalize_cycling_plot_mode(
        st.session_state.get("cycling_plot_mode", CYCLING_PLOT_MODE_SINGLE)
    )
    if not isinstance(st.session_state.get("cycling_compare_samples"), list):
        st.session_state["cycling_compare_samples"] = selected_samples
    st.session_state["cycling_compare_samples"] = [
        sample for sample in st.session_state.get("cycling_compare_samples", selected_samples)
        if sample in selected_samples
    ] or list(selected_samples)

    # Older versions of this app could leave these text fields as empty strings
    # in session_state. Refill only blank text labels so the Text tab always
    # opens with useful defaults.
    text_style_defaults = {
        "cycling_plot_title": "{sample}",
        "cycling_x_label": "Cycle Index",
        "cycling_cap_y_label": "Capacity Retention (%)",
        "cycling_ce_y_label": "Coulombic Efficiency (%)",
        "cycling_legend_title": "Files",
    }
    for key, value in text_style_defaults.items():
        if not str(st.session_state.get(key, "")).strip():
            st.session_state[key] = value

    # Guard against stale or invalid axis values from older sessions.
    if float(st.session_state.get("cycling_cap_y_max", 110.0)) <= float(st.session_state.get("cycling_cap_y_min", 75.0)):
        st.session_state["cycling_cap_y_min"] = 75.0
        st.session_state["cycling_cap_y_max"] = 110.0
    if float(st.session_state.get("cycling_ce_y_max", 100.5)) <= float(st.session_state.get("cycling_ce_y_min", 90.0)):
        st.session_state["cycling_ce_y_min"] = 90.0
        st.session_state["cycling_ce_y_max"] = 100.5
    if float(st.session_state.get("cycling_x_max", 500.0)) <= float(st.session_state.get("cycling_x_min", 0.0)):
        st.session_state["cycling_x_min"] = 0.0
        st.session_state["cycling_x_max"] = 500.0

    def current_style_values() -> dict[str, object]:
        return {
            "plot_title": st.session_state.get("cycling_plot_title", "{sample}"),
            "x_label": st.session_state.get("cycling_x_label", "Cycle Index"),
            "cap_y_label": st.session_state.get("cycling_cap_y_label", "Capacity Retention (%)"),
            "ce_y_label": st.session_state.get("cycling_ce_y_label", "Coulombic Efficiency (%)"),
            "show_legend": bool(st.session_state.get("cycling_show_legend", True)),
            "legend_position": st.session_state.get("cycling_legend_position", "Top"),
            "legend_title": st.session_state.get("cycling_legend_title", "Files"),
            "legend_label_max_len": int(st.session_state.get("cycling_legend_label_max_len", 24)),
            "legend_columns": int(st.session_state.get("cycling_legend_columns", 3)),
            "auto_x_range": bool(st.session_state.get("cycling_auto_x_range", True)),
            "x_min": float(st.session_state.get("cycling_x_min", 0.0)),
            "x_max": float(st.session_state.get("cycling_x_max", 500.0)),
            "cap_y_min": float(st.session_state.get("cycling_cap_y_min", 75.0)),
            "cap_y_max": float(st.session_state.get("cycling_cap_y_max", 110.0)),
            "ce_y_min": float(st.session_state.get("cycling_ce_y_min", 90.0)),
            "ce_y_max": float(st.session_state.get("cycling_ce_y_max", 100.5)),
            "palette_name": st.session_state.get("cycling_palette_name", "Set2 + Dark2 + tab20"),
            "plot_mode": normalize_cycling_plot_mode(st.session_state.get("cycling_plot_mode", CYCLING_PLOT_MODE_SINGLE)),
            "compare_repeat": st.session_state.get("cycling_compare_repeat", ""),
            "compare_samples": list(st.session_state.get("cycling_compare_samples", selected_samples)),
            "marker_size": int(st.session_state.get("cycling_marker_size", 80)),
            "fig_width": float(st.session_state.get("cycling_fig_width", 9.5)),
            "fig_height": float(st.session_state.get("cycling_fig_height", 5.8)),
            "dpi": int(st.session_state.get("cycling_dpi", 300)),
        }

    def current_sample_colors(style: dict[str, object]) -> dict[str, str]:
        colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
        palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}
        return {
            sample: st.session_state.get(
                f"cycling_color_{safe_filename(sample)}",
                palette_color_map[sample],
            )
            for sample in selected_samples
        }

    def selected_paths_for_output(sample: str) -> list[str] | None:
        return selected_relative_paths_for_sample_saved_first(
            sample,
            folder_map[sample],
            root_dir,
            manual_selection=manual_selection,
        )

    def available_repeats_for_selected_paths() -> list[str]:
        repeats: set[str] = set()
        for sample in selected_samples:
            selected_paths = selected_paths_for_output(sample)
            selected_set = set(selected_paths) if selected_paths is not None else None
            for record in capacity_file_records(sample, folder_map[sample], root_dir):
                if selected_set is not None and str(record["relative_path"]) not in selected_set:
                    continue
                repeats.add(str(record.get("repeat") or "repeat"))
        return sorted(repeats)

    def cycling_comparison_samples_from_style(style: dict[str, object]) -> list[str]:
        compare_samples = [
            sample for sample in list(style.get("compare_samples", selected_samples))
            if sample in selected_samples
        ]
        return compare_samples or list(selected_samples)

    def render_final_output_cache(cache: dict[str, object]) -> None:
        rendered_outputs = cache["rendered_outputs"]
        summary_df = cache["summary_df"]
        selected_file_summary_df = cache["selected_file_summary_df"]
        selection_rows = cache["selection_rows"]

        st.subheader("Final figures")
        output_cols = st.columns(2)
        for i, item in enumerate(rendered_outputs):
            display_name = str(item.get("sample", item.get("repeat", "comparison")))
            safe_name = safe_filename(display_name)
            with output_cols[i % 2]:
                st.markdown(f"#### {display_name}")
                display_fig = render_capacity_output_figure(item)
                st.pyplot(display_fig, clear_figure=True)
                plt.close(display_fig)

                caption = f"Files: {item['files_plotted']} | Points: {len(item['plot_df'])}"
                if bool(item["adjusted_limits"]) and int(item["numeric_points"]) > 0:
                    caption += " | Axis range auto-expanded to show data"
                st.caption(caption)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "CSV",
                        data=item["csv_bytes"],
                        file_name=item["csv_file_name"],
                        mime="text/csv",
                        key=f"download_csv_live_{safe_name}",
                    )
                with d2:
                    st.download_button(
                        "PNG",
                        data=item["png_bytes"],
                        file_name=item["png_file_name"],
                        mime="image/png",
                        key=f"download_png_live_{safe_name}",
                    )

                with st.expander("Data table"):
                    st.dataframe(item["plot_df"], use_container_width=True)

        st.success(f"Batch cycling analysis completed. Results saved to: `{cache['output_dir']}`")
        st.caption("Final figures are rendered from Matplotlib; PNG downloads and ZIP packaging use the saved figure data.")

        st.subheader("Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Selected-file metric summary")
        if selected_file_summary_df.empty:
            st.info("No selected-file summary rows were generated.")
        else:
            display_cols = ["Sample", "Repeat", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Note"]
            st.dataframe(
                format_selected_file_summary_for_display(selected_file_summary_df[display_cols]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download selected-file summary CSV",
                data=selected_file_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="capacity_selected_file_summary.csv",
                mime="text/csv",
            )

        st.download_button(
            "Download all cycling results ZIP",
            data=cache["zip_bytes"],
            file_name="capacity_batch_results.zip",
            mime="application/zip",
        )

        if selection_rows:
            with st.expander("File selection record"):
                st.dataframe(pd.DataFrame(selection_rows), use_container_width=True)

    if workflow_view == "1. Data preview & file selection":
        st.markdown("### Data preview & file selection")
        st.caption(
            "Each Excel file is shown as its own cycling plot. Valid files are shown first; unreadable or unusable files are moved to the end and unchecked by default. Save the current sample to continue to the next sample."
        )

        preview_axis_mode = "Fixed common range"
        preview_min_retention = min_retention

        if cycling_bulk_preview:
            if use_retention_filter:
                st.caption(f"Current retention cutoff: {float(min_capacity_retention):g}%. The same cutoff is applied to file previews, final sample plots, and exported CSV files.")
            else:
                st.caption("Retention cutoff is disabled. File previews, final sample plots, and exported CSV files will keep all valid points.")

            all_loaded_entries: dict[str, list[tuple[dict[str, object], pd.DataFrame | None, pd.DataFrame | None, str | None]]] = {}
            total_files = sum(len(capacity_file_records(sample, folder_map[sample], root_dir)) for sample in selected_samples)
            all_jobs = [
                (sample, record)
                for sample in selected_samples
                for record in capacity_file_records(sample, folder_map[sample], root_dir)
            ]
            bulk_signature = hashlib.sha1(
                repr(
                    {
                        "root_dir": str(root_dir),
                        "selected_samples": selected_samples,
                        "files": {
                            sample: [file_record_signature(record) for record in capacity_file_records(sample, folder_map[sample], root_dir)]
                            for sample in selected_samples
                        },
                        "sheet_name": sheet_name,
                        "capacity_col": capacity_col,
                        "efficiency_col": efficiency_col,
                        "skip_initial_rows": int(skip_initial_rows),
                        "preview_min_retention": preview_min_retention,
                        "implementation": "cycling_bulk_preview_v1",
                    }
                ).encode("utf-8")
            ).hexdigest()
            cached_bulk = st.session_state.get("cycling_bulk_preview_cache")
            cache_is_current = bool(cached_bulk and cached_bulk.get("signature") == bulk_signature)
            loaded_from_cache = cache_is_current
            reload_col, cache_col = st.columns([1, 3])
            with reload_col:
                if st.button("Reload preview data", key="cycling_reload_bulk_preview", use_container_width=True):
                    st.session_state.pop("cycling_bulk_preview_cache", None)
                    cached_bulk = None
                    cache_is_current = False
                    loaded_from_cache = False
            with cache_col:
                st.caption("Preview data is reused while files and data settings stay unchanged.")

            sample_default_state = {
                sample: (
                    sample in st.session_state.get("cycling_saved_selection", {}),
                    f"cycling_valid_defaults_applied_{stable_key_part(sample)}",
                    bool(st.session_state.get(f"cycling_valid_defaults_applied_{stable_key_part(sample)}", False)),
                )
                for sample in selected_samples
            }

            def cycling_worker_args(job: tuple[str, dict[str, object]]):
                sample, record = job
                return (
                    sample,
                    record,
                    root_dir,
                    sheet_name,
                    capacity_col,
                    efficiency_col,
                    int(skip_initial_rows),
                    preview_min_retention,
                    cycling_persistent_cache_dir,
                )

            if cache_is_current:
                all_loaded_entries = cached_bulk["all_loaded_entries"]
            else:
                completed = 0
                preview_progress = st.progress(0)
                preview_status = st.empty()
                if cycling_parallel_load and all_jobs:
                    max_workers = min(cycling_parallel_workers, len(all_jobs))
                    executor_cls = ProcessPoolExecutor if cycling_parallel_backend == "Processes" else ThreadPoolExecutor

                    def consume_cycling_executor(executor) -> None:
                        nonlocal completed
                        futures = [
                            executor.submit(read_cycling_preview_job_worker, cycling_worker_args(job))
                            for job in all_jobs
                        ]
                        for future in as_completed(futures):
                            sample, record, file_df, raw_df, combined_error = future.result()
                            completed += 1
                            preview_status.write(f"Reading {completed}/{total_files}: {sample} / {record['source_file']}")
                            saved_selection_exists, _default_marker_key, defaults_already_applied = sample_default_state[sample]
                            key = cycling_file_include_key(sample, str(record["relative_path"]))
                            if file_df is None:
                                st.session_state[key] = False
                            elif not saved_selection_exists and not defaults_already_applied:
                                st.session_state[key] = True
                            all_loaded_entries.setdefault(sample, []).append((record, file_df, raw_df, combined_error))
                            preview_progress.progress(completed / max(1, total_files))

                    try:
                        with executor_cls(max_workers=max_workers) as executor:
                            consume_cycling_executor(executor)
                    except Exception as exc:
                        if cycling_parallel_backend != "Processes":
                            raise
                        st.warning(f"Process backend could not start or complete: {exc}. Falling back to Threads.")
                        completed = 0
                        all_loaded_entries = {}
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            consume_cycling_executor(executor)
                else:
                    for job in all_jobs:
                        sample, record, file_df, raw_df, combined_error = read_cycling_preview_job_worker(cycling_worker_args(job))
                        completed += 1
                        preview_status.write(f"Reading {completed}/{total_files}: {sample} / {record['source_file']}")
                        saved_selection_exists, _default_marker_key, defaults_already_applied = sample_default_state[sample]
                        key = cycling_file_include_key(sample, str(record["relative_path"]))
                        if file_df is None:
                            st.session_state[key] = False
                        elif not saved_selection_exists and not defaults_already_applied:
                            st.session_state[key] = True
                        all_loaded_entries.setdefault(sample, []).append((record, file_df, raw_df, combined_error))
                        preview_progress.progress(completed / max(1, total_files))

                for sample in selected_samples:
                    saved_selection_exists, default_marker_key, defaults_already_applied = sample_default_state[sample]
                    if not saved_selection_exists and not defaults_already_applied:
                        st.session_state[default_marker_key] = True
                    order = {str(record["relative_path"]): i for i, record in enumerate(capacity_file_records(sample, folder_map[sample], root_dir))}
                    all_loaded_entries[sample] = sorted(
                        all_loaded_entries.get(sample, []),
                        key=lambda entry: order.get(str(entry[0]["relative_path"]), 10**9),
                    )
                st.session_state["cycling_bulk_preview_cache"] = {
                    "signature": bulk_signature,
                    "all_loaded_entries": all_loaded_entries,
                }
                preview_status.empty()
                preview_progress.empty()

            total_valid = sum(1 for entries in all_loaded_entries.values() for entry in entries if entry[1] is not None)
            total_invalid = total_files - total_valid
            selected_total = sum(len(selected_relative_paths_for_sample_saved_first(sample, folder_map[sample], root_dir, True)) for sample in selected_samples)
            st.success(f"Loaded {total_files} files across {len(selected_samples)} samples. Valid: {total_valid}; unavailable: {total_invalid}; selected: {selected_total}.")
            if not loaded_from_cache:
                notify_load_all_complete(
                    notification_id=f"cycling_load_all_complete_{bulk_signature}",
                    title="Cycling data preview loaded",
                    body=f"Loaded {total_files} cycling files across {len(selected_samples)} samples. Valid: {total_valid}; unavailable: {total_invalid}.",
                    enabled=bool(cycling_notify_load_complete),
                )

            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button("Select all valid", use_container_width=True, key="cycling_bulk_select_valid"):
                    for sample, entries in all_loaded_entries.items():
                        for record, file_df, _raw_df, _error in entries:
                            st.session_state[cycling_file_include_key(sample, str(record["relative_path"]))] = file_df is not None
                        save_selection_for_sample(sample, capacity_file_records(sample, folder_map[sample], root_dir))
                    rerun_streamlit_app()
            with b2:
                if st.button("Clear all", use_container_width=True, key="cycling_bulk_clear_all"):
                    for sample in selected_samples:
                        for record in capacity_file_records(sample, folder_map[sample], root_dir):
                            st.session_state[cycling_file_include_key(sample, str(record["relative_path"]))] = False
                        save_selection_for_sample(sample, capacity_file_records(sample, folder_map[sample], root_dir))
                    rerun_streamlit_app()
            with b3:
                st.caption(f"Current selection across all samples: {selected_total} / {total_files} files included.")

            summary_frames = []
            for sample in selected_samples:
                entries = all_loaded_entries[sample]
                selected_count = sum(
                    bool(st.session_state.get(cycling_file_include_key(sample, str(record["relative_path"])), False))
                    for record, _file_df, _raw_df, _error in entries
                )
                with st.expander(f"{sample} ({selected_count}/{len(entries)} selected)", expanded=True):
                    valid_entries = [entry for entry in entries if entry[1] is not None]
                    invalid_entries = [entry for entry in entries if entry[1] is None]
                    file_cols = st.columns(4)
                    for i, (record, file_df, raw_df, row_error) in enumerate(valid_entries + invalid_entries):
                        rel = str(record["relative_path"])
                        key = cycling_file_include_key(sample, rel)
                        with file_cols[i % 4]:
                            render_file_preview_card(
                                record=record,
                                file_df=file_df,
                                raw_df=raw_df,
                                error=row_error,
                                checkbox_key=key,
                                color_hex=default_color_map[sample],
                                preview_axis_mode=preview_axis_mode,
                                cap_y_min=75,
                                cap_y_max=110,
                                ce_y_min=90,
                                ce_y_max=100.5,
                            )
                    current_summary = load_capacity_current_sample_summary_from_entries(entries)
                    if not current_summary.empty:
                        summary_frames.append(current_summary)
                        display_cols = ["Repeat", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Note"]
                        st.dataframe(
                            format_selected_file_summary_for_display(current_summary[display_cols]),
                            use_container_width=True,
                            hide_index=True,
                        )

            if summary_frames:
                all_summary = pd.concat(summary_frames, ignore_index=True)
                st.download_button(
                    "Download selected-file summary CSV",
                    data=all_summary.to_csv(index=False).encode("utf-8"),
                    file_name="capacity_selected_file_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.button(
                "Save all selections and continue to style preview",
                type="primary",
                use_container_width=True,
                on_click=save_all_cycling_selections_and_go_style,
                args=(selected_samples, folder_map, root_dir),
            )
            return

        inspect_sample = st.selectbox(
            "Sample to inspect",
            options=selected_samples,
            key="cycling_inspect_sample",
            help="Review this sample's files. After saving, the app moves to the next selected sample.",
        )

        file_records = capacity_file_records(inspect_sample, folder_map[inspect_sample], root_dir)
        if not file_records:
            st.warning(f"No Excel files found for sample `{inspect_sample}`.")
            return

        control_col1, control_col2, control_col3 = st.columns([1.0, 1.0, 2.2])
        with control_col1:
            if st.button("Select all valid", use_container_width=True):
                for record in file_records:
                    rel = str(record["relative_path"])
                    key = cycling_file_include_key(inspect_sample, rel)
                    file_df, _ = load_cached_capacity_file(
                        file_path=record["path"],
                        sample_name=inspect_sample,
                        root_dir=root_dir,
                        sheet_name=sheet_name,
                        capacity_col=capacity_col,
                        efficiency_col=efficiency_col,
                        skip_initial_rows=int(skip_initial_rows),
                        min_retention=min_retention,
                        persistent_cache_dir=cycling_persistent_cache_dir,
                    )
                    st.session_state[key] = file_df is not None
                save_selection_for_sample(inspect_sample, file_records)
                st.rerun()
        with control_col2:
            if st.button("Clear all", use_container_width=True):
                for record in file_records:
                    st.session_state[cycling_file_include_key(inspect_sample, str(record["relative_path"]))] = False
                save_selection_for_sample(inspect_sample, file_records)
                st.rerun()
        with control_col3:
            st.caption(saved_selection_summary(inspect_sample, file_records))

        if use_retention_filter:
            st.caption(f"Current retention cutoff: {float(min_capacity_retention):g}%. The same cutoff is applied to file previews, final sample plots, and exported CSV files.")
        else:
            st.caption("Retention cutoff is disabled. File previews, final sample plots, and exported CSV files will keep all valid points.")

        loaded_entries: list[tuple[dict[str, object], pd.DataFrame | None, pd.DataFrame | None, str | None]] = []
        preview_progress = st.progress(0)
        preview_status = st.empty()
        for file_index, record in enumerate(file_records, start=1):
            preview_status.write(
                f"Reading {file_index}/{len(file_records)}: {record['source_file']}"
            )
            rel = str(record["relative_path"])
            key = cycling_file_include_key(inspect_sample, rel)
            file_df, error = load_cached_capacity_file(
                file_path=record["path"],
                sample_name=inspect_sample,
                root_dir=root_dir,
                sheet_name=sheet_name,
                capacity_col=capacity_col,
                efficiency_col=efficiency_col,
                skip_initial_rows=int(skip_initial_rows),
                min_retention=preview_min_retention,
                persistent_cache_dir=cycling_persistent_cache_dir,
            )
            raw_df, raw_error = load_raw_capacity_metrics_df(
                record=record,
                sample_name=inspect_sample,
                root_dir=root_dir,
                sheet_name=sheet_name,
                capacity_col=capacity_col,
                efficiency_col=efficiency_col,
                skip_initial_rows=int(skip_initial_rows),
                persistent_cache_dir=cycling_persistent_cache_dir,
            )
            combined_error = error or raw_error
            # Invalid/unusable files are always moved to the end and unchecked
            # before their checkbox is rendered.
            if file_df is None:
                st.session_state[key] = False
            loaded_entries.append((record, file_df, raw_df, combined_error))
            preview_progress.progress(file_index / len(file_records))
        preview_status.empty()
        preview_progress.empty()

        default_marker_key = f"cycling_valid_defaults_applied_{stable_key_part(inspect_sample)}"
        if (
            inspect_sample not in st.session_state.get("cycling_saved_selection", {})
            and not bool(st.session_state.get(default_marker_key, False))
        ):
            for record, file_df, _raw_df, _row_error in loaded_entries:
                rel = str(record["relative_path"])
                key = cycling_file_include_key(inspect_sample, rel)
                st.session_state[key] = file_df is not None
            st.session_state[default_marker_key] = True

        valid_entries = [entry for entry in loaded_entries if entry[1] is not None]
        invalid_entries = [entry for entry in loaded_entries if entry[1] is None]
        display_entries = valid_entries + invalid_entries

        selected_count = sum(
            bool(st.session_state.get(cycling_file_include_key(inspect_sample, str(r[0]["relative_path"])), True))
            for r in display_entries
        )
        st.info(
            f"{selected_count} / {len(file_records)} files currently selected for `{inspect_sample}`. "
            f"Valid: {len(valid_entries)}; unavailable: {len(invalid_entries)}."
        )

        st.markdown("#### File previews and checklist")
        st.caption("Each compact card shows the filtered preview plus ICE, Cycle Life, ACE, and ACE cycle. Valid files appear first; unavailable files are listed last and unchecked.")

        file_cols = st.columns(4)
        for i, (record, file_df, raw_df, row_error) in enumerate(display_entries):
            rel = str(record["relative_path"])
            key = cycling_file_include_key(inspect_sample, rel)
            with file_cols[i % 4]:
                render_file_preview_card(
                    record=record,
                    file_df=file_df,
                    raw_df=raw_df,
                    error=row_error,
                    checkbox_key=key,
                    color_hex=default_color_map[inspect_sample],
                    preview_axis_mode=preview_axis_mode,
                    cap_y_min=75,
                    cap_y_max=110,
                    ce_y_min=90,
                    ce_y_max=100.5,
                )

        current_sample_summary_df = load_capacity_current_sample_summary_from_entries(loaded_entries)
        st.markdown("#### Selected-file summary for this sample")
        if current_sample_summary_df.empty:
            st.info("No files are currently selected for this sample.")
        else:
            display_cols = ["Repeat", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Note"]
            st.dataframe(
                format_selected_file_summary_for_display(current_sample_summary_df[display_cols]),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download this sample summary CSV",
                data=current_sample_summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_filename(inspect_sample)}_selected_file_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        current_sample_index = selected_samples.index(inspect_sample)
        if current_sample_index < len(selected_samples) - 1:
            next_sample = selected_samples[current_sample_index + 1]
            save_button_label = f"Save this sample and continue to {shorten_label(next_sample, 28)}"
            save_help = "Save the current file selection and open the next selected sample for review."
        else:
            save_button_label = "Save this sample and continue to style preview"
            save_help = "Save the current file selection and move to figure style preview."

        save_col1, save_col2 = st.columns([1.35, 2.2])
        with save_col1:
            st.button(
                save_button_label,
                type="primary",
                use_container_width=True,
                on_click=save_current_cycling_selection_and_advance,
                args=(inspect_sample, selected_samples, folder_map, root_dir),
            )
        with save_col2:
            st.caption(save_help)

        st.divider()
        st.markdown("#### Selection overview for all selected samples")
        overview_rows = []
        for sample in selected_samples:
            records = capacity_file_records(sample, folder_map[sample], root_dir)
            saved = st.session_state["cycling_saved_selection"].get(sample)
            if saved is not None:
                selected = sum(
                    bool(saved.get(
                        str(r["relative_path"]),
                        st.session_state.get(cycling_file_include_key(sample, str(r["relative_path"])), True),
                    ))
                    for r in records
                )
                status = "saved"
            else:
                selected = sum(
                    bool(st.session_state.get(cycling_file_include_key(sample, str(r["relative_path"])), True))
                    for r in records
                )
                status = "not saved yet"
            overview_rows.append({
                "sample": sample,
                "selected_files": selected,
                "total_files": len(records),
                "status": status,
            })
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)
        return

    if workflow_view == "2. Style preview":
        st.markdown("### Style preview")
        st.caption("Tune figure style on the left. The preview on the right always uses real selected files after the retention cutoff.")
        if st.session_state.get("cycling_style_defaults_signature") != cycling_style_defaults_signature(selected_samples):
            apply_cycling_style_defaults_for_preview(selected_samples)

        unsaved_samples = [
            sample for sample in selected_samples
            if sample not in st.session_state.get("cycling_saved_selection", {})
        ]
        if unsaved_samples:
            st.warning(
                "Some selected samples have not been explicitly saved yet: "
                + ", ".join(shorten_label(s, 28) for s in unsaved_samples)
                + ". Go back to Data preview & file selection if you want to review them before output."
            )

        style_controls_col, style_preview_col = st.columns([0.9, 1.55], gap="large")

        with style_controls_col:
            compare_mode_enabled = is_cycling_compare_mode(st.session_state.get("cycling_plot_mode"))
            preview_sample = st.selectbox(
                "Preview sample",
                options=selected_samples,
                key="cycling_preview_sample",
                disabled=compare_mode_enabled,
                help="Disabled in compare mode because the preview is the single combined comparison figure.",
            )
            repeat_options = available_repeats_for_selected_paths()
            if repeat_options and st.session_state.get("cycling_compare_repeat") not in repeat_options:
                st.session_state["cycling_compare_repeat"] = repeat_options[0]
            st.selectbox(
                "Plot mode",
                [CYCLING_PLOT_MODE_SINGLE, CYCLING_PLOT_MODE_COMPARE],
                key="cycling_plot_mode",
                on_change=reset_cycling_visual_style_defaults,
                help="Changing mode resets visual style to the same default so only the plotted data grouping changes.",
            )
            compare_mode_enabled = is_cycling_compare_mode(st.session_state.get("cycling_plot_mode"))
            st.multiselect(
                "Samples in comparison",
                options=selected_samples,
                key="cycling_compare_samples",
                disabled=not compare_mode_enabled,
                help="In comparison mode, preview and final output combine all selected files from these samples into one figure.",
            )

            control_tab_1, control_tab_2, control_tab_3, control_tab_4 = st.tabs(
                ["Text", "Legend", "Axes", "Style"]
            )

            with control_tab_1:
                st.text_input(
                    "Plot title",
                    key="cycling_plot_title",
                    placeholder="{sample}",
                    help='Use "{sample}" to insert the sample folder name.',
                )
                st.text_input("X-axis label", key="cycling_x_label", placeholder="Cycle Index")
                st.text_input("Left Y-axis label", key="cycling_cap_y_label", placeholder="Capacity Retention (%)")
                st.text_input("Right Y-axis label", key="cycling_ce_y_label", placeholder="Coulombic Efficiency (%)")

            with control_tab_2:
                st.checkbox("Show legend", key="cycling_show_legend")
                show_legend = bool(st.session_state.get("cycling_show_legend", True))
                legend_position = st.selectbox(
                    "Legend position",
                    ["Top", "Right", "Inside", "Hide"],
                    key="cycling_legend_position",
                    disabled=not show_legend,
                )
                st.text_input("Legend title", key="cycling_legend_title", disabled=not show_legend)
                st.slider(
                    "Label length",
                    min_value=8,
                    max_value=80,
                    key="cycling_legend_label_max_len",
                    step=1,
                    disabled=not show_legend,
                )
                st.slider(
                    "Top legend columns",
                    min_value=1,
                    max_value=6,
                    key="cycling_legend_columns",
                    step=1,
                    disabled=(not show_legend or legend_position != "Top"),
                )

            with control_tab_3:
                st.checkbox("Auto X-axis range", key="cycling_auto_x_range")
                auto_x_range = bool(st.session_state.get("cycling_auto_x_range", True))
                x1, x2 = st.columns(2)
                with x1:
                    st.number_input("X min", step=10.0, key="cycling_x_min", disabled=auto_x_range)
                with x2:
                    st.number_input("X max", step=10.0, key="cycling_x_max", disabled=auto_x_range)

                y1, y2 = st.columns(2)
                with y1:
                    st.number_input("Cap. Y min", step=1.0, key="cycling_cap_y_min")
                    st.number_input("CE Y min", step=0.5, key="cycling_ce_y_min")
                with y2:
                    st.number_input("Cap. Y max", step=1.0, key="cycling_cap_y_max")
                    st.number_input("CE Y max", step=0.5, key="cycling_ce_y_max")

            with control_tab_4:
                st.selectbox(
                    "Default color palette",
                    ["Set2 + Dark2 + tab20", "Set2", "Dark2", "tab10", "tab20", "tab20 + tab20b"],
                    key="cycling_palette_name",
                )
                st.slider("Marker size", min_value=20, max_value=200, key="cycling_marker_size", step=5)

                f1, f2 = st.columns(2)
                with f1:
                    st.number_input("Figure width", min_value=4.0, max_value=20.0, key="cycling_fig_width", step=0.5)
                    st.number_input("DPI", min_value=72, max_value=600, key="cycling_dpi", step=50)
                with f2:
                    st.number_input("Figure height", min_value=3.0, max_value=15.0, key="cycling_fig_height", step=0.5)

                style = current_style_values()
                colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
                palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}

                with st.expander("Sample colors", expanded=False):
                    for i, sample in enumerate(selected_samples, start=1):
                        st.color_picker(
                            compact_widget_label("Color", i, sample, max_len=18),
                            value=st.session_state.get(f"cycling_color_{safe_filename(sample)}", palette_color_map[sample]),
                            key=f"cycling_color_{safe_filename(sample)}",
                            help=f"Full sample name: {sample}",
                        )

            st.button(
                "Generate final outputs",
                type="primary",
                use_container_width=True,
                on_click=save_cycling_style_and_go_final,
                args=(selected_samples, sample_names),
            )

        with style_preview_col:
            st.markdown("### Live style preview")
            style = current_style_values()
            sample_colors = current_sample_colors(style)
            selected_preview_paths = selected_paths_for_output(preview_sample)
            plot_mode = str(style.get("plot_mode", CYCLING_PLOT_MODE_SINGLE))

            if is_cycling_compare_mode(plot_mode):
                preview_frames = []
                preview_file_count = 0
                for sample in cycling_comparison_samples_from_style(style):
                    selected_paths = selected_paths_for_output(sample)
                    if selected_paths is not None and len(selected_paths) == 0:
                        continue
                    with st.spinner(f"Loading selected files for {sample}..."):
                        sample_df, sample_file_count = load_capacity_sample_plot_data(
                            sample_name=sample,
                            sample_dir=folder_map[sample],
                            root_dir=root_dir,
                            sheet_name=sheet_name,
                            capacity_col=capacity_col,
                            efficiency_col=efficiency_col,
                            skip_initial_rows=int(skip_initial_rows),
                            min_retention=min_retention,
                            top_n_value=None,
                            selected_relative_paths=selected_paths,
                            persistent_cache_dir=cycling_persistent_cache_dir,
                        )
                    preview_file_count += sample_file_count
                    if sample_df is not None and not sample_df.empty:
                        preview_frames.append(sample_df)
                preview_df = pd.concat(preview_frames, ignore_index=True) if preview_frames else None
            else:
                if selected_preview_paths is not None and len(selected_preview_paths) == 0:
                    preview_df, preview_file_count = None, 0
                else:
                    with st.spinner(f"Loading selected preview files for {preview_sample}..."):
                        preview_df, preview_file_count = load_capacity_sample_plot_data(
                            sample_name=preview_sample,
                            sample_dir=folder_map[preview_sample],
                            root_dir=root_dir,
                            sheet_name=sheet_name,
                            capacity_col=capacity_col,
                            efficiency_col=efficiency_col,
                            skip_initial_rows=int(skip_initial_rows),
                            min_retention=min_retention,
                            top_n_value=top_n_value,
                            selected_relative_paths=selected_preview_paths,
                            persistent_cache_dir=cycling_persistent_cache_dir,
                        )

            if preview_file_count == 0:
                st.warning("No selected Excel files found for preview.")
            elif preview_df is None:
                st.warning("No valid cycling data found for this preview.")
            else:
                if is_cycling_compare_mode(plot_mode):
                    preview_fig = make_capacity_sample_comparison_figure(
                        plot_df=preview_df,
                        repeat_name="Selected sample comparison",
                        sample_colors=sample_colors,
                        plot_title=str(style["plot_title"]),
                        x_label=str(style["x_label"]),
                        cap_y_label=str(style["cap_y_label"]),
                        ce_y_label=str(style["ce_y_label"]),
                        legend_title=str(style["legend_title"]),
                        show_legend=bool(style["show_legend"]),
                        legend_position=str(style["legend_position"]),
                        legend_label_max_len=int(style["legend_label_max_len"]),
                        legend_columns=int(style["legend_columns"]),
                        auto_x_range=bool(style["auto_x_range"]),
                        x_min=float(style["x_min"]),
                        x_max=float(style["x_max"]),
                        cap_y_min=float(style["cap_y_min"]),
                        cap_y_max=float(style["cap_y_max"]),
                        ce_y_min=float(style["ce_y_min"]),
                        ce_y_max=float(style["ce_y_max"]),
                        marker_size=int(style["marker_size"]),
                        fig_width=float(style["fig_width"]),
                        fig_height=float(style["fig_height"]),
                    )
                else:
                    preview_fig = make_capacity_figure(
                        plot_df=preview_df,
                        sample_name=preview_sample,
                        color_hex=sample_colors[preview_sample],
                        plot_title=str(style["plot_title"]),
                        x_label=str(style["x_label"]),
                        cap_y_label=str(style["cap_y_label"]),
                        ce_y_label=str(style["ce_y_label"]),
                        legend_title=str(style["legend_title"]),
                        show_legend=bool(style["show_legend"]),
                        legend_position=str(style["legend_position"]),
                        legend_label_max_len=int(style["legend_label_max_len"]),
                        legend_columns=int(style["legend_columns"]),
                        auto_x_range=bool(style["auto_x_range"]),
                        x_min=float(style["x_min"]),
                        x_max=float(style["x_max"]),
                        cap_y_min=float(style["cap_y_min"]),
                        cap_y_max=float(style["cap_y_max"]),
                        ce_y_min=float(style["ce_y_min"]),
                        ce_y_max=float(style["ce_y_max"]),
                        marker_size=int(style["marker_size"]),
                        fig_width=float(style["fig_width"]),
                        fig_height=float(style["fig_height"]),
                    )
                st.pyplot(preview_fig, clear_figure=True)
                plt.close(preview_fig)

                c1, c2, c3 = st.columns(3)
                c1.metric("Preview files", preview_df["relative_path"].nunique() if "relative_path" in preview_df else preview_df["source_file"].nunique())
                c2.metric("Preview points", len(preview_df))
                c3.metric("Max cycle", _fmt_num(preview_df["cycle_index"].max()))
        return

    # Final output step
    st.markdown("### Final output")
    st.caption("Generating, saving, previewing, and packaging the selected sample plots. Output figures are shown two per row for compact review.")

    style = current_style_values()
    sample_colors = current_sample_colors(style)
    selected_paths_by_sample = {sample: selected_paths_for_output(sample) for sample in selected_samples}
    final_signature = hashlib.sha1(
        repr(
            {
                "root_dir": str(root_dir),
                "output_dir": str(output_dir),
                "selected_samples": selected_samples,
                "selected_paths": selected_paths_by_sample,
                "files": {
                    sample: [file_record_signature(record) for record in capacity_file_records(sample, folder_map[sample], root_dir)]
                    for sample in selected_samples
                },
                "sheet_name": sheet_name,
                "capacity_col": capacity_col,
                "efficiency_col": efficiency_col,
                "skip_initial_rows": int(skip_initial_rows),
                "min_retention": min_retention,
                "top_n_value": top_n_value,
                "style": style,
                "sample_colors": sample_colors,
                "implementation": "cycling_compare_all_bulk_v2",
            }
        ).encode("utf-8")
    ).hexdigest()

    unsaved_samples = [
        sample for sample in selected_samples
        if sample not in st.session_state.get("cycling_saved_selection", {})
    ]
    if unsaved_samples:
        st.warning(
            "Some selected samples have not been explicitly saved yet: "
            + ", ".join(shorten_label(s, 28) for s in unsaved_samples)
            + ". Go back to Data preview & file selection if you want to review them before output."
        )

    st.button(
        "Back to style preview",
        use_container_width=False,
        on_click=set_cycling_workflow_step,
        args=("2. Style preview",),
    )

    cached_output = st.session_state.get("cycling_final_output_cache")
    cache_is_current = bool(
        cached_output
        and cached_output.get("signature") == final_signature
        and all("figure_kwargs" in item for item in cached_output.get("rendered_outputs", []))
    )
    if cached_output and not cache_is_current:
        st.session_state.pop("cycling_final_output_cache", None)
    if cache_is_current:
        if st.button("Regenerate final outputs", type="primary", use_container_width=False):
            st.session_state.pop("cycling_final_output_cache", None)
        else:
            render_final_output_cache(cached_output)
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    selection_rows = []
    selected_file_summary_frames: list[pd.DataFrame] = []
    rendered_outputs: list[dict[str, object]] = []
    zip_buffer = io.BytesIO()

    progress = st.progress(0)
    status = st.empty()
    plot_mode = str(style.get("plot_mode", CYCLING_PLOT_MODE_SINGLE))

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        if is_cycling_compare_mode(plot_mode):
            comparison_frames = []
            total_files_found = 0
            compare_samples = cycling_comparison_samples_from_style(style)
            for idx, sample_name in enumerate(compare_samples, start=1):
                status.write(f"Processing {sample_name} ({idx}/{len(compare_samples)})...")
                selected_paths = selected_paths_by_sample[sample_name]
                selected_paths_set = set(selected_paths) if selected_paths is not None else None

                for record in capacity_file_records(sample_name, folder_map[sample_name], root_dir):
                    rel = str(record["relative_path"])
                    included = selected_paths_set is None or rel in selected_paths_set
                    selection_rows.append(
                        {
                            "sample": sample_name,
                            "repeat": record.get("repeat", ""),
                            "source_file": record["source_file"],
                            "relative_path": rel,
                            "included": included,
                        }
                    )

                plot_df, excel_file_count = load_capacity_sample_plot_data(
                    sample_name=sample_name,
                    sample_dir=folder_map[sample_name],
                    root_dir=root_dir,
                    sheet_name=sheet_name,
                    capacity_col=capacity_col,
                    efficiency_col=efficiency_col,
                    skip_initial_rows=int(skip_initial_rows),
                    min_retention=min_retention,
                    top_n_value=None,
                    selected_relative_paths=selected_paths,
                    persistent_cache_dir=cycling_persistent_cache_dir,
                )
                total_files_found += excel_file_count

                sample_file_summary_df = load_capacity_sample_file_summary(
                    sample_name=sample_name,
                    sample_dir=folder_map[sample_name],
                    root_dir=root_dir,
                    sheet_name=sheet_name,
                    capacity_col=capacity_col,
                    efficiency_col=efficiency_col,
                    skip_initial_rows=int(skip_initial_rows),
                    selected_relative_paths=selected_paths,
                    persistent_cache_dir=cycling_persistent_cache_dir,
                )
                if not sample_file_summary_df.empty:
                    selected_file_summary_frames.append(sample_file_summary_df)

                if plot_df is not None and not plot_df.empty:
                    comparison_frames.append(plot_df)
                progress.progress(idx / len(compare_samples))

            if comparison_frames:
                plot_df = pd.concat(comparison_frames, ignore_index=True)
                safe_name = safe_filename("selected_sample_comparison")
                csv_path = output_dir / f"{safe_name}_plot_data.csv"
                png_path = output_dir / f"{safe_name}_capacity_summary.png"
                plot_df.to_csv(csv_path, index=False)
                effective_limits, numeric_plot_df, adjusted_limits = capacity_figure_limits(plot_df, style)

                figure_kwargs = dict(
                    plot_df=plot_df,
                    repeat_name="Selected sample comparison",
                    sample_colors=sample_colors,
                    plot_title=str(style["plot_title"]),
                    x_label=str(style["x_label"]),
                    cap_y_label=str(style["cap_y_label"]),
                    ce_y_label=str(style["ce_y_label"]),
                    legend_title=str(style["legend_title"]),
                    show_legend=bool(style["show_legend"]),
                    legend_position=str(style["legend_position"]),
                    legend_label_max_len=int(style["legend_label_max_len"]),
                    legend_columns=int(style["legend_columns"]),
                    auto_x_range=bool(effective_limits["auto_x_range"]),
                    x_min=float(effective_limits["x_min"]),
                    x_max=float(effective_limits["x_max"]),
                    cap_y_min=float(effective_limits["cap_y_min"]),
                    cap_y_max=float(effective_limits["cap_y_max"]),
                    ce_y_min=float(effective_limits["ce_y_min"]),
                    ce_y_max=float(effective_limits["ce_y_max"]),
                    marker_size=int(style["marker_size"]),
                    fig_width=float(style["fig_width"]),
                    fig_height=float(style["fig_height"]),
                )

                save_fig = make_capacity_sample_comparison_figure(**figure_kwargs)
                save_fig.canvas.draw()
                save_fig.savefig(png_path, dpi=int(style["dpi"]), bbox_inches="tight")
                png_buffer = io.BytesIO()
                save_fig.savefig(png_buffer, format="png", dpi=int(style["dpi"]), bbox_inches="tight")
                png_buffer.seek(0)
                png_bytes = png_buffer.getvalue()
                plt.close(save_fig)

                csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
                zipf.writestr(f"{safe_name}/{safe_name}_plot_data.csv", csv_bytes)
                zipf.writestr(f"{safe_name}/{safe_name}_capacity_summary.png", png_bytes)

                rendered_outputs.append(
                    {
                        "plot_kind": "sample_comparison",
                        "repeat": "Selected sample comparison",
                        "csv_bytes": csv_bytes,
                        "png_bytes": png_bytes,
                        "csv_file_name": f"{safe_name}_plot_data.csv",
                        "png_file_name": f"{safe_name}_capacity_summary.png",
                        "plot_df": plot_df,
                        "csv_path": str(csv_path),
                        "png_path": str(png_path),
                        "files_plotted": plot_df["relative_path"].nunique(),
                        "adjusted_limits": adjusted_limits,
                        "numeric_points": len(numeric_plot_df),
                        "figure_kwargs": figure_kwargs,
                    }
                )
                summary_rows.append(
                    {
                        "sample": "sample comparison",
                        "repeat": "all selected",
                        "files_found": total_files_found,
                        "files_plotted": rendered_outputs[-1]["files_plotted"],
                        "points_plotted": len(plot_df),
                        "color": "by sample",
                        "csv_path": str(csv_path),
                        "png_path": str(png_path),
                    }
                )
            else:
                st.warning("No selected cycling data found for the selected comparison samples.")

        samples_to_process = [] if is_cycling_compare_mode(plot_mode) else selected_samples
        for idx, sample_name in enumerate(samples_to_process, start=1):
            status.write(f"Processing {sample_name} ({idx}/{len(selected_samples)})...")

            selected_paths = selected_paths_by_sample[sample_name]
            selected_paths_set = set(selected_paths) if selected_paths is not None else None

            for record in capacity_file_records(sample_name, folder_map[sample_name], root_dir):
                rel = str(record["relative_path"])
                included = selected_paths_set is None or rel in selected_paths_set
                selection_rows.append(
                    {
                        "sample": sample_name,
                        "source_file": record["source_file"],
                        "relative_path": rel,
                        "included": included,
                    }
                )

            plot_df, excel_file_count = load_capacity_sample_plot_data(
                sample_name=sample_name,
                sample_dir=folder_map[sample_name],
                root_dir=root_dir,
                sheet_name=sheet_name,
                capacity_col=capacity_col,
                efficiency_col=efficiency_col,
                skip_initial_rows=int(skip_initial_rows),
                min_retention=min_retention,
                top_n_value=top_n_value,
                selected_relative_paths=selected_paths,
                persistent_cache_dir=cycling_persistent_cache_dir,
            )

            if excel_file_count == 0:
                st.warning(f"No Excel files found for sample `{sample_name}`.")
                progress.progress(idx / len(selected_samples))
                continue

            if plot_df is None:
                st.warning(f"No valid selected cycling data found for sample `{sample_name}`.")
                progress.progress(idx / len(selected_samples))
                continue

            sample_file_summary_df = load_capacity_sample_file_summary(
                sample_name=sample_name,
                sample_dir=folder_map[sample_name],
                root_dir=root_dir,
                sheet_name=sheet_name,
                capacity_col=capacity_col,
                efficiency_col=efficiency_col,
                skip_initial_rows=int(skip_initial_rows),
                selected_relative_paths=selected_paths,
                persistent_cache_dir=cycling_persistent_cache_dir,
            )
            if not sample_file_summary_df.empty:
                selected_file_summary_frames.append(sample_file_summary_df)

            safe_name = safe_filename(sample_name)
            sample_output_dir = output_dir / safe_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = sample_output_dir / f"{safe_name}_plot_data.csv"
            png_path = sample_output_dir / f"{safe_name}_capacity_summary.png"

            plot_df.to_csv(csv_path, index=False)
            effective_limits, numeric_plot_df, adjusted_limits = capacity_figure_limits(plot_df, style)

            figure_kwargs = dict(
                plot_df=plot_df,
                sample_name=sample_name,
                color_hex=sample_colors[sample_name],
                plot_title=str(style["plot_title"]),
                x_label=str(style["x_label"]),
                cap_y_label=str(style["cap_y_label"]),
                ce_y_label=str(style["ce_y_label"]),
                legend_title=str(style["legend_title"]),
                show_legend=bool(style["show_legend"]),
                legend_position=str(style["legend_position"]),
                legend_label_max_len=int(style["legend_label_max_len"]),
                legend_columns=int(style["legend_columns"]),
                auto_x_range=bool(effective_limits["auto_x_range"]),
                x_min=float(effective_limits["x_min"]),
                x_max=float(effective_limits["x_max"]),
                cap_y_min=float(effective_limits["cap_y_min"]),
                cap_y_max=float(effective_limits["cap_y_max"]),
                ce_y_min=float(effective_limits["ce_y_min"]),
                ce_y_max=float(effective_limits["ce_y_max"]),
                marker_size=int(style["marker_size"]),
                fig_width=float(style["fig_width"]),
                fig_height=float(style["fig_height"]),
            )

            # Render once to PNG and use the same bytes for preview/download.
            # This avoids Streamlit/Matplotlib figure-clear timing issues.
            save_fig = make_capacity_figure(**figure_kwargs)
            save_fig.canvas.draw()
            save_fig.savefig(png_path, dpi=int(style["dpi"]), bbox_inches="tight")

            png_buffer = io.BytesIO()
            save_fig.savefig(png_buffer, format="png", dpi=int(style["dpi"]), bbox_inches="tight")
            png_buffer.seek(0)
            png_bytes = png_buffer.getvalue()
            plt.close(save_fig)

            csv_bytes = plot_df.to_csv(index=False).encode("utf-8")

            zipf.writestr(f"{safe_name}/{safe_name}_plot_data.csv", csv_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_capacity_summary.png", png_bytes)

            files_plotted = plot_df["relative_path"].nunique() if "relative_path" in plot_df else plot_df["source_file"].nunique()

            rendered_outputs.append(
                {
                    "sample": sample_name,
                    "csv_bytes": csv_bytes,
                    "png_bytes": png_bytes,
                    "csv_file_name": f"{safe_name}_plot_data.csv",
                    "png_file_name": f"{safe_name}_capacity_summary.png",
                    "plot_df": plot_df,
                    "csv_path": str(csv_path),
                    "png_path": str(png_path),
                    "files_plotted": files_plotted,
                    "adjusted_limits": adjusted_limits,
                    "numeric_points": len(numeric_plot_df),
                    "figure_kwargs": figure_kwargs,
                }
            )

            summary_rows.append(
                {
                    "sample": sample_name,
                    "files_found": excel_file_count,
                    "files_plotted": rendered_outputs[-1]["files_plotted"],
                    "points_plotted": len(plot_df),
                    "color": sample_colors[sample_name],
                    "csv_path": str(csv_path),
                    "png_path": str(png_path),
                }
            )

            progress.progress(idx / len(selected_samples))

        if selection_rows:
            selection_df_for_zip = pd.DataFrame(selection_rows)
            zipf.writestr("capacity_file_selection.csv", selection_df_for_zip.to_csv(index=False).encode("utf-8"))

        if selected_file_summary_frames:
            selected_file_summary_for_zip = pd.concat(selected_file_summary_frames, ignore_index=True)
            zipf.writestr(
                "capacity_selected_file_summary.csv",
                selected_file_summary_for_zip.to_csv(index=False).encode("utf-8"),
            )

    status.empty()
    progress.empty()

    if not summary_rows:
        st.warning("No valid results were generated.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "capacity_batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if selection_rows:
        selection_df = pd.DataFrame(selection_rows)
        selection_path = output_dir / "capacity_file_selection.csv"
        selection_df.to_csv(selection_path, index=False)

    selected_file_summary_df = (
        pd.concat(selected_file_summary_frames, ignore_index=True)
        if selected_file_summary_frames
        else pd.DataFrame(columns=["Sample", "Repeat", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Relative path", "Note"])
    )
    selected_file_summary_path = output_dir / "capacity_selected_file_summary.csv"
    selected_file_summary_df.to_csv(selected_file_summary_path, index=False)

    zip_buffer.seek(0)
    st.session_state["cycling_final_output_cache"] = {
        "signature": final_signature,
        "output_dir": str(output_dir),
        "rendered_outputs": rendered_outputs,
        "summary_df": summary_df,
        "selected_file_summary_df": selected_file_summary_df,
        "selection_rows": selection_rows,
        "zip_bytes": zip_buffer.getvalue(),
    }
    rerun_streamlit_app()


# -----------------------------------------------------------------------------
# Stripping-cell batch analysis
# -----------------------------------------------------------------------------


def ensure_stripping_selection_store() -> None:
    if "stripping_saved_selection" not in st.session_state:
        st.session_state["stripping_saved_selection"] = {}


STRIPPING_PLOT_MODE_SINGLE = "Single sample repeat overlay"
STRIPPING_PLOT_MODE_COMPARE = "Compare all selected samples"
STRIPPING_COMPARE_MODE_ALIASES = {
    STRIPPING_PLOT_MODE_COMPARE,
    "Compare all selected samples by one repeat",
    "Compare samples by one repeat",
}


def is_stripping_compare_mode(plot_mode: object) -> bool:
    return str(plot_mode) in STRIPPING_COMPARE_MODE_ALIASES


def normalize_stripping_plot_mode(plot_mode: object) -> str:
    return STRIPPING_PLOT_MODE_COMPARE if is_stripping_compare_mode(plot_mode) else STRIPPING_PLOT_MODE_SINGLE


def stripping_file_include_key(sample: str, repeat: str, source_path: str) -> str:
    return f"stripping_include_{stable_key_part(sample)}_{stable_key_part(repeat)}_{stable_key_part(source_path)}"


def collect_stripping_file_records(root_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir_name = output_dir.name if output_dir.parent.resolve() == root_dir.resolve() else "__streamlit_stripping_outputs__"
    files = stripping.collect_excel_files(root_dir, output_dir_name=output_dir_name)
    records = []
    for file_info in files:
        try:
            if output_dir.resolve() in file_info.path.resolve().parents:
                continue
        except Exception:
            pass
        records.append(
            {
                "sample": file_info.sample,
                "repeat": file_info.repeat,
                "source_file": file_info.path.name,
                "relative_path": str(file_info.path.relative_to(root_dir)),
                "path": file_info.path,
            }
        )
    return records


def save_stripping_selection_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> None:
    ensure_stripping_selection_store()
    saved = {}
    for record in file_records:
        rel = str(record["relative_path"])
        key = stripping_file_include_key(sample_name, str(record["repeat"]), rel)
        saved[rel] = bool(st.session_state.get(key, True))
    st.session_state["stripping_saved_selection"][sample_name] = saved


def sync_stripping_checkbox_to_saved(sample_name: str, relative_path: str, checkbox_key: str) -> None:
    """Persist one stripping checkbox immediately so page navigation does not lose it."""
    ensure_stripping_selection_store()
    all_saved = dict(st.session_state["stripping_saved_selection"])
    saved = dict(all_saved.get(sample_name, {}))
    saved[relative_path] = bool(st.session_state.get(checkbox_key, True))
    all_saved[sample_name] = saved
    st.session_state["stripping_saved_selection"] = all_saved


def save_current_stripping_selection_and_advance(
    current_sample: str,
    selected_samples: list[str],
    records_by_sample: dict[str, list[dict[str, object]]],
) -> None:
    """Save the current stripping sample selection and advance the workflow."""
    ensure_stripping_selection_store()
    if current_sample not in selected_samples:
        return

    save_stripping_selection_for_sample(current_sample, records_by_sample[current_sample])
    current_idx = selected_samples.index(current_sample)
    if current_idx < len(selected_samples) - 1:
        st.session_state["stripping_inspect_sample"] = selected_samples[current_idx + 1]
        st.session_state["stripping_workflow_step"] = "1. Data preview & file selection"
    else:
        apply_stripping_style_defaults_for_preview(selected_samples)
        st.session_state["stripping_workflow_step"] = "2. Style preview"


def set_stripping_workflow_step(step: str) -> None:
    """Set the stripping workflow step from a Streamlit callback."""
    st.session_state["stripping_workflow_step"] = step


def reset_stripping_visual_style_defaults() -> None:
    """Keep stripping plot-mode changes from carrying stale visual styling."""
    normalization = st.session_state.get("stripping_normalization", "area")
    if normalization == "area":
        x_label = "Capacity (mAh/cm$^2$)"
    elif normalization == "test-dchg":
        x_label = "Normalized capacity by test DChg divisor"
    else:
        x_label = "Capacity (mAh)"

    defaults = {
        "stripping_plot_title": "{sample}",
        "stripping_x_label": x_label,
        "stripping_y_label": "Voltage (V)",
        "stripping_show_legend": True,
        "stripping_legend_position": "Top",
        "stripping_legend_title": "Repeats",
        "stripping_legend_label_max_len": 28,
        "stripping_legend_columns": 3,
        "stripping_auto_x_range": True,
        "stripping_x_min": -0.5,
        "stripping_x_max": 7.5,
        "stripping_small_capacity_limit": 2.0,
        "stripping_large_capacity_limit": 7.5,
        "stripping_y_min": -1.0,
        "stripping_y_max": 0.2,
        "stripping_palette_name": "Set2 + Dark2 + tab20",
        "stripping_linewidth": 2.2,
        "stripping_fig_width": 6.0,
        "stripping_fig_height": 4.6,
        "stripping_dpi": 300,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def stripping_style_defaults_signature(selected_samples: list[str]) -> str:
    return hashlib.sha1(
        repr(
            {
                "selected_samples": list(selected_samples),
                "normalization": st.session_state.get("stripping_normalization", "area"),
                "defaults_version": "stripping_visual_defaults_v3",
            }
        ).encode("utf-8")
    ).hexdigest()


def apply_stripping_style_defaults_for_preview(selected_samples: list[str]) -> None:
    reset_stripping_visual_style_defaults()
    st.session_state["stripping_style_defaults_signature"] = stripping_style_defaults_signature(selected_samples)


def save_all_stripping_selections_and_go_style(
    selected_samples: list[str],
    records_by_sample: dict[str, list[dict[str, object]]],
) -> None:
    """Save all currently visible stripping selections and advance to style preview."""
    ensure_stripping_selection_store()
    for sample in selected_samples:
        if sample in records_by_sample:
            save_stripping_selection_for_sample(sample, records_by_sample[sample])
    apply_stripping_style_defaults_for_preview(selected_samples)
    st.session_state["stripping_workflow_step"] = "2. Style preview"


def selected_stripping_paths_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> list[str]:
    ensure_stripping_selection_store()
    saved = st.session_state["stripping_saved_selection"].get(sample_name)
    selected = []
    for record in file_records:
        rel = str(record["relative_path"])
        key = stripping_file_include_key(sample_name, str(record["repeat"]), rel)
        include = bool(saved.get(rel, st.session_state.get(key, True))) if saved is not None else bool(st.session_state.get(key, True))
        if include:
            selected.append(rel)
    return selected


def stripping_summary_is_short(summary_row: dict[str, object]) -> bool:
    return "short" in str(summary_row.get("Status", "")).lower()


def read_stripping_file_uncached(
    file_path_str: str,
    sample: str,
    repeat: str,
    area: float,
    operator: str,
    normalization: str,
    step_type: str,
    valley_window: int,
    short_capacity_threshold: float,
) -> tuple[pd.DataFrame | None, dict[str, object], str | None]:
    file_info = stripping.FileInfo(path=Path(file_path_str), sample=sample, repeat=repeat)
    metadata = stripping.read_metadata(file_info.path, area=float(area), default_operator=operator)
    try:
        record_numeric, plot_df = stripping.read_record_data(
            file_info=file_info,
            metadata=metadata,
            area=float(area),
            normalization=normalization,
            step_type=step_type,
        )
        nucleation_mV, plateau_mV, overp_mV = stripping.compute_summary_metrics(
            record_numeric=record_numeric,
            metadata=metadata,
            valley_window=int(valley_window),
        )
        file_max = stripping.max_plot_capacity(plot_df)
        is_short = file_max is not None and file_max > float(short_capacity_threshold)
        if is_short:
            metadata.status = stripping.append_status(metadata.status, "short")
        plot_df.attrs["is_short"] = is_short
    except Exception as exc:
        plot_df = None
        nucleation_mV = "N/A"
        plateau_mV = "N/A"
        overp_mV = "N/A"
        metadata.status = f"record error: {exc}"
        return None, {
            "Sample": sample,
            "Repeat": repeat,
            "Nucleation (ohm)": nucleation_mV,
            "Plateau (mV)": plateau_mV,
            "Overp. (mV)": overp_mV,
            "Cap. (mAh/cm2)": metadata.areal_capacity_mAh_cm2,
            "Time": metadata.time,
            "Operator": metadata.operator,
            "File name": metadata.displayed_file_name,
            "Source path": file_path_str,
            "Status": metadata.status,
        }, str(exc)

    return plot_df, {
        "Sample": sample,
        "Repeat": repeat,
        "Nucleation (ohm)": nucleation_mV,
        "Plateau (mV)": plateau_mV,
        "Overp. (mV)": overp_mV,
        "Cap. (mAh/cm2)": metadata.areal_capacity_mAh_cm2,
        "Time": metadata.time,
        "Operator": metadata.operator,
        "File name": metadata.displayed_file_name,
        "Source path": file_path_str,
        "Status": metadata.status,
    }, None


@st.cache_data(show_spinner=False)
def cached_read_stripping_file(
    file_path_str: str,
    sample: str,
    repeat: str,
    area: float,
    operator: str,
    normalization: str,
    step_type: str,
    valley_window: int,
    short_capacity_threshold: float,
    file_size: int,
    file_mtime: float,
) -> tuple[pd.DataFrame | None, dict[str, object], str | None]:
    _ = file_size, file_mtime
    return read_stripping_file_uncached(
        file_path_str=file_path_str,
        sample=sample,
        repeat=repeat,
        area=area,
        operator=operator,
        normalization=normalization,
        step_type=step_type,
        valley_window=valley_window,
        short_capacity_threshold=short_capacity_threshold,
    )


def normalize_stripping_cache_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    for col in ["Capacity (mAh)", "Voltage (V)", "Plot x", "Plot y"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def stripping_cache_key(
    record: dict[str, object],
    area: float,
    operator: str,
    normalization: str,
    step_type: str,
    valley_window: int,
    short_capacity_threshold: float,
) -> str:
    path = Path(record["path"])
    stat = path.stat()
    return persistent_cache_key(
        {
            "kind": "stripping_plot_v1",
            "file_path": str(path),
            "relative_path": str(record.get("relative_path", path.name)),
            "sample": str(record["sample"]),
            "repeat": str(record["repeat"]),
            "area": float(area),
            "operator": operator,
            "normalization": normalization,
            "step_type": step_type,
            "valley_window": int(valley_window),
            "short_capacity_threshold": float(short_capacity_threshold),
            "file_size": int(stat.st_size),
            "file_mtime": float(stat.st_mtime),
        }
    )


def invalid_stripping_summary_row(
    record: dict[str, object],
    status: str,
) -> dict[str, object]:
    return {
        "Sample": str(record.get("sample", "")),
        "Repeat": str(record.get("repeat", "")),
        "Nucleation (ohm)": "N/A",
        "Plateau (mV)": "N/A",
        "Overp. (mV)": "N/A",
        "Cap. (mAh/cm2)": "N/A",
        "Time": "",
        "Operator": "",
        "File name": str(record.get("source_file", "")),
        "Source path": str(record.get("path", "")),
        "Status": status,
    }


def load_persistent_stripping_file(
    record: dict[str, object],
    area: float,
    operator: str,
    normalization: str,
    step_type: str,
    valley_window: int,
    short_capacity_threshold: float,
    cache_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, object], str | None]:
    cache_key = stripping_cache_key(
        record=record,
        area=float(area),
        operator=operator,
        normalization=normalization,
        step_type=step_type,
        valley_window=int(valley_window),
        short_capacity_threshold=float(short_capacity_threshold),
    )
    hit, cached_df, meta = read_persistent_dataframe_cache(cache_dir, cache_key)
    if hit:
        summary_row = meta.get("summary_row") if isinstance(meta, dict) else None
        if not isinstance(summary_row, dict):
            summary_row = invalid_stripping_summary_row(record, str(meta.get("error") or "cache error"))
        return normalize_stripping_cache_df(cached_df), summary_row, meta.get("error") if isinstance(meta, dict) else None

    plot_df, summary_row, error = read_stripping_file_uncached(
        file_path_str=str(record["path"]),
        sample=str(record["sample"]),
        repeat=str(record["repeat"]),
        area=float(area),
        operator=operator,
        normalization=normalization,
        step_type=step_type,
        valley_window=int(valley_window),
        short_capacity_threshold=float(short_capacity_threshold),
    )
    plot_df = normalize_stripping_cache_df(plot_df)
    write_persistent_dataframe_cache(
        cache_dir,
        cache_key,
        plot_df,
        error,
        extra_meta={"summary_row": summary_row},
    )
    return plot_df, summary_row, error


def load_cached_stripping_file(
    record: dict[str, object],
    area: float,
    operator: str,
    normalization: str,
    step_type: str,
    valley_window: int,
    short_capacity_threshold: float,
    persistent_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, dict[str, object], str | None]:
    if persistent_cache_dir is not None:
        return load_persistent_stripping_file(
            record=record,
            area=float(area),
            operator=operator,
            normalization=normalization,
            step_type=step_type,
            valley_window=int(valley_window),
            short_capacity_threshold=float(short_capacity_threshold),
            cache_dir=persistent_cache_dir,
        )

    path = Path(record["path"])
    stat = path.stat()
    return cached_read_stripping_file(
        file_path_str=str(path),
        sample=str(record["sample"]),
        repeat=str(record["repeat"]),
        area=float(area),
        operator=operator,
        normalization=normalization,
        step_type=step_type,
        valley_window=int(valley_window),
        short_capacity_threshold=float(short_capacity_threshold),
        file_size=int(stat.st_size),
        file_mtime=float(stat.st_mtime),
    )


def read_stripping_preview_job_worker(
    args: tuple[
        str,
        dict[str, object],
        float,
        str,
        str,
        str,
        int,
        float,
        Path | None,
    ],
) -> tuple[str, dict[str, object], pd.DataFrame | None, dict[str, object], str | None]:
    """Pickle-safe stripping preview worker for both threads and processes."""
    (
        sample,
        record,
        area,
        operator,
        normalization,
        step_type,
        valley_window,
        short_capacity_threshold,
        persistent_cache_dir,
    ) = args

    if persistent_cache_dir is None:
        plot_df, summary_row, error = read_stripping_file_uncached(
            file_path_str=str(record["path"]),
            sample=str(record["sample"]),
            repeat=str(record["repeat"]),
            area=float(area),
            operator=operator,
            normalization=normalization,
            step_type=step_type,
            valley_window=int(valley_window),
            short_capacity_threshold=float(short_capacity_threshold),
        )
    else:
        plot_df, summary_row, error = load_cached_stripping_file(
            record,
            float(area),
            operator,
            normalization,
            step_type,
            int(valley_window),
            float(short_capacity_threshold),
            persistent_cache_dir=persistent_cache_dir,
        )
    return sample, record, plot_df, summary_row, error


def stripping_x_axis_label(normalization: str) -> str:
    if normalization == "area":
        return "Capacity (mAh/cm$^2$)"
    if normalization == "test-dchg":
        return "Normalized capacity by test DChg divisor"
    return "Capacity (mAh)"


def clean_stripping_plot_df(plot_df: pd.DataFrame) -> pd.DataFrame:
    df = plot_df.copy()
    for col in ["Plot x", "Plot y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Plot x", "Plot y"])


def stripping_figure_limits(plot_df: pd.DataFrame, style: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, bool]:
    numeric_df = clean_stripping_plot_df(plot_df)
    if numeric_df.empty:
        return {
            "auto_x_range": bool(style["auto_x_range"]),
            "x_min": float(style["x_min"]),
            "x_max": float(style["x_max"]),
            "y_min": float(style["y_min"]),
            "y_max": float(style["y_max"]),
        }, numeric_df, False

    adjusted = False
    if bool(style["auto_x_range"]):
        x_min = float(style["x_min"])
        max_x = float(numeric_df["Plot x"].max())
        x_max = float(style["small_capacity_limit"]) if max_x <= float(style["small_capacity_limit"]) else float(style["large_capacity_limit"])
    else:
        x_min = float(style["x_min"])
        x_max = float(style["x_max"])
    y_min = float(style["y_min"])
    y_max = float(style["y_max"])

    visible = numeric_df["Plot x"].between(x_min, x_max)
    if not visible.any():
        x_min = min(float(style["x_min"]), float(numeric_df["Plot x"].min()))
        x_max = max(float(style["large_capacity_limit"]), float(numeric_df["Plot x"].max()))
        visible = numeric_df["Plot x"].between(x_min, x_max)
        adjusted = True
    if not (visible & numeric_df["Plot y"].between(y_min, y_max)).any():
        y_min, y_max = padded_limits(numeric_df.loc[visible, "Plot y"], -1.0, 0.2, 0.05)
        adjusted = True
    return {"auto_x_range": bool(style["auto_x_range"]), "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}, numeric_df, adjusted


def make_stripping_figure(
    plot_df: pd.DataFrame,
    title_name: str,
    sample_colors: dict[str, str],
    plot_mode: str,
    plot_title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    show_legend: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    linewidth: float,
    fig_width: float,
    fig_height: float,
    legend_position: str = "Top",
    legend_label_max_len: int = 28,
    legend_columns: int = 3,
):
    apply_common_plot_style()
    df = clean_stripping_plot_df(plot_df)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if df.empty:
        ax.text(0.5, 0.5, "No valid numeric plotting data", ha="center", va="center", transform=ax.transAxes)

    seen_labels: set[str] = set()
    for source_path, group in df.groupby("Source path", sort=True):
        group = group.sort_values("Plot x")
        sample = str(group["Sample"].iloc[0])
        repeat = str(group["Repeat"].iloc[0])
        file_stem = Path(str(source_path)).stem
        color = sample_colors.get(sample, "#4E79A7")
        if is_stripping_compare_mode(plot_mode):
            label_raw = sample
        else:
            label_raw = repeat if repeat == file_stem else f"{repeat} | {file_stem}"
        legend_label = shorten_label(label_raw, legend_label_max_len)
        if is_stripping_compare_mode(plot_mode) and label_raw in seen_labels:
            legend_label = "_nolegend_"
        seen_labels.add(label_raw)
        ax.plot(
            group["Plot x"].to_numpy(float),
            group["Plot y"].to_numpy(float),
            color=color,
            linewidth=float(linewidth),
            alpha=0.9,
            label=legend_label,
        )

    title = plot_title.replace("{sample}", title_name).replace("{repeat}", title_name)
    if title.strip():
        ax.set_title(title, fontsize=18, pad=14)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_xlabel(x_label, fontsize=17, labelpad=8)
    ax.set_ylabel(y_label, fontsize=17, labelpad=8)
    ax.tick_params(axis="both", which="major", direction="in", labelsize=14, length=6, width=1.4, pad=6)
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        n_labels = max(1, len(labels))
        ncol = max(1, min(int(legend_columns), n_labels))
        if legend_position == "Top":
            top = 0.74 if title.strip() else 0.80
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=top)
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=ncol, title=legend_title, fontsize=10, title_fontsize=12, frameon=False)
        elif legend_position == "Right":
            fig.subplots_adjust(left=0.12, right=0.72, bottom=0.15, top=0.90)
            fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.985, 0.53), ncol=1, title=legend_title, fontsize=10, title_fontsize=12, frameon=False)
        else:
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=0.90)
            ax.legend(loc="best", title=legend_title, fontsize=10, title_fontsize=12, frameon=False)
    else:
        fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, top=0.90)
    return fig


def make_single_stripping_preview_figure(plot_df: pd.DataFrame | None, color_hex: str, fig_width: float = 3.7, fig_height: float = 2.25):
    if plot_df is None or plot_df.empty:
        return make_empty_single_file_capacity_preview_figure("No valid preview", fig_width=fig_width, fig_height=fig_height)
    apply_common_plot_style()
    df = clean_stripping_plot_df(plot_df)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if not df.empty:
        ax.plot(df["Plot x"], df["Plot y"], color=color_hex, linewidth=1.5)
    ax.set_xlabel("Capacity", fontsize=9, labelpad=4)
    ax.set_ylabel("Voltage (V)", fontsize=9, labelpad=4)
    ax.tick_params(axis="both", which="major", direction="in", labelsize=8, length=4, width=1.0, pad=3)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.22, top=0.92)
    return fig


def render_stripping_file_preview_card(
    record: dict[str, object],
    plot_df: pd.DataFrame | None,
    summary_row: dict[str, object],
    error: str | None,
    checkbox_key: str,
    color_hex: str,
) -> None:
    """Render a stripping preview card using the same card shell as cycling."""
    rel = str(record["relative_path"])
    file_stem = Path(rel).stem
    repeat = str(record.get("repeat", ""))
    label = repeat if repeat == file_stem else f"{repeat} | {file_stem}"

    status = str(summary_row.get("Status", "") or "").strip()
    note_text = str(error or "")
    if not note_text and status and status != "ok":
        note_text = status

    try:
        card_ctx = st.container(border=True)
    except TypeError:
        card_ctx = st.container()

    with card_ctx:
        st.checkbox(
            shorten_label(label, 30),
            key=checkbox_key,
            help=rel,
        )
        sync_stripping_checkbox_to_saved(str(record["sample"]), rel, checkbox_key)

        fig = make_single_stripping_preview_figure(
            plot_df=plot_df,
            color_hex=color_hex,
            fig_width=3.7,
            fig_height=2.25,
        )
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        render_preview_metric_grid(
            [
                ("Nuc.", preview_metric_text(summary_row.get("Nucleation (ohm)"))),
                ("Plateau", preview_metric_text(summary_row.get("Plateau (mV)"))),
                ("Overp.", preview_metric_text(summary_row.get("Overp. (mV)"))),
                ("Cap.", preview_metric_text(summary_row.get("Cap. (mAh/cm2)"))),
            ]
        )
        render_preview_note(note_text)


def render_stripping_analysis_page() -> None:
    st.title("Stripping Overpotential")
    st.caption("Batch summary and voltage-capacity plotting for stripping-cell data.")

    with st.sidebar:
        st.header("Stripping input")
        stripping_input_mode = st.radio(
            "Data access mode",
            ["Local/server folder path", "Demo ZIP upload only"],
            index=0,
            key="stripping_input_mode",
            help=(
                "Use Local/server folder path for real datasets. "
                "ZIP upload is only for small demo datasets."
            ),
        )
        if stripping_input_mode == "Local/server folder path":
            root_dir_str = st.text_input(
                "Root data directory on this machine/server",
                value="",
                help="Folder containing one first-level folder per sample.",
                key="stripping_root_dir",
            )
            root_dir = Path(root_dir_str).expanduser().resolve() if root_dir_str.strip() else None
            output_dir_str = st.text_input(
                "Output directory",
                value="",
                help="Leave empty to save to <root_dir>/stripping_outputs.",
                key="stripping_output_dir",
            )
            if output_dir_str.strip():
                output_dir = Path(output_dir_str).expanduser().resolve()
            elif root_dir is not None:
                output_dir = root_dir / "stripping_outputs"
            else:
                output_dir = None
            input_message = "Enter a root data directory to start."
        else:
            st.warning("ZIP upload is for small demo data only. For larger datasets, use Local/server folder path.")
            uploaded_zip = st.file_uploader(
                "Upload a small demo ZIP containing sample folders",
                type=["zip"],
                accept_multiple_files=False,
                key="stripping_uploaded_zip",
            )
            if uploaded_zip is None:
                root_dir = None
                output_dir = None
                input_message = "Upload a small demo ZIP to start, or switch to Local/server folder path."
            else:
                try:
                    root_dir = get_or_create_uploaded_stripping_zip_root(uploaded_zip)
                    output_dir = root_dir / "stripping_outputs"
                    input_message = f"Temporary extracted root: `{root_dir}`"
                    st.caption(input_message)
                except Exception as exc:
                    st.error(f"Could not extract ZIP: {exc}")
                    root_dir = None
                    output_dir = None
                    input_message = "ZIP extraction failed."

        if root_dir is not None and output_dir is None:
            output_dir = root_dir / "stripping_outputs"
        if not bool(st.session_state.get("stripping_bulk_preview_default_migrated", False)):
            st.session_state["stripping_bulk_preview"] = True
            st.session_state["stripping_bulk_preview_default_migrated"] = True
        bulk_preview = st.checkbox(
            "Load all selected samples at once in data preview",
            value=True,
            key="stripping_bulk_preview",
            help="Read all selected samples in one pass, then review and select files without stepping sample-by-sample.",
        )
        stripping_parallel_load = st.checkbox(
            "Parallel file loading (experimental)",
            value=False,
            key="stripping_parallel_load",
            disabled=not bulk_preview,
            help=(
                "Experimental. Reads multiple Excel files at the same time during load-all preview. "
                "This can use much more CPU, memory, and disk/network bandwidth, and may be less stable with very large files, cloud drives, or openpyxl."
            ),
        )
        stripping_parallel_backend = st.selectbox(
            "Parallel backend",
            ["Threads", "Processes"],
            key="stripping_parallel_backend",
            disabled=not stripping_parallel_load,
            help=(
                "Threads are lighter but Python Excel parsing may not fully use all CPU cores. "
                "Processes use true multi-core parallelism, but use more memory and can stress disk/cloud storage."
            ),
        )
        stripping_parallel_workers = int(
            st.number_input(
                "Parallel workers",
                min_value=1,
                max_value=64,
                value=12,
                step=1,
                key="stripping_parallel_workers",
                disabled=not stripping_parallel_load,
                help="Maximum Excel files to parse at the same time when parallel loading is enabled. Higher values can increase CPU, RAM, and disk/cloud-drive pressure.",
            )
        )
        stripping_use_parsed_cache = st.checkbox(
            "Cache parsed Excel data",
            value=True,
            key="stripping_use_parsed_excel_cache",
            help="Store parsed per-file plot data as parquet when available, otherwise CSV. Preview, style preview, and final output reuse this cache until the Excel file or read settings change.",
        )
        stripping_notify_load_complete = st.checkbox(
            "Notify when load-all preview finishes",
            value=True,
            key="stripping_notify_load_complete",
            disabled=not bulk_preview,
            help="Shows a Streamlit toast and attempts a browser notification after the load-all data preview finishes. Browser notifications may require permission.",
        )

        st.header("Data settings")
        area = st.number_input("Electrode area (cm2)", min_value=0.0001, value=1.27, step=0.01, key="stripping_area")
        if (
            not bool(st.session_state.get("stripping_operator_default_migrated", False))
            and st.session_state.get("stripping_operator", "Vincent") == "Sravani"
        ):
            st.session_state["stripping_operator"] = "Vincent"
            st.session_state["stripping_operator_default_migrated"] = True
        operator = st.text_input("Default operator", value="Vincent", key="stripping_operator")
        normalization = st.selectbox("Capacity normalization", ["area", "test-dchg", "none"], index=0, key="stripping_normalization")
        step_type = st.text_input("Record Step Type for plotting", value="CC DChg", key="stripping_step_type")
        valley_window = st.number_input("Valley detection window", min_value=1, max_value=25, value=3, step=1, key="stripping_valley_window")
        short_capacity_threshold = st.number_input("Short capacity threshold", min_value=0.1, value=7.5, step=0.5, key="stripping_short_capacity_threshold")

    if root_dir is None or output_dir is None:
        st.info(input_message)
        return

    stripping_persistent_cache_dir = (
        parsed_excel_cache_dir(output_dir, "stripping")
        if bool(stripping_use_parsed_cache)
        else None
    )

    if not root_dir.exists() or not root_dir.is_dir():
        st.error(f"Root path is not a directory on this runtime machine: `{root_dir}`")
        return

    records = collect_stripping_file_records(root_dir, output_dir)
    if not records:
        st.warning("No valid `.xlsx` stripping files found under the root directory.")
        return

    sample_names = sorted({str(r["sample"]) for r in records})
    records_by_sample = {sample: [r for r in records if str(r["sample"]) == sample] for sample in sample_names}
    default_colors = palette_to_hex_colors("Set2 + Dark2 + tab20", len(sample_names))
    default_color_map = {sample: default_colors[i] for i, sample in enumerate(sample_names)}
    ensure_stripping_selection_store()

    st.subheader("Stripping workflow")
    workflow_options = ["1. Data preview & file selection", "2. Style preview", "3. Final output"]
    if st.session_state.get("stripping_workflow_step") not in workflow_options:
        st.session_state["stripping_workflow_step"] = workflow_options[0]
    workflow_view = st.radio("Choose workflow step", workflow_options, horizontal=True, key="stripping_workflow_step")
    selected_samples = st.multiselect("Samples to process", options=sample_names, default=sample_names, key="stripping_selected_samples")
    if not selected_samples:
        st.warning("Select at least one sample.")
        return

    for sample in selected_samples:
        saved = st.session_state["stripping_saved_selection"].get(sample, {})
        for record in records_by_sample[sample]:
            rel = str(record["relative_path"])
            key = stripping_file_include_key(sample, str(record["repeat"]), rel)
            if key not in st.session_state:
                st.session_state[key] = bool(saved.get(rel, True))

    style_defaults = {
        "stripping_plot_mode": STRIPPING_PLOT_MODE_SINGLE,
        "stripping_compare_repeat": "",
        "stripping_compare_samples": selected_samples,
        "stripping_plot_title": "{sample}",
        "stripping_x_label": stripping_x_axis_label(normalization),
        "stripping_y_label": "Voltage (V)",
        "stripping_show_legend": True,
        "stripping_legend_position": "Top",
        "stripping_legend_title": "Repeats",
        "stripping_legend_label_max_len": 28,
        "stripping_legend_columns": 3,
        "stripping_auto_x_range": True,
        "stripping_x_min": -0.5,
        "stripping_x_max": 7.5,
        "stripping_small_capacity_limit": 2.0,
        "stripping_large_capacity_limit": 7.5,
        "stripping_y_min": -1.0,
        "stripping_y_max": 0.2,
        "stripping_palette_name": "Set2 + Dark2 + tab20",
        "stripping_linewidth": 2.2,
        "stripping_fig_width": 6.0,
        "stripping_fig_height": 4.6,
        "stripping_dpi": 300,
    }
    for key, value in style_defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state["stripping_plot_mode"] = normalize_stripping_plot_mode(
        st.session_state.get("stripping_plot_mode", STRIPPING_PLOT_MODE_SINGLE)
    )
    if not isinstance(st.session_state.get("stripping_compare_samples"), list):
        st.session_state["stripping_compare_samples"] = selected_samples
    st.session_state["stripping_compare_samples"] = [
        sample for sample in st.session_state.get("stripping_compare_samples", selected_samples)
        if sample in selected_samples
    ] or list(selected_samples)

    text_style_defaults = {
        "stripping_plot_title": "{sample}",
        "stripping_x_label": stripping_x_axis_label(normalization),
        "stripping_y_label": "Voltage (V)",
        "stripping_legend_title": "Repeats",
    }
    for key, value in text_style_defaults.items():
        if not str(st.session_state.get(key, "")).strip():
            st.session_state[key] = value

    if float(st.session_state.get("stripping_x_max", 7.5)) <= float(st.session_state.get("stripping_x_min", -0.5)):
        st.session_state["stripping_x_min"] = -0.5
        st.session_state["stripping_x_max"] = 7.5
    if float(st.session_state.get("stripping_y_max", 0.2)) <= float(st.session_state.get("stripping_y_min", -1.0)):
        st.session_state["stripping_y_min"] = -1.0
        st.session_state["stripping_y_max"] = 0.2
    if float(st.session_state.get("stripping_large_capacity_limit", 7.5)) <= float(st.session_state.get("stripping_small_capacity_limit", 2.0)):
        st.session_state["stripping_small_capacity_limit"] = 2.0
        st.session_state["stripping_large_capacity_limit"] = 7.5

    def current_stripping_style() -> dict[str, object]:
        return {
            "plot_mode": normalize_stripping_plot_mode(st.session_state.get("stripping_plot_mode", STRIPPING_PLOT_MODE_SINGLE)),
            "compare_repeat": st.session_state.get("stripping_compare_repeat", ""),
            "compare_samples": list(st.session_state.get("stripping_compare_samples", selected_samples)),
            "plot_title": st.session_state.get("stripping_plot_title", "{sample}"),
            "x_label": st.session_state.get("stripping_x_label", stripping_x_axis_label(normalization)),
            "y_label": st.session_state.get("stripping_y_label", "Voltage (V)"),
            "show_legend": bool(st.session_state.get("stripping_show_legend", True)),
            "legend_position": st.session_state.get("stripping_legend_position", "Top"),
            "legend_title": st.session_state.get("stripping_legend_title", "Repeats"),
            "legend_label_max_len": int(st.session_state.get("stripping_legend_label_max_len", 28)),
            "legend_columns": int(st.session_state.get("stripping_legend_columns", 3)),
            "auto_x_range": bool(st.session_state.get("stripping_auto_x_range", True)),
            "x_min": float(st.session_state.get("stripping_x_min", -0.5)),
            "x_max": float(st.session_state.get("stripping_x_max", 7.5)),
            "small_capacity_limit": float(st.session_state.get("stripping_small_capacity_limit", 2.0)),
            "large_capacity_limit": float(st.session_state.get("stripping_large_capacity_limit", 7.5)),
            "y_min": float(st.session_state.get("stripping_y_min", -1.0)),
            "y_max": float(st.session_state.get("stripping_y_max", 0.2)),
            "palette_name": st.session_state.get("stripping_palette_name", "Set2 + Dark2 + tab20"),
            "linewidth": float(st.session_state.get("stripping_linewidth", 2.2)),
            "fig_width": float(st.session_state.get("stripping_fig_width", 6.0)),
            "fig_height": float(st.session_state.get("stripping_fig_height", 4.6)),
            "dpi": int(st.session_state.get("stripping_dpi", 300)),
        }

    def current_stripping_colors(style: dict[str, object]) -> dict[str, str]:
        colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
        palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}
        return {
            sample: st.session_state.get(f"stripping_color_{safe_filename(sample)}", palette_color_map[sample])
            for sample in selected_samples
        }

    def selected_stripping_records(sample: str) -> list[dict[str, object]]:
        selected_paths = set(selected_stripping_paths_for_sample(sample, records_by_sample[sample]))
        return [r for r in records_by_sample[sample] if str(r["relative_path"]) in selected_paths]

    def available_stripping_repeats() -> list[str]:
        repeats = set()
        for sample in selected_samples:
            for record in selected_stripping_records(sample):
                repeats.add(str(record["repeat"]))
        return sorted(repeats)

    def resolve_compare_repeat(repeat_options: list[str] | None = None, update_state: bool = True) -> str:
        options = repeat_options if repeat_options is not None else available_stripping_repeats()
        current = str(st.session_state.get("stripping_compare_repeat", "") or "")
        if current in options:
            return current
        fallback = options[0] if options else ""
        if fallback and update_state:
            st.session_state["stripping_compare_repeat"] = fallback
        return fallback

    def comparison_samples_from_style(style: dict[str, object]) -> list[str]:
        compare_samples = [
            sample for sample in list(style.get("compare_samples", selected_samples))
            if sample in selected_samples
        ]
        return compare_samples or list(selected_samples)

    def load_stripping_records_for_plot(records_to_load: list[dict[str, object]]) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        plot_frames = []
        summary_rows = []
        for record in records_to_load:
            plot_df, summary_row, _error = load_cached_stripping_file(
                record=record,
                area=float(area),
                operator=operator,
                normalization=normalization,
                step_type=step_type,
                valley_window=int(valley_window),
                short_capacity_threshold=float(short_capacity_threshold),
                persistent_cache_dir=stripping_persistent_cache_dir,
            )
            summary_rows.append(summary_row)
            if plot_df is not None and not plot_df.empty:
                plot_frames.append(plot_df)
        combined_plot = pd.concat(plot_frames, ignore_index=True) if plot_frames else None
        return combined_plot, pd.DataFrame(summary_rows)

    if workflow_view == "1. Data preview & file selection":
        st.markdown("### Data preview & file selection")
        if bulk_preview:
            st.caption("All selected samples are loaded in one pass. Short files are unchecked by default on first load.")
            all_loaded_entries: dict[str, list[tuple[dict[str, object], pd.DataFrame | None, dict[str, object], str | None]]] = {}
            total_files = sum(len(records_by_sample[sample]) for sample in selected_samples)
            all_jobs = [
                (sample, record)
                for sample in selected_samples
                for record in records_by_sample[sample]
            ]
            bulk_signature = hashlib.sha1(
                repr(
                    {
                        "root_dir": str(root_dir),
                        "selected_samples": selected_samples,
                        "files": {
                            sample: [file_record_signature(record) for record in records_by_sample[sample]]
                            for sample in selected_samples
                        },
                        "area": float(area),
                        "operator": operator,
                        "normalization": normalization,
                        "step_type": step_type,
                        "valley_window": int(valley_window),
                        "short_capacity_threshold": float(short_capacity_threshold),
                        "implementation": "stripping_bulk_preview_v1",
                    }
                ).encode("utf-8")
            ).hexdigest()
            cached_bulk = st.session_state.get("stripping_bulk_preview_cache")
            cache_is_current = bool(cached_bulk and cached_bulk.get("signature") == bulk_signature)
            loaded_from_cache = cache_is_current
            reload_col, cache_col = st.columns([1, 3])
            with reload_col:
                if st.button("Reload preview data", key="stripping_reload_bulk_preview", use_container_width=True):
                    st.session_state.pop("stripping_bulk_preview_cache", None)
                    cached_bulk = None
                    cache_is_current = False
                    loaded_from_cache = False
            with cache_col:
                st.caption("Preview data is reused while files and data settings stay unchanged.")

            sample_default_state = {
                sample: (
                    sample in st.session_state.get("stripping_saved_selection", {}),
                    f"stripping_valid_defaults_applied_{stable_key_part(sample)}",
                    bool(st.session_state.get(f"stripping_valid_defaults_applied_{stable_key_part(sample)}", False)),
                )
                for sample in selected_samples
            }

            def stripping_worker_args(job: tuple[str, dict[str, object]]):
                sample, record = job
                return (
                    sample,
                    record,
                    float(area),
                    operator,
                    normalization,
                    step_type,
                    int(valley_window),
                    float(short_capacity_threshold),
                    stripping_persistent_cache_dir,
                )

            if cache_is_current:
                all_loaded_entries = cached_bulk["all_loaded_entries"]
            else:
                completed = 0
                progress = st.progress(0)
                status = st.empty()
                if stripping_parallel_load and all_jobs:
                    max_workers = min(stripping_parallel_workers, len(all_jobs))
                    executor_cls = ProcessPoolExecutor if stripping_parallel_backend == "Processes" else ThreadPoolExecutor

                    def consume_stripping_executor(executor) -> None:
                        nonlocal completed
                        futures = [
                            executor.submit(read_stripping_preview_job_worker, stripping_worker_args(job))
                            for job in all_jobs
                        ]
                        for future in as_completed(futures):
                            sample, record, plot_df, summary_row, error = future.result()
                            completed += 1
                            status.write(f"Reading {completed}/{total_files}: {sample} / {record['source_file']}")
                            saved_selection_exists, _default_marker_key, defaults_already_applied = sample_default_state[sample]
                            key = stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                            if plot_df is None:
                                st.session_state[key] = False
                            elif not saved_selection_exists and not defaults_already_applied and stripping_summary_is_short(summary_row):
                                st.session_state[key] = False
                            all_loaded_entries.setdefault(sample, []).append((record, plot_df, summary_row, error))
                            progress.progress(completed / max(1, total_files))

                    try:
                        with executor_cls(max_workers=max_workers) as executor:
                            consume_stripping_executor(executor)
                    except Exception as exc:
                        if stripping_parallel_backend != "Processes":
                            raise
                        st.warning(f"Process backend could not start or complete: {exc}. Falling back to Threads.")
                        completed = 0
                        all_loaded_entries = {}
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            consume_stripping_executor(executor)
                else:
                    for job in all_jobs:
                        sample, record, plot_df, summary_row, error = read_stripping_preview_job_worker(stripping_worker_args(job))
                        completed += 1
                        status.write(f"Reading {completed}/{total_files}: {sample} / {record['source_file']}")
                        saved_selection_exists, _default_marker_key, defaults_already_applied = sample_default_state[sample]
                        key = stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                        if plot_df is None:
                            st.session_state[key] = False
                        elif not saved_selection_exists and not defaults_already_applied and stripping_summary_is_short(summary_row):
                            st.session_state[key] = False
                        all_loaded_entries.setdefault(sample, []).append((record, plot_df, summary_row, error))
                        progress.progress(completed / max(1, total_files))

                for sample in selected_samples:
                    saved_selection_exists, default_marker_key, defaults_already_applied = sample_default_state[sample]
                    if not saved_selection_exists and not defaults_already_applied:
                        st.session_state[default_marker_key] = True
                    order = {str(record["relative_path"]): i for i, record in enumerate(records_by_sample[sample])}
                    all_loaded_entries[sample] = sorted(
                        all_loaded_entries.get(sample, []),
                        key=lambda entry: order.get(str(entry[0]["relative_path"]), 10**9),
                    )
                st.session_state["stripping_bulk_preview_cache"] = {
                    "signature": bulk_signature,
                    "all_loaded_entries": all_loaded_entries,
                }
                status.empty()
                progress.empty()

            total_valid = sum(1 for entries in all_loaded_entries.values() for entry in entries if entry[1] is not None)
            total_short = sum(1 for entries in all_loaded_entries.values() for entry in entries if entry[1] is not None and stripping_summary_is_short(entry[2]))
            total_invalid = total_files - total_valid
            st.success(f"Loaded {total_files} files across {len(selected_samples)} samples. Valid: {total_valid}; short/default unchecked: {total_short}; unavailable: {total_invalid}.")
            if not loaded_from_cache:
                notify_load_all_complete(
                    notification_id=f"stripping_load_all_complete_{bulk_signature}",
                    title="Stripping data preview loaded",
                    body=f"Loaded {total_files} stripping files across {len(selected_samples)} samples. Valid: {total_valid}; unavailable: {total_invalid}.",
                    enabled=bool(stripping_notify_load_complete),
                )

            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button("Select all valid non-short", use_container_width=True, key="stripping_bulk_select_valid"):
                    for sample, entries in all_loaded_entries.items():
                        for record, plot_df, summary_row, _error in entries:
                            key = stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                            st.session_state[key] = plot_df is not None and not stripping_summary_is_short(summary_row)
                        save_stripping_selection_for_sample(sample, records_by_sample[sample])
                    rerun_streamlit_app()
            with b2:
                if st.button("Clear all", use_container_width=True, key="stripping_bulk_clear_all"):
                    for sample in selected_samples:
                        for record in records_by_sample[sample]:
                            st.session_state[stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))] = False
                        save_stripping_selection_for_sample(sample, records_by_sample[sample])
                    rerun_streamlit_app()
            with b3:
                selected_total = sum(len(selected_stripping_paths_for_sample(sample, records_by_sample[sample])) for sample in selected_samples)
                st.caption(f"Current selection across all samples: {selected_total} / {total_files} files included.")

            summary_frames = []
            for sample in selected_samples:
                entries = all_loaded_entries[sample]
                selected_count = sum(
                    bool(st.session_state.get(stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"])), False))
                    for record, _plot_df, _summary_row, _error in entries
                )
                with st.expander(f"{sample} ({selected_count}/{len(entries)} selected)", expanded=True):
                    file_cols = st.columns(4)
                    valid_entries = [entry for entry in entries if entry[1] is not None]
                    invalid_entries = [entry for entry in entries if entry[1] is None]
                    for idx, (record, plot_df, summary_row, error) in enumerate(valid_entries + invalid_entries):
                        key = stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                        with file_cols[idx % 4]:
                            render_stripping_file_preview_card(
                                record=record,
                                plot_df=plot_df,
                                summary_row=summary_row,
                                error=error,
                                checkbox_key=key,
                                color_hex=default_color_map[sample],
                            )
                    current_summary = pd.DataFrame([
                        summary_row for record, _plot_df, summary_row, _error in entries
                        if bool(st.session_state.get(stripping_file_include_key(sample, str(record["repeat"]), str(record["relative_path"])), False))
                    ])
                    if not current_summary.empty:
                        summary_frames.append(current_summary)
                        st.dataframe(current_summary, use_container_width=True, hide_index=True)

            if summary_frames:
                all_summary = pd.concat(summary_frames, ignore_index=True)
                st.download_button(
                    "Download selected stripping summary CSV",
                    data=all_summary.to_csv(index=False).encode("utf-8"),
                    file_name="selected_stripping_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.button(
                "Save all selections and continue to style preview",
                type="primary",
                use_container_width=True,
                on_click=save_all_stripping_selections_and_go_style,
                args=(selected_samples, records_by_sample),
            )
            return

        if st.session_state.get("stripping_inspect_sample") not in selected_samples:
            st.session_state["stripping_inspect_sample"] = selected_samples[0]
        inspect_sample = st.selectbox("Sample to inspect", options=selected_samples, key="stripping_inspect_sample")
        file_records = records_by_sample[inspect_sample]

        c1, c2, c3 = st.columns([1, 1, 2.2])
        with c1:
            if st.button("Select all valid", use_container_width=True, key="stripping_select_all_valid"):
                for record in file_records:
                    plot_df, summary_row, _error = load_cached_stripping_file(
                        record,
                        float(area),
                        operator,
                        normalization,
                        step_type,
                        int(valley_window),
                        float(short_capacity_threshold),
                        persistent_cache_dir=stripping_persistent_cache_dir,
                    )
                    st.session_state[stripping_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))] = (
                        plot_df is not None and not stripping_summary_is_short(summary_row)
                    )
                save_stripping_selection_for_sample(inspect_sample, file_records)
                rerun_streamlit_app()
        with c2:
            if st.button("Clear all", use_container_width=True, key="stripping_clear_all"):
                for record in file_records:
                    st.session_state[stripping_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))] = False
                save_stripping_selection_for_sample(inspect_sample, file_records)
                rerun_streamlit_app()
        with c3:
            selected_count = len(selected_stripping_paths_for_sample(inspect_sample, file_records))
            st.caption(f"Saved/current selection: {selected_count} / {len(file_records)} files included.")

        loaded_entries = []
        default_marker_key = f"stripping_valid_defaults_applied_{stable_key_part(inspect_sample)}"
        saved_selection_exists = inspect_sample in st.session_state.get("stripping_saved_selection", {})
        defaults_already_applied = bool(st.session_state.get(default_marker_key, False))
        progress = st.progress(0)
        status = st.empty()
        for idx, record in enumerate(file_records, start=1):
            status.write(f"Reading {idx}/{len(file_records)}: {record['source_file']}")
            plot_df, summary_row, error = load_cached_stripping_file(
                record,
                float(area),
                operator,
                normalization,
                step_type,
                int(valley_window),
                float(short_capacity_threshold),
                persistent_cache_dir=stripping_persistent_cache_dir,
            )
            key = stripping_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))
            if plot_df is None:
                st.session_state[key] = False
            elif not saved_selection_exists and not defaults_already_applied and stripping_summary_is_short(summary_row):
                st.session_state[key] = False
            loaded_entries.append((record, plot_df, summary_row, error))
            progress.progress(idx / len(file_records))
        status.empty()
        progress.empty()
        if not saved_selection_exists and not defaults_already_applied:
            st.session_state[default_marker_key] = True

        valid_entries = [entry for entry in loaded_entries if entry[1] is not None]
        invalid_entries = [entry for entry in loaded_entries if entry[1] is None]
        short_count = sum(stripping_summary_is_short(entry[2]) for entry in valid_entries)
        st.info(f"Valid: {len(valid_entries)}; short/default unchecked: {short_count}; unavailable: {len(invalid_entries)}.")
        file_cols = st.columns(4)
        for idx, (record, plot_df, summary_row, error) in enumerate(valid_entries + invalid_entries):
            rel = str(record["relative_path"])
            key = stripping_file_include_key(inspect_sample, str(record["repeat"]), rel)
            with file_cols[idx % 4]:
                render_stripping_file_preview_card(
                    record=record,
                    plot_df=plot_df,
                    summary_row=summary_row,
                    error=error,
                    checkbox_key=key,
                    color_hex=default_color_map[inspect_sample],
                )

        current_summary = pd.DataFrame([
            summary_row for record, _plot_df, summary_row, _error in loaded_entries
            if bool(st.session_state.get(stripping_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"])), False))
        ])
        st.markdown("#### Selected-file summary for this sample")
        if current_summary.empty:
            st.info("No files are currently selected for this sample.")
        else:
            st.dataframe(current_summary, use_container_width=True, hide_index=True)
            st.download_button(
                "Download this sample stripping summary CSV",
                data=current_summary.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_filename(inspect_sample)}_stripping_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()
        current_idx = selected_samples.index(inspect_sample)
        if current_idx < len(selected_samples) - 1:
            next_sample = selected_samples[current_idx + 1]
            button_label = f"Save this sample and continue to {shorten_label(next_sample, 28)}"
        else:
            button_label = "Save this sample and continue to style preview"
        st.button(
            button_label,
            type="primary",
            use_container_width=True,
            on_click=save_current_stripping_selection_and_advance,
            args=(inspect_sample, selected_samples, records_by_sample),
        )
        return

    if workflow_view == "2. Style preview":
        st.markdown("### Style preview")
        if st.session_state.get("stripping_style_defaults_signature") != stripping_style_defaults_signature(selected_samples):
            apply_stripping_style_defaults_for_preview(selected_samples)
        style_controls_col, style_preview_col = st.columns([0.9, 1.55], gap="large")
        with style_controls_col:
            compare_mode_enabled = is_stripping_compare_mode(st.session_state.get("stripping_plot_mode"))
            preview_sample = st.selectbox(
                "Preview sample",
                options=selected_samples,
                key="stripping_preview_sample",
                disabled=compare_mode_enabled,
                help="Disabled in compare mode because the preview is the single combined comparison figure.",
            )
            repeat_options = available_stripping_repeats()
            if repeat_options and st.session_state.get("stripping_compare_repeat") not in repeat_options:
                st.session_state["stripping_compare_repeat"] = repeat_options[0]
            st.selectbox(
                "Plot mode",
                [STRIPPING_PLOT_MODE_SINGLE, STRIPPING_PLOT_MODE_COMPARE],
                key="stripping_plot_mode",
                on_change=reset_stripping_visual_style_defaults,
                help="Changing mode resets visual style to the same default so only the plotted data grouping changes.",
            )
            compare_mode_enabled = is_stripping_compare_mode(st.session_state.get("stripping_plot_mode"))
            st.multiselect(
                "Samples in comparison",
                options=selected_samples,
                key="stripping_compare_samples",
                disabled=not compare_mode_enabled,
                help="In comparison mode, preview and final output combine all selected files from these samples into one figure.",
            )

            tab_text, tab_legend, tab_axes, tab_style = st.tabs(["Text", "Legend", "Axes", "Style"])
            with tab_text:
                st.text_input("Plot title", key="stripping_plot_title", help='Use "{sample}" or "{repeat}".')
                st.text_input("X-axis label", key="stripping_x_label")
                st.text_input("Y-axis label", key="stripping_y_label")
            with tab_legend:
                st.checkbox("Show legend", key="stripping_show_legend")
                show_legend = bool(st.session_state.get("stripping_show_legend", True))
                st.selectbox("Legend position", ["Top", "Right", "Inside"], key="stripping_legend_position", disabled=not show_legend)
                st.text_input("Legend title", key="stripping_legend_title", disabled=not show_legend)
                st.slider("Label length", min_value=8, max_value=80, key="stripping_legend_label_max_len", disabled=not show_legend)
                st.slider("Top legend columns", min_value=1, max_value=6, key="stripping_legend_columns", disabled=not show_legend)
            with tab_axes:
                st.checkbox("Auto X-axis upper limit", key="stripping_auto_x_range")
                auto_x = bool(st.session_state.get("stripping_auto_x_range", True))
                a1, a2 = st.columns(2)
                with a1:
                    st.number_input("X min", step=0.1, key="stripping_x_min")
                    st.number_input("X max", step=0.5, key="stripping_x_max", disabled=auto_x)
                    st.number_input("Y min", step=0.1, key="stripping_y_min")
                with a2:
                    st.number_input("Small capacity limit", step=0.5, key="stripping_small_capacity_limit", disabled=not auto_x)
                    st.number_input("Large capacity limit", step=0.5, key="stripping_large_capacity_limit", disabled=not auto_x)
                    st.number_input("Y max", step=0.1, key="stripping_y_max")
            with tab_style:
                st.selectbox("Default color palette", ["Set2 + Dark2 + tab20", "Set2", "Dark2", "tab10", "tab20", "tab20 + tab20b"], key="stripping_palette_name")
                st.slider("Line width", min_value=0.5, max_value=5.0, step=0.1, key="stripping_linewidth")
                f1, f2 = st.columns(2)
                with f1:
                    st.number_input("Figure width", min_value=3.0, max_value=20.0, key="stripping_fig_width", step=0.5)
                    st.number_input("DPI", min_value=72, max_value=600, key="stripping_dpi", step=50)
                with f2:
                    st.number_input("Figure height", min_value=3.0, max_value=15.0, key="stripping_fig_height", step=0.5)
                style = current_stripping_style()
                colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
                palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}
                with st.expander("Sample colors", expanded=False):
                    for i, sample in enumerate(selected_samples, start=1):
                        st.color_picker(compact_widget_label("Color", i, sample, max_len=18), value=st.session_state.get(f"stripping_color_{safe_filename(sample)}", palette_color_map[sample]), key=f"stripping_color_{safe_filename(sample)}")
            st.button(
                "Generate final outputs",
                type="primary",
                use_container_width=True,
                on_click=set_stripping_workflow_step,
                args=("3. Final output",),
            )

        with style_preview_col:
            st.markdown("### Live style preview")
            style = current_stripping_style()
            sample_colors = current_stripping_colors(style)
            if is_stripping_compare_mode(style["plot_mode"]):
                compare_samples = comparison_samples_from_style(style)
                preview_records = [
                    record
                    for sample in compare_samples
                    for record in selected_stripping_records(sample)
                ]
                title_name = "Selected sample comparison"
            else:
                preview_records = selected_stripping_records(preview_sample)
                title_name = preview_sample
            preview_plot_df, preview_summary = load_stripping_records_for_plot(preview_records)
            if preview_plot_df is None:
                st.warning("No valid stripping data found for this preview.")
            else:
                effective_limits, _numeric_df, adjusted = stripping_figure_limits(preview_plot_df, style)
                fig = make_stripping_figure(
                    plot_df=preview_plot_df,
                    title_name=title_name,
                    sample_colors=sample_colors,
                    plot_mode=str(style["plot_mode"]),
                    plot_title=str(style["plot_title"]),
                    x_label=str(style["x_label"]),
                    y_label=str(style["y_label"]),
                    legend_title=str(style["legend_title"]),
                    show_legend=bool(style["show_legend"]),
                    x_min=float(effective_limits["x_min"]),
                    x_max=float(effective_limits["x_max"]),
                    y_min=float(effective_limits["y_min"]),
                    y_max=float(effective_limits["y_max"]),
                    linewidth=float(style["linewidth"]),
                    fig_width=float(style["fig_width"]),
                    fig_height=float(style["fig_height"]),
                    legend_position=str(style["legend_position"]),
                    legend_label_max_len=int(style["legend_label_max_len"]),
                    legend_columns=int(style["legend_columns"]),
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                if adjusted:
                    st.caption("Axis range was expanded to keep data visible.")
                st.dataframe(preview_summary, use_container_width=True, hide_index=True)
        return

    st.markdown("### Final output")
    style = current_stripping_style()
    sample_colors = current_stripping_colors(style)
    signature = hashlib.sha1(
        repr(
            {
                "root_dir": str(root_dir),
                "output_dir": str(output_dir),
                "selected_samples": selected_samples,
                "selected_paths": {s: selected_stripping_paths_for_sample(s, records_by_sample[s]) for s in selected_samples},
                "files": {
                    sample: [file_record_signature(record) for record in records_by_sample[sample]]
                    for sample in selected_samples
                },
                "settings": [area, operator, normalization, step_type, valley_window, short_capacity_threshold],
                "style": style,
                "colors": sample_colors,
                "implementation": "stripping_compare_bulk_v2",
            }
        ).encode("utf-8")
    ).hexdigest()
    cached = st.session_state.get("stripping_final_output_cache")
    if cached and cached.get("signature") == signature:
        if st.button("Regenerate final outputs", type="primary"):
            st.session_state.pop("stripping_final_output_cache", None)
        else:
            for item in cached["rendered_outputs"]:
                safe_item_key = safe_filename(str(item["title"]))
                st.markdown(f"#### {item['title']}")
                fig = make_stripping_figure(**item["figure_kwargs"])
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                c1, c2 = st.columns(2)
                c1.download_button("CSV", data=item["csv_bytes"], file_name=item["csv_file_name"], mime="text/csv", key=f"stripping_csv_{safe_item_key}")
                c2.download_button("PNG", data=item["png_bytes"], file_name=item["png_file_name"], mime="image/png", key=f"stripping_png_{safe_item_key}")
            st.success(f"Batch stripping analysis completed. Results saved to: `{cached['output_dir']}`")
            st.dataframe(cached["summary_df"], use_container_width=True, hide_index=True)
            st.download_button("Download all stripping results ZIP", data=cached["zip_bytes"], file_name="stripping_results.zip", mime="application/zip")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "plot_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    rendered_outputs = []
    summary_frames = []
    zip_buffer = io.BytesIO()

    if is_stripping_compare_mode(style["plot_mode"]):
        compare_samples = comparison_samples_from_style(style)
        output_specs = [(
            "Selected sample comparison",
            "Selected sample comparison",
            [
                record
                for sample in compare_samples
                for record in selected_stripping_records(sample)
            ],
        )]
    else:
        output_specs = [(sample, sample, selected_stripping_records(sample)) for sample in selected_samples]

    progress = st.progress(0)
    status = st.empty()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, (title, title_name, spec_records) in enumerate(output_specs, start=1):
            status.write(f"Processing {title} ({idx}/{len(output_specs)})...")
            plot_df, summary_df = load_stripping_records_for_plot(spec_records)
            if not summary_df.empty:
                summary_frames.append(summary_df)
            if plot_df is None:
                st.warning(f"No valid stripping data found for `{title}`.")
                progress.progress(idx / len(output_specs))
                continue
            safe_name = safe_filename(title)
            csv_path = data_dir / f"{safe_name}_plot_data.csv"
            png_path = figures_dir / f"{safe_name}_stripping.png"
            summary_path = data_dir / f"{safe_name}_summary.csv"
            plot_df.to_csv(csv_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            effective_limits, numeric_df, adjusted = stripping_figure_limits(plot_df, style)
            figure_kwargs = dict(
                plot_df=plot_df,
                title_name=title_name,
                sample_colors=sample_colors,
                plot_mode=str(style["plot_mode"]),
                plot_title=str(style["plot_title"]),
                x_label=str(style["x_label"]),
                y_label=str(style["y_label"]),
                legend_title=str(style["legend_title"]),
                show_legend=bool(style["show_legend"]),
                x_min=float(effective_limits["x_min"]),
                x_max=float(effective_limits["x_max"]),
                y_min=float(effective_limits["y_min"]),
                y_max=float(effective_limits["y_max"]),
                linewidth=float(style["linewidth"]),
                fig_width=float(style["fig_width"]),
                fig_height=float(style["fig_height"]),
                legend_position=str(style["legend_position"]),
                legend_label_max_len=int(style["legend_label_max_len"]),
                legend_columns=int(style["legend_columns"]),
            )
            fig = make_stripping_figure(**figure_kwargs)
            fig.canvas.draw()
            fig.savefig(png_path, dpi=int(style["dpi"]), bbox_inches="tight")
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=int(style["dpi"]), bbox_inches="tight")
            png_buffer.seek(0)
            png_bytes = png_buffer.getvalue()
            plt.close(fig)
            csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
            summary_bytes = summary_df.to_csv(index=False).encode("utf-8")
            zipf.writestr(f"{safe_name}/{safe_name}_plot_data.csv", csv_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_summary.csv", summary_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_stripping.png", png_bytes)
            rendered_outputs.append(
                {
                    "title": title,
                    "csv_bytes": csv_bytes,
                    "png_bytes": png_bytes,
                    "csv_file_name": f"{safe_name}_plot_data.csv",
                    "png_file_name": f"{safe_name}_stripping.png",
                    "figure_kwargs": figure_kwargs,
                    "adjusted_limits": adjusted,
                    "numeric_points": len(numeric_df),
                }
            )
            progress.progress(idx / len(output_specs))

    status.empty()
    progress.empty()
    if not rendered_outputs:
        st.warning("No valid stripping results were generated.")
        return

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_df.to_csv(output_dir / "stripping_summary.csv", index=False)
    zip_buffer.seek(0)
    cache = {
        "signature": signature,
        "output_dir": str(output_dir),
        "rendered_outputs": rendered_outputs,
        "summary_df": summary_df,
        "zip_bytes": zip_buffer.getvalue(),
    }
    st.session_state["stripping_final_output_cache"] = cache
    rerun_streamlit_app()


# -----------------------------------------------------------------------------
# dQ/dV / V-Q profile batch analysis
# -----------------------------------------------------------------------------


def get_or_create_uploaded_dqdv_zip_root(uploaded_zip) -> Path:
    """Extract an uploaded dQ/dV demo ZIP once per session and return the inferred root."""
    upload_sig = hashlib.sha1(uploaded_zip.getvalue()).hexdigest()[:16]
    state_key = "dqdv_uploaded_zip_state"
    current = st.session_state.get(state_key)

    if current and current.get("signature") == upload_sig:
        root = Path(current["root_dir"])
        if root.exists():
            return root

    extract_dir = Path(tempfile.mkdtemp(prefix="battery_dqdv_zip_"))
    safe_extract_zip_to_dir(uploaded_zip, extract_dir)
    root_dir = infer_cycling_root_after_unzip(extract_dir)
    st.session_state[state_key] = {
        "signature": upload_sig,
        "extract_dir": str(extract_dir),
        "root_dir": str(root_dir),
    }
    return root_dir


def ensure_dqdv_selection_store() -> None:
    if "dqdv_saved_selection" not in st.session_state:
        st.session_state["dqdv_saved_selection"] = {}


def dqdv_file_include_key(sample: str, repeat: str, source_path: str) -> str:
    return f"dqdv_include_{stable_key_part(sample)}_{stable_key_part(repeat)}_{stable_key_part(source_path)}"


def collect_dqdv_file_records(root_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for file_path in dqdv.find_excel_files(root_dir, output_dir=output_dir):
        meta = dqdv.infer_file_meta(file_path, root_dir)
        try:
            relative_path = str(file_path.relative_to(root_dir))
        except ValueError:
            relative_path = file_path.name
        records.append(
            {
                "sample": meta.sample_name,
                "sample_folder": meta.sample_folder,
                "repeat": meta.repeat_name,
                "source_file": file_path.name,
                "relative_path": relative_path,
                "path": file_path,
            }
        )
    return sorted(records, key=lambda r: (str(r["sample"]), str(r["repeat"]), str(r["relative_path"])))


def save_dqdv_selection_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> None:
    ensure_dqdv_selection_store()
    saved: dict[str, bool] = {}
    for record in file_records:
        rel = str(record["relative_path"])
        key = dqdv_file_include_key(sample_name, str(record["repeat"]), rel)
        saved[rel] = bool(st.session_state.get(key, True))
    st.session_state["dqdv_saved_selection"][sample_name] = saved


def sync_dqdv_checkbox_to_saved(sample_name: str, relative_path: str, checkbox_key: str) -> None:
    ensure_dqdv_selection_store()
    all_saved = dict(st.session_state["dqdv_saved_selection"])
    sample_saved = dict(all_saved.get(sample_name, {}))
    sample_saved[relative_path] = bool(st.session_state.get(checkbox_key, False))
    all_saved[sample_name] = sample_saved
    st.session_state["dqdv_saved_selection"] = all_saved


def selected_dqdv_paths_for_sample(sample_name: str, file_records: list[dict[str, object]]) -> list[str]:
    ensure_dqdv_selection_store()
    saved = st.session_state["dqdv_saved_selection"].get(sample_name)
    if saved is not None:
        return [rel for rel, include in saved.items() if include]
    paths = []
    for record in file_records:
        rel = str(record["relative_path"])
        key = dqdv_file_include_key(sample_name, str(record["repeat"]), rel)
        if bool(st.session_state.get(key, True)):
            paths.append(rel)
    return paths


def set_dqdv_workflow_step(step: str) -> None:
    st.session_state["dqdv_workflow_step"] = step


def save_current_dqdv_selection_and_advance(
    current_sample: str,
    selected_samples: list[str],
    records_by_sample: dict[str, list[dict[str, object]]],
) -> None:
    ensure_dqdv_selection_store()
    save_dqdv_selection_for_sample(current_sample, records_by_sample[current_sample])
    current_idx = selected_samples.index(current_sample)
    if current_idx < len(selected_samples) - 1:
        st.session_state["dqdv_inspect_sample"] = selected_samples[current_idx + 1]
        st.session_state["dqdv_workflow_step"] = "1. Data preview & file selection"
    else:
        apply_dqdv_style_defaults_for_preview(selected_samples)
        st.session_state["dqdv_workflow_step"] = "2. Cycle & style preview"


def save_all_dqdv_selections_and_go_style(
    selected_samples: list[str],
    records_by_sample: dict[str, list[dict[str, object]]],
) -> None:
    ensure_dqdv_selection_store()
    for sample in selected_samples:
        if sample in records_by_sample:
            save_dqdv_selection_for_sample(sample, records_by_sample[sample])
    apply_dqdv_style_defaults_for_preview(selected_samples)
    st.session_state["dqdv_workflow_step"] = "2. Cycle & style preview"


def reset_dqdv_visual_style_defaults() -> None:
    defaults = {
        "dqdv_plot_title": "{sample} - {repeat}",
        "dqdv_x_label": "Capacity (mAh cm$^{-2}$)",
        "dqdv_y_label": "Voltage (V)",
        "dqdv_show_legend": True,
        "dqdv_legend_position": "Inside",
        "dqdv_legend_title": "Cycle Index",
        "dqdv_legend_label_max_len": 28,
        "dqdv_legend_columns": 4,
        "dqdv_auto_x_range": True,
        "dqdv_x_min": -0.25,
        "dqdv_x_max": 5.0,
        "dqdv_y_min": 2.5,
        "dqdv_y_max": 4.5,
        "dqdv_palette_name": "Set2 + Dark2 + tab20",
        "dqdv_linewidth": 2.1,
        "dqdv_fig_width": 8.2,
        "dqdv_fig_height": 6.2,
        "dqdv_dpi": 300,
    }
    for key, value in defaults.items():
        st.session_state[key] = value


def dqdv_style_defaults_signature(selected_samples: list[str]) -> str:
    return hashlib.sha1(
        repr({"samples": selected_samples, "defaults_version": "dqdv_visual_defaults_v1"}).encode("utf-8")
    ).hexdigest()


def apply_dqdv_style_defaults_for_preview(selected_samples: list[str]) -> None:
    reset_dqdv_visual_style_defaults()
    st.session_state["dqdv_style_defaults_signature"] = dqdv_style_defaults_signature(selected_samples)


def dqdv_record_to_meta(record: dict[str, object]) -> dqdv.FileMeta:
    return dqdv.FileMeta(
        file_path=Path(record["path"]),
        sample_folder=str(record.get("sample_folder") or record["sample"]),
        sample_name=str(record["sample"]),
        repeat_name=str(record["repeat"]),
    )


def dqdv_cache_key(
    record: dict[str, object],
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
) -> str:
    path = Path(record["path"])
    stat = path.stat()
    return persistent_cache_key(
        {
            "kind": "dqdv_processed_v1",
            "relative_path": str(record.get("relative_path", path.name)),
            "file_size": int(stat.st_size),
            "file_mtime": float(stat.st_mtime),
            "cycles_override": cycles_override,
            "cycle_start": int(cycle_start),
            "cycle_step": int(cycle_step),
            "charge_step": charge_step,
            "discharge_step": discharge_step,
            "retention_cutoff": retention_cutoff,
            "stop_at_retention_cutoff": bool(stop_at_retention_cutoff),
        }
    )


def normalize_dqdv_plot_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    for col in ["area_cm2", "cycle_index", "point_index", "capacity_mAh", "areal_capacity_mAh_cm2", "voltage_V"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def process_dqdv_record_uncached(
    record: dict[str, object],
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]:
    processed = dqdv.process_one_file(
        meta=dqdv_record_to_meta(record),
        cycles_override=cycles_override,
        cycle_start=int(cycle_start),
        cycle_step=int(cycle_step),
        charge_step=charge_step,
        discharge_step=discharge_step,
        retention_cutoff=retention_cutoff,
        stop_at_retention_cutoff=bool(stop_at_retention_cutoff),
    )
    raw_df = normalize_dqdv_plot_df(processed.raw_plot_data)
    cycle_summary = processed.cycle_summary if processed.cycle_summary is not None else pd.DataFrame()
    summary_row = processed.file_summary or {}
    if summary_row:
        summary_row["relative_path"] = str(record.get("relative_path", record.get("source_file", "")))
    error = None if processed.status in {"ok", "skipped"} else processed.note
    if raw_df is not None and raw_df.empty:
        raw_df = None
    return raw_df, cycle_summary, summary_row, error


@st.cache_data(show_spinner=False)
def cached_process_dqdv_record(
    record_payload: dict[str, object],
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
    file_size: int,
    file_mtime: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]:
    _ = file_size, file_mtime
    return process_dqdv_record_uncached(
        record=record_payload,
        cycles_override=cycles_override,
        cycle_start=cycle_start,
        cycle_step=cycle_step,
        charge_step=charge_step,
        discharge_step=discharge_step,
        retention_cutoff=retention_cutoff,
        stop_at_retention_cutoff=stop_at_retention_cutoff,
    )


def load_persistent_dqdv_record(
    record: dict[str, object],
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
    cache_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]:
    cache_key = dqdv_cache_key(
        record,
        cycles_override,
        cycle_start,
        cycle_step,
        charge_step,
        discharge_step,
        retention_cutoff,
        stop_at_retention_cutoff,
    )
    hit, cached_df, meta = read_persistent_dataframe_cache(cache_dir, cache_key)
    if hit:
        raw_df = normalize_dqdv_plot_df(cached_df)
        cycle_summary = pd.DataFrame(meta.get("cycle_summary", [])) if isinstance(meta, dict) else pd.DataFrame()
        summary_row = dict(meta.get("summary_row", {})) if isinstance(meta, dict) else {}
        error = meta.get("error") if isinstance(meta, dict) else None
        return raw_df, cycle_summary, summary_row, error

    raw_df, cycle_summary, summary_row, error = process_dqdv_record_uncached(
        record,
        cycles_override,
        cycle_start,
        cycle_step,
        charge_step,
        discharge_step,
        retention_cutoff,
        stop_at_retention_cutoff,
    )
    write_persistent_dataframe_cache(
        cache_dir,
        cache_key,
        raw_df,
        error,
        extra_meta={
            "summary_row": summary_row,
            "cycle_summary": cycle_summary.to_dict("records") if not cycle_summary.empty else [],
        },
    )
    return raw_df, cycle_summary, summary_row, error


def load_cached_dqdv_record(
    record: dict[str, object],
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
    persistent_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]:
    if persistent_cache_dir is not None:
        return load_persistent_dqdv_record(
            record,
            cycles_override,
            cycle_start,
            cycle_step,
            charge_step,
            discharge_step,
            retention_cutoff,
            stop_at_retention_cutoff,
            persistent_cache_dir,
        )

    path = Path(record["path"])
    stat = path.stat()
    payload = dict(record)
    payload["path"] = str(path)
    return cached_process_dqdv_record(
        record_payload=payload,
        cycles_override=cycles_override,
        cycle_start=cycle_start,
        cycle_step=cycle_step,
        charge_step=charge_step,
        discharge_step=discharge_step,
        retention_cutoff=retention_cutoff,
        stop_at_retention_cutoff=stop_at_retention_cutoff,
        file_size=int(stat.st_size),
        file_mtime=float(stat.st_mtime),
    )


def read_dqdv_preview_job_worker(args: tuple) -> tuple[str, dict[str, object], pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]:
    (
        sample,
        record,
        cycles_override,
        cycle_start,
        cycle_step,
        charge_step,
        discharge_step,
        retention_cutoff,
        stop_at_retention_cutoff,
        persistent_cache_dir,
    ) = args
    cache_dir = Path(persistent_cache_dir) if persistent_cache_dir else None
    raw_df, cycle_summary, summary_row, error = load_cached_dqdv_record(
        record,
        cycles_override,
        int(cycle_start),
        int(cycle_step),
        charge_step,
        discharge_step,
        retention_cutoff,
        bool(stop_at_retention_cutoff),
        persistent_cache_dir=cache_dir,
    )
    return sample, record, raw_df, cycle_summary, summary_row, error


def clean_dqdv_plot_df(plot_df: pd.DataFrame | None) -> pd.DataFrame:
    if plot_df is None:
        return pd.DataFrame()
    df = plot_df.copy()
    required_cols = ["cycle_index", "point_index", "areal_capacity_mAh_cm2", "voltage_V"]
    if df.empty or any(col not in df.columns for col in required_cols):
        return pd.DataFrame()
    for col in ["cycle_index", "point_index", "areal_capacity_mAh_cm2", "voltage_V"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required_cols)


def dqdv_figure_limits(plot_df: pd.DataFrame, style: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, bool]:
    numeric_df = clean_dqdv_plot_df(plot_df)
    if numeric_df.empty:
        return {
            "x_min": float(style["x_min"]),
            "x_max": float(style["x_max"]),
            "y_min": float(style["y_min"]),
            "y_max": float(style["y_max"]),
        }, numeric_df, False

    adjusted = False
    if bool(style["auto_x_range"]):
        x_max = float(dqdv.choose_auto_xmax(float(numeric_df["areal_capacity_mAh_cm2"].max())))
        x_min = -0.05 * x_max
    else:
        x_min = float(style["x_min"])
        x_max = float(style["x_max"])

    y_min = float(style["y_min"])
    y_max = float(style["y_max"])
    x_visible = numeric_df["areal_capacity_mAh_cm2"].between(x_min, x_max)
    if not x_visible.any():
        x_max = float(dqdv.choose_auto_xmax(float(numeric_df["areal_capacity_mAh_cm2"].max())))
        x_min = -0.05 * x_max
        x_visible = numeric_df["areal_capacity_mAh_cm2"].between(x_min, x_max)
        adjusted = True
    if not (x_visible & numeric_df["voltage_V"].between(y_min, y_max)).any():
        y_min, y_max = padded_limits(numeric_df.loc[x_visible, "voltage_V"], 2.5, 4.5, 0.15)
        adjusted = True
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}, numeric_df, adjusted


def make_dqdv_figure(
    plot_df: pd.DataFrame,
    sample_name: str,
    repeat_name: str,
    source_file: str,
    color_hex: str,
    plot_title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    show_legend: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    linewidth: float,
    fig_width: float,
    fig_height: float,
    legend_position: str = "Inside",
    legend_label_max_len: int = 28,
    legend_columns: int = 4,
):
    apply_common_plot_style()
    df = clean_dqdv_plot_df(plot_df)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if df.empty:
        ax.text(0.5, 0.5, "No valid numeric plotting data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        cycles = sorted(df["cycle_index"].dropna().astype(int).unique())
        base_color = hex_to_rgb_tuple(color_hex)
        cycle_colors = {
            cycle: dqdv.faded_color(base_color, idx, len(cycles))
            for idx, cycle in enumerate(cycles)
        }
        seen_cycles: set[int] = set()
        for (_, cycle, step_type), group in df.groupby(["source_file", "cycle_index", "step_type"], sort=True):
            group = group.sort_values("point_index")
            cycle_int = int(cycle)
            label = shorten_label(str(cycle_int), legend_label_max_len) if cycle_int not in seen_cycles else "_nolegend_"
            seen_cycles.add(cycle_int)
            ax.plot(
                group["areal_capacity_mAh_cm2"].to_numpy(float),
                group["voltage_V"].to_numpy(float),
                color=cycle_colors.get(cycle_int, base_color),
                linestyle="-",
                linewidth=float(linewidth),
                alpha=0.92,
                label=label,
            )

    title = (
        plot_title
        .replace("{sample}", sample_name)
        .replace("{repeat}", repeat_name)
        .replace("{file}", Path(source_file).stem)
    )
    if title.strip():
        ax.set_title(title, fontsize=17, pad=12)
    ax.set_xlabel(x_label, fontsize=17, labelpad=8)
    ax.set_ylabel(y_label, fontsize=17, labelpad=8)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    dqdv.apply_axis_style(ax)

    legend_position = legend_position if show_legend else "Hide"
    handles, labels = ax.get_legend_handles_labels()
    legend_pairs = [(handle, label) for handle, label in zip(handles, labels) if label != "_nolegend_"]
    handles = [handle for handle, _label in legend_pairs]
    labels = [label for _handle, label in legend_pairs]
    n_labels = max(1, len(labels))
    ncol = max(1, min(int(legend_columns), n_labels))

    if legend_position == "Top" and labels:
        top = 0.74 if title.strip() else 0.80
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=top)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=ncol,
            title=legend_title,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
    elif legend_position == "Right" and labels:
        fig.subplots_adjust(left=0.13, right=0.75, bottom=0.13, top=0.90)
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.985, 0.53), ncol=1, title=legend_title, frameon=False)
    elif legend_position == "Inside" and labels:
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.90)
        ax.legend(handles, labels, loc="best", title=legend_title, frameon=False)
    else:
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.90)
    return fig


def make_single_dqdv_preview_figure(plot_df: pd.DataFrame | None, color_hex: str, fig_width: float = 3.7, fig_height: float = 2.25):
    style = {
        "auto_x_range": True,
        "x_min": -0.25,
        "x_max": 5.0,
        "y_min": 2.5,
        "y_max": 4.5,
    }
    limits, _numeric, _adjusted = dqdv_figure_limits(plot_df if plot_df is not None else pd.DataFrame(), style)
    sample = str(plot_df["sample"].iloc[0]) if plot_df is not None and not plot_df.empty and "sample" in plot_df.columns else "Preview"
    repeat = str(plot_df["repeat"].iloc[0]) if plot_df is not None and not plot_df.empty and "repeat" in plot_df.columns else ""
    source = str(plot_df["source_file"].iloc[0]) if plot_df is not None and not plot_df.empty and "source_file" in plot_df.columns else ""
    return make_dqdv_figure(
        plot_df=plot_df if plot_df is not None else pd.DataFrame(),
        sample_name=sample,
        repeat_name=repeat,
        source_file=source,
        color_hex=color_hex,
        plot_title="",
        x_label="Capacity",
        y_label="Voltage",
        legend_title="Cycle",
        show_legend=False,
        x_min=float(limits["x_min"]),
        x_max=float(limits["x_max"]),
        y_min=float(limits["y_min"]),
        y_max=float(limits["y_max"]),
        linewidth=1.6,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def render_dqdv_file_preview_card(
    record: dict[str, object],
    plot_df: pd.DataFrame | None,
    summary_row: dict[str, object],
    error: str | None,
    checkbox_key: str,
    color_hex: str,
) -> None:
    rel = str(record["relative_path"])
    file_stem = Path(rel).stem
    repeat = str(record.get("repeat", ""))
    label = repeat if repeat == file_stem else f"{repeat} | {file_stem}"
    status_text = str(summary_row.get("status", "") or "").strip()
    note_text = str(error or summary_row.get("note", "") or "")
    if not note_text and status_text and status_text != "ok":
        note_text = status_text

    try:
        card_ctx = st.container(border=True)
    except TypeError:
        card_ctx = st.container()

    with card_ctx:
        st.checkbox(shorten_label(label, 30), key=checkbox_key, help=rel)
        sync_dqdv_checkbox_to_saved(str(record["sample"]), rel, checkbox_key)
        fig = make_single_dqdv_preview_figure(plot_df, color_hex=color_hex)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        render_preview_metric_grid(
            [
                ("Max cyc", preview_metric_text(summary_row.get("max_record_cycle"), digits=0)),
                ("Plotted", preview_metric_text(summary_row.get("n_cycles_included_in_plot"), digits=0)),
                ("First", preview_metric_text(summary_row.get("first_plotted_cycle"), digits=0)),
                ("Last", preview_metric_text(summary_row.get("last_plotted_cycle"), digits=0)),
            ]
        )
        render_preview_note(note_text if status_text != "ok" else "")


def parse_dqdv_cycles_from_state() -> tuple[list[int] | None, str | None]:
    mode = st.session_state.get("dqdv_cycle_mode", "Interval")
    if mode == "Custom list":
        raw = st.session_state.get("dqdv_cycle_list", "")
        try:
            return dqdv.parse_cycle_list(raw), None
        except Exception as exc:
            return None, f"Could not parse cycle list: {exc}"
    return None, None


def render_dqdv_analysis_page() -> None:
    st.title("dQ/dV Analysis")
    st.caption("Batch cycle selection and per-repeat V-Q profile plotting using `dqdv_batch.py`.")

    with st.sidebar:
        st.header("dQ/dV input")
        input_mode = st.radio(
            "Data access mode",
            ["Local/server folder path", "Demo ZIP upload only"],
            index=0,
            key="dqdv_input_mode",
            help="Use Local/server folder path for real datasets. ZIP upload is only for small demo datasets.",
        )
        if input_mode == "Local/server folder path":
            root_dir_str = st.text_input("Root data directory on this machine/server", value="", help="Folder containing one first-level folder per sample.", key="dqdv_root_dir")
            root_dir = Path(root_dir_str).expanduser().resolve() if root_dir_str.strip() else None
            output_dir_str = st.text_input("Output directory", value="", help="Leave empty to save to <root_dir>/dqdv_analysis_outputs.", key="dqdv_output_dir")
            if output_dir_str.strip():
                output_dir = Path(output_dir_str).expanduser().resolve()
            elif root_dir is not None:
                output_dir = root_dir / "dqdv_analysis_outputs"
            else:
                output_dir = None
            input_message = "Enter a root data directory to start."
        else:
            st.warning("ZIP upload is for small demo data only. For larger datasets, use Local/server folder path.")
            uploaded_zip = st.file_uploader("Upload a small demo ZIP containing sample folders", type=["zip"], accept_multiple_files=False, key="dqdv_uploaded_zip")
            if uploaded_zip is None:
                root_dir = None
                output_dir = None
                input_message = "Upload a small demo ZIP to start, or switch to Local/server folder path."
            else:
                try:
                    root_dir = get_or_create_uploaded_dqdv_zip_root(uploaded_zip)
                    output_dir = root_dir / "dqdv_analysis_outputs"
                    input_message = f"Temporary extracted root: `{root_dir}`"
                    st.caption(input_message)
                except Exception as exc:
                    st.error(f"Could not extract ZIP: {exc}")
                    root_dir = None
                    output_dir = None
                    input_message = "ZIP extraction failed."

        if root_dir is not None and output_dir is None:
            output_dir = root_dir / "dqdv_analysis_outputs"

        bulk_preview = st.checkbox("Load all selected samples at once in data preview", value=True, key="dqdv_bulk_preview")
        parallel_load = st.checkbox(
            "Parallel file loading (experimental)",
            value=False,
            key="dqdv_parallel_load",
            disabled=not bulk_preview,
            help="Reads multiple Excel files at the same time during load-all preview.",
        )
        parallel_backend = st.selectbox("Parallel backend", ["Threads", "Processes"], key="dqdv_parallel_backend", disabled=not parallel_load)
        parallel_workers = int(st.number_input("Parallel workers", min_value=1, max_value=64, value=12, step=1, key="dqdv_parallel_workers", disabled=not parallel_load))
        use_parsed_cache = st.checkbox("Cache parsed Excel data", value=True, key="dqdv_use_parsed_excel_cache")
        notify_load_complete = st.checkbox("Notify when load-all preview finishes", value=True, key="dqdv_notify_load_complete", disabled=not bulk_preview)

        st.header("Cycle extraction")
        cycle_start = int(st.number_input("Initial cycle index", min_value=0, value=3, step=1, key="dqdv_cycle_start", help="Also used as the initial discharge capacity reference."))
        st.selectbox("Cycles to plot", ["Interval", "Custom list"], key="dqdv_cycle_mode")
        if st.session_state.get("dqdv_cycle_mode", "Interval") == "Interval":
            cycle_step = int(st.number_input("Cycle interval", min_value=1, value=20, step=1, key="dqdv_cycle_step"))
        else:
            cycle_step = int(st.number_input("Fallback interval", min_value=1, value=20, step=1, key="dqdv_cycle_step"))
            st.text_input("Custom cycle list", value="3,23,43,63", key="dqdv_cycle_list", help="Comma-separated cycle indices. This overrides the interval.")
        charge_step = st.text_input("Charge Step Type", value="CC Chg", key="dqdv_charge_step")
        discharge_step = st.text_input("Discharge Step Type", value="CC DChg", key="dqdv_discharge_step")
        use_retention_cutoff = st.checkbox("Apply retention cutoff", value=True, key="dqdv_use_retention_cutoff")
        retention_cutoff = st.number_input("Retention cutoff (%)", min_value=0.0, max_value=200.0, value=80.0, step=1.0, key="dqdv_retention_cutoff", disabled=not use_retention_cutoff)
        stop_at_retention_cutoff = st.checkbox("Stop checking after cutoff is reached", value=True, key="dqdv_stop_at_retention_cutoff", disabled=not use_retention_cutoff)

    if root_dir is None or output_dir is None:
        st.info(input_message)
        return
    if not root_dir.exists() or not root_dir.is_dir():
        st.error(f"Root path is not a directory on this runtime machine: `{root_dir}`")
        return

    cycles_override, cycle_error = parse_dqdv_cycles_from_state()
    if cycle_error:
        st.error(cycle_error)
        return
    effective_retention_cutoff = float(retention_cutoff) if use_retention_cutoff else None
    dqdv_persistent_cache_dir = parsed_excel_cache_dir(output_dir, "dqdv") if bool(use_parsed_cache) else None

    records = collect_dqdv_file_records(root_dir, output_dir)
    if not records:
        st.warning("No valid `.xlsx` dQ/dV files found under the root directory.")
        return

    sample_names = sorted({str(r["sample"]) for r in records})
    records_by_sample = {sample: [r for r in records if str(r["sample"]) == sample] for sample in sample_names}
    default_colors = palette_to_hex_colors("Set2 + Dark2 + tab20", len(sample_names))
    default_color_map = {sample: default_colors[i] for i, sample in enumerate(sample_names)}
    ensure_dqdv_selection_store()

    st.subheader("dQ/dV workflow")
    workflow_options = ["1. Data preview & file selection", "2. Cycle & style preview", "3. Final output"]
    if st.session_state.get("dqdv_workflow_step") not in workflow_options:
        st.session_state["dqdv_workflow_step"] = workflow_options[0]
    workflow_view = st.radio("Choose workflow step", workflow_options, horizontal=True, key="dqdv_workflow_step")
    selected_samples = st.multiselect("Samples to process", options=sample_names, default=sample_names, key="dqdv_selected_samples")
    if not selected_samples:
        st.warning("Select at least one sample.")
        return

    for sample in selected_samples:
        saved = st.session_state["dqdv_saved_selection"].get(sample, {})
        for record in records_by_sample[sample]:
            rel = str(record["relative_path"])
            key = dqdv_file_include_key(sample, str(record["repeat"]), rel)
            if key not in st.session_state:
                st.session_state[key] = bool(saved.get(rel, True))

    style_defaults = {
        "dqdv_plot_title": "{sample} - {repeat}",
        "dqdv_x_label": "Capacity (mAh cm$^{-2}$)",
        "dqdv_y_label": "Voltage (V)",
        "dqdv_show_legend": True,
        "dqdv_legend_position": "Inside",
        "dqdv_legend_title": "Cycle Index",
        "dqdv_legend_label_max_len": 28,
        "dqdv_legend_columns": 4,
        "dqdv_auto_x_range": True,
        "dqdv_x_min": -0.25,
        "dqdv_x_max": 5.0,
        "dqdv_y_min": 2.5,
        "dqdv_y_max": 4.5,
        "dqdv_palette_name": "Set2 + Dark2 + tab20",
        "dqdv_linewidth": 2.1,
        "dqdv_fig_width": 8.2,
        "dqdv_fig_height": 6.2,
        "dqdv_dpi": 300,
    }
    for key, value in style_defaults.items():
        st.session_state.setdefault(key, value)
    if float(st.session_state.get("dqdv_x_max", 5.0)) <= float(st.session_state.get("dqdv_x_min", -0.25)):
        st.session_state["dqdv_x_min"] = -0.25
        st.session_state["dqdv_x_max"] = 5.0
    if float(st.session_state.get("dqdv_y_max", 4.5)) <= float(st.session_state.get("dqdv_y_min", 2.5)):
        st.session_state["dqdv_y_min"] = 2.5
        st.session_state["dqdv_y_max"] = 4.5

    def current_dqdv_style() -> dict[str, object]:
        return {
            "plot_title": st.session_state.get("dqdv_plot_title", "{sample} - {repeat}"),
            "x_label": st.session_state.get("dqdv_x_label", "Capacity (mAh cm$^{-2}$)"),
            "y_label": st.session_state.get("dqdv_y_label", "Voltage (V)"),
            "show_legend": bool(st.session_state.get("dqdv_show_legend", True)),
            "legend_position": st.session_state.get("dqdv_legend_position", "Inside"),
            "legend_title": st.session_state.get("dqdv_legend_title", "Cycle Index"),
            "legend_label_max_len": int(st.session_state.get("dqdv_legend_label_max_len", 28)),
            "legend_columns": int(st.session_state.get("dqdv_legend_columns", 4)),
            "auto_x_range": bool(st.session_state.get("dqdv_auto_x_range", True)),
            "x_min": float(st.session_state.get("dqdv_x_min", -0.25)),
            "x_max": float(st.session_state.get("dqdv_x_max", 5.0)),
            "y_min": float(st.session_state.get("dqdv_y_min", 2.5)),
            "y_max": float(st.session_state.get("dqdv_y_max", 4.5)),
            "palette_name": st.session_state.get("dqdv_palette_name", "Set2 + Dark2 + tab20"),
            "linewidth": float(st.session_state.get("dqdv_linewidth", 2.1)),
            "fig_width": float(st.session_state.get("dqdv_fig_width", 8.2)),
            "fig_height": float(st.session_state.get("dqdv_fig_height", 6.2)),
            "dpi": int(st.session_state.get("dqdv_dpi", 300)),
        }

    def current_dqdv_colors(style: dict[str, object]) -> dict[str, str]:
        colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
        palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}
        return {
            sample: st.session_state.get(f"dqdv_color_{safe_filename(sample)}", palette_color_map[sample])
            for sample in selected_samples
        }

    def selected_dqdv_records(sample: str) -> list[dict[str, object]]:
        selected_paths = set(selected_dqdv_paths_for_sample(sample, records_by_sample[sample]))
        return [r for r in records_by_sample[sample] if str(r["relative_path"]) in selected_paths]

    def load_dqdv_records_for_plot(records_to_load: list[dict[str, object]]) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
        plot_frames = []
        summary_rows = []
        cycle_frames = []
        for record in records_to_load:
            plot_df, cycle_summary, summary_row, _error = load_cached_dqdv_record(
                record=record,
                cycles_override=cycles_override,
                cycle_start=int(cycle_start),
                cycle_step=int(cycle_step),
                charge_step=charge_step,
                discharge_step=discharge_step,
                retention_cutoff=effective_retention_cutoff,
                stop_at_retention_cutoff=bool(stop_at_retention_cutoff),
                persistent_cache_dir=dqdv_persistent_cache_dir,
            )
            summary_rows.append(summary_row)
            if cycle_summary is not None and not cycle_summary.empty:
                cycle_frames.append(cycle_summary)
            if plot_df is not None and not plot_df.empty:
                plot_frames.append(plot_df)
        combined_plot = pd.concat(plot_frames, ignore_index=True) if plot_frames else None
        combined_cycles = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
        return combined_plot, pd.DataFrame(summary_rows), combined_cycles

    if workflow_view == "1. Data preview & file selection":
        st.markdown("### Data preview & file selection")
        if cycles_override:
            st.caption(f"Custom cycles: {', '.join(str(c) for c in cycles_override)}")
        else:
            st.caption(f"Cycle interval: start {int(cycle_start)}, every {int(cycle_step)} cycles.")

        if bulk_preview:
            all_loaded_entries: dict[str, list[tuple[dict[str, object], pd.DataFrame | None, pd.DataFrame, dict[str, object], str | None]]] = {}
            all_jobs = [(sample, record) for sample in selected_samples for record in records_by_sample[sample]]
            bulk_signature = hashlib.sha1(
                repr(
                    {
                        "root_dir": str(root_dir),
                        "selected_samples": selected_samples,
                        "files": {sample: [file_record_signature(record) for record in records_by_sample[sample]] for sample in selected_samples},
                        "cycles_override": cycles_override,
                        "cycle_start": int(cycle_start),
                        "cycle_step": int(cycle_step),
                        "charge_step": charge_step,
                        "discharge_step": discharge_step,
                        "retention_cutoff": effective_retention_cutoff,
                        "stop_at_retention_cutoff": bool(stop_at_retention_cutoff),
                        "implementation": "dqdv_bulk_preview_v1",
                    }
                ).encode("utf-8")
            ).hexdigest()
            cached_bulk = st.session_state.get("dqdv_bulk_preview_cache")
            cache_is_current = bool(cached_bulk and cached_bulk.get("signature") == bulk_signature)
            reload_col, cache_col = st.columns([1, 3])
            with reload_col:
                if st.button("Reload preview data", key="dqdv_reload_bulk_preview", use_container_width=True):
                    st.session_state.pop("dqdv_bulk_preview_cache", None)
                    rerun_streamlit_app()
            with cache_col:
                if cache_is_current:
                    st.caption("Using the current in-session preview cache.")

            if cache_is_current:
                all_loaded_entries = cached_bulk["entries"]
            else:
                progress = st.progress(0)
                status = st.empty()
                for sample in selected_samples:
                    all_loaded_entries[sample] = []

                def dqdv_worker_args(job: tuple[str, dict[str, object]]):
                    sample, record = job
                    return (
                        sample,
                        record,
                        cycles_override,
                        int(cycle_start),
                        int(cycle_step),
                        charge_step,
                        discharge_step,
                        effective_retention_cutoff,
                        bool(stop_at_retention_cutoff),
                        dqdv_persistent_cache_dir,
                    )

                if parallel_load and all_jobs:
                    max_workers = min(int(parallel_workers), len(all_jobs))
                    executor_cls = ProcessPoolExecutor if parallel_backend == "Processes" else ThreadPoolExecutor

                    def consume_dqdv_executor(executor) -> None:
                        futures = {
                            executor.submit(read_dqdv_preview_job_worker, dqdv_worker_args(job)): job
                            for job in all_jobs
                        }
                        for done_idx, future in enumerate(as_completed(futures), start=1):
                            sample, record, plot_df, cycle_summary, summary_row, error = future.result()
                            key = dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                            if plot_df is None:
                                st.session_state[key] = False
                            all_loaded_entries[sample].append((record, plot_df, cycle_summary, summary_row, error))
                            status.write(f"Loaded {done_idx}/{len(all_jobs)}: {record['source_file']}")
                            progress.progress(done_idx / len(all_jobs))

                    try:
                        with executor_cls(max_workers=max_workers) as executor:
                            consume_dqdv_executor(executor)
                    except Exception as exc:
                        if parallel_backend != "Processes":
                            raise
                        st.warning(f"Process backend could not start or complete: {exc}. Falling back to Threads.")
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            consume_dqdv_executor(executor)
                else:
                    for idx, job in enumerate(all_jobs, start=1):
                        sample, record, plot_df, cycle_summary, summary_row, error = read_dqdv_preview_job_worker(dqdv_worker_args(job))
                        key = dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                        if plot_df is None:
                            st.session_state[key] = False
                        all_loaded_entries[sample].append((record, plot_df, cycle_summary, summary_row, error))
                        status.write(f"Reading {idx}/{len(all_jobs)}: {record['source_file']}")
                        progress.progress(idx / len(all_jobs))
                status.empty()
                progress.empty()
                st.session_state["dqdv_bulk_preview_cache"] = {"signature": bulk_signature, "entries": all_loaded_entries}

            total_files = sum(len(entries) for entries in all_loaded_entries.values())
            total_valid = sum(1 for entries in all_loaded_entries.values() for entry in entries if entry[1] is not None)
            total_invalid = total_files - total_valid
            st.success(f"Loaded {total_files} files across {len(selected_samples)} samples. Valid: {total_valid}; unavailable: {total_invalid}.")
            if notify_load_complete:
                notify_load_all_complete(
                    notification_id=f"dqdv_load_all_complete_{bulk_signature}",
                    title="dQ/dV data preview loaded",
                    body=f"Loaded {total_files} dQ/dV files across {len(selected_samples)} samples.",
                    enabled=True,
                )
            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button("Select all valid", use_container_width=True, key="dqdv_bulk_select_valid"):
                    for sample, entries in all_loaded_entries.items():
                        for record, plot_df, _cycle_summary, _summary_row, _error in entries:
                            st.session_state[dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))] = plot_df is not None
                        save_dqdv_selection_for_sample(sample, records_by_sample[sample])
                    rerun_streamlit_app()
            with b2:
                if st.button("Clear all", use_container_width=True, key="dqdv_bulk_clear_all"):
                    for sample in selected_samples:
                        for record in records_by_sample[sample]:
                            st.session_state[dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))] = False
                        save_dqdv_selection_for_sample(sample, records_by_sample[sample])
                    rerun_streamlit_app()
            with b3:
                selected_total = sum(len(selected_dqdv_paths_for_sample(sample, records_by_sample[sample])) for sample in selected_samples)
                st.caption(f"Current selection across all samples: {selected_total} / {total_files} files included.")

            summary_frames = []
            for sample in selected_samples:
                entries = all_loaded_entries.get(sample, [])
                selected_count = sum(bool(st.session_state.get(dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"])), False)) for record, *_ in entries)
                with st.expander(f"{sample} ({selected_count}/{len(entries)} selected)", expanded=True):
                    file_cols = st.columns(4)
                    for idx, (record, plot_df, _cycle_summary, summary_row, error) in enumerate(entries):
                        key = dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"]))
                        with file_cols[idx % 4]:
                            render_dqdv_file_preview_card(record, plot_df, summary_row, error, key, default_color_map[sample])
                    selected_rows = [
                        summary_row for record, _plot_df, _cycle_summary, summary_row, _error in entries
                        if bool(st.session_state.get(dqdv_file_include_key(sample, str(record["repeat"]), str(record["relative_path"])), False))
                    ]
                    if selected_rows:
                        current_summary = pd.DataFrame(selected_rows)
                        summary_frames.append(current_summary)
                        st.dataframe(current_summary, use_container_width=True, hide_index=True)
            if summary_frames:
                selected_summary = pd.concat(summary_frames, ignore_index=True)
                st.download_button("Download selected dQ/dV summary CSV", data=selected_summary.to_csv(index=False).encode("utf-8"), file_name="selected_dqdv_summary.csv", mime="text/csv", use_container_width=True)
            st.button("Save selections and continue to cycle/style preview", type="primary", use_container_width=True, on_click=save_all_dqdv_selections_and_go_style, args=(selected_samples, records_by_sample))
            return

        if st.session_state.get("dqdv_inspect_sample") not in selected_samples:
            st.session_state["dqdv_inspect_sample"] = selected_samples[0]
        inspect_sample = st.selectbox("Sample to inspect", options=selected_samples, key="dqdv_inspect_sample")
        file_records = records_by_sample[inspect_sample]
        c1, c2, c3 = st.columns([1, 1, 2.2])
        with c1:
            if st.button("Select all valid", use_container_width=True, key="dqdv_select_all_valid"):
                for record in file_records:
                    plot_df, *_ = load_cached_dqdv_record(record, cycles_override, int(cycle_start), int(cycle_step), charge_step, discharge_step, effective_retention_cutoff, bool(stop_at_retention_cutoff), persistent_cache_dir=dqdv_persistent_cache_dir)
                    st.session_state[dqdv_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))] = plot_df is not None
                save_dqdv_selection_for_sample(inspect_sample, file_records)
                rerun_streamlit_app()
        with c2:
            if st.button("Clear all", use_container_width=True, key="dqdv_clear_all"):
                for record in file_records:
                    st.session_state[dqdv_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))] = False
                save_dqdv_selection_for_sample(inspect_sample, file_records)
                rerun_streamlit_app()
        with c3:
            selected_count = len(selected_dqdv_paths_for_sample(inspect_sample, file_records))
            st.caption(f"Saved/current selection: {selected_count} / {len(file_records)} files included.")

        loaded_entries = []
        progress = st.progress(0)
        status = st.empty()
        for idx, record in enumerate(file_records, start=1):
            status.write(f"Reading {idx}/{len(file_records)}: {record['source_file']}")
            plot_df, cycle_summary, summary_row, error = load_cached_dqdv_record(record, cycles_override, int(cycle_start), int(cycle_step), charge_step, discharge_step, effective_retention_cutoff, bool(stop_at_retention_cutoff), persistent_cache_dir=dqdv_persistent_cache_dir)
            key = dqdv_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"]))
            if plot_df is None:
                st.session_state[key] = False
            loaded_entries.append((record, plot_df, cycle_summary, summary_row, error))
            progress.progress(idx / len(file_records))
        status.empty()
        progress.empty()

        valid_entries = [entry for entry in loaded_entries if entry[1] is not None]
        invalid_entries = [entry for entry in loaded_entries if entry[1] is None]
        st.info(f"Valid: {len(valid_entries)}; unavailable: {len(invalid_entries)}.")
        file_cols = st.columns(4)
        for idx, (record, plot_df, _cycle_summary, summary_row, error) in enumerate(valid_entries + invalid_entries):
            rel = str(record["relative_path"])
            key = dqdv_file_include_key(inspect_sample, str(record["repeat"]), rel)
            with file_cols[idx % 4]:
                render_dqdv_file_preview_card(record, plot_df, summary_row, error, key, default_color_map[inspect_sample])

        current_summary = pd.DataFrame([
            summary_row for record, _plot_df, _cycle_summary, summary_row, _error in loaded_entries
            if bool(st.session_state.get(dqdv_file_include_key(inspect_sample, str(record["repeat"]), str(record["relative_path"])), False))
        ])
        st.markdown("#### Selected-file summary for this sample")
        if current_summary.empty:
            st.info("No files are currently selected for this sample.")
        else:
            st.dataframe(current_summary, use_container_width=True, hide_index=True)
            st.download_button("Download this sample dQ/dV summary CSV", data=current_summary.to_csv(index=False).encode("utf-8"), file_name=f"{safe_filename(inspect_sample)}_dqdv_summary.csv", mime="text/csv", use_container_width=True)

        st.divider()
        current_idx = selected_samples.index(inspect_sample)
        button_label = f"Save this sample and continue to {shorten_label(selected_samples[current_idx + 1], 28)}" if current_idx < len(selected_samples) - 1 else "Save this sample and continue to cycle/style preview"
        st.button(button_label, type="primary", use_container_width=True, on_click=save_current_dqdv_selection_and_advance, args=(inspect_sample, selected_samples, records_by_sample))
        return

    if workflow_view == "2. Cycle & style preview":
        st.markdown("### Cycle & style preview")
        if st.session_state.get("dqdv_style_defaults_signature") != dqdv_style_defaults_signature(selected_samples):
            apply_dqdv_style_defaults_for_preview(selected_samples)
        style_controls_col, style_preview_col = st.columns([0.9, 1.55], gap="large")
        with style_controls_col:
            preview_sample = st.selectbox("Preview sample", options=selected_samples, key="dqdv_preview_sample")
            preview_records_for_sample = selected_dqdv_records(preview_sample)
            preview_options = [
                str(record["relative_path"]) for record in preview_records_for_sample
            ]
            preview_rel = st.selectbox("Preview file/repeat", options=preview_options, key="dqdv_preview_relative_path") if preview_options else None

            tab_text, tab_legend, tab_axes, tab_style = st.tabs(["Text", "Legend", "Axes", "Style"])
            with tab_text:
                st.text_input("Plot title", key="dqdv_plot_title", help='Use "{sample}", "{repeat}", or "{file}".')
                st.text_input("X-axis label", key="dqdv_x_label")
                st.text_input("Y-axis label", key="dqdv_y_label")
            with tab_legend:
                st.checkbox("Show legend", key="dqdv_show_legend")
                show_legend = bool(st.session_state.get("dqdv_show_legend", True))
                st.selectbox("Legend position", ["Top", "Right", "Inside"], key="dqdv_legend_position", disabled=not show_legend)
                st.text_input("Legend title", key="dqdv_legend_title", disabled=not show_legend)
                st.slider("Label length", min_value=8, max_value=80, key="dqdv_legend_label_max_len", disabled=not show_legend)
                st.slider("Top legend columns", min_value=1, max_value=8, key="dqdv_legend_columns", disabled=not show_legend)
            with tab_axes:
                st.checkbox("Auto X-axis upper limit", key="dqdv_auto_x_range")
                auto_x = bool(st.session_state.get("dqdv_auto_x_range", True))
                a1, a2 = st.columns(2)
                with a1:
                    st.number_input("X min", step=0.1, key="dqdv_x_min", disabled=auto_x)
                    st.number_input("Y min", step=0.1, key="dqdv_y_min")
                with a2:
                    st.number_input("X max", step=0.5, key="dqdv_x_max", disabled=auto_x)
                    st.number_input("Y max", step=0.1, key="dqdv_y_max")
            with tab_style:
                st.selectbox("Default color palette", ["Set2 + Dark2 + tab20", "Set2", "Dark2", "tab10", "tab20", "tab20 + tab20b"], key="dqdv_palette_name")
                st.slider("Line width", min_value=0.5, max_value=5.0, step=0.1, key="dqdv_linewidth")
                f1, f2 = st.columns(2)
                with f1:
                    st.number_input("Figure width", min_value=3.0, max_value=20.0, key="dqdv_fig_width", step=0.5)
                    st.number_input("DPI", min_value=72, max_value=600, key="dqdv_dpi", step=50)
                with f2:
                    st.number_input("Figure height", min_value=3.0, max_value=15.0, key="dqdv_fig_height", step=0.5)
                style = current_dqdv_style()
                colors = palette_to_hex_colors(str(style["palette_name"]), len(sample_names))
                palette_color_map = {sample: colors[i] for i, sample in enumerate(sample_names)}
                with st.expander("Sample colors", expanded=False):
                    for i, sample in enumerate(selected_samples, start=1):
                        st.color_picker(compact_widget_label("Color", i, sample, max_len=18), value=st.session_state.get(f"dqdv_color_{safe_filename(sample)}", palette_color_map[sample]), key=f"dqdv_color_{safe_filename(sample)}")
            st.button("Generate final outputs", type="primary", use_container_width=True, on_click=set_dqdv_workflow_step, args=("3. Final output",))

        with style_preview_col:
            st.markdown("### Live style preview")
            style = current_dqdv_style()
            sample_colors = current_dqdv_colors(style)
            preview_record = next((record for record in preview_records_for_sample if str(record["relative_path"]) == str(preview_rel)), None)
            if preview_record is None:
                st.warning("No selected dQ/dV file found for this preview.")
            else:
                preview_plot_df, preview_summary, preview_cycles = load_dqdv_records_for_plot([preview_record])
                if preview_plot_df is None:
                    st.warning("No valid dQ/dV data found for this preview.")
                    st.dataframe(preview_summary, use_container_width=True, hide_index=True)
                else:
                    effective_limits, _numeric_df, adjusted = dqdv_figure_limits(preview_plot_df, style)
                    fig = make_dqdv_figure(
                        plot_df=preview_plot_df,
                        sample_name=str(preview_record["sample"]),
                        repeat_name=str(preview_record["repeat"]),
                        source_file=str(preview_record["source_file"]),
                        color_hex=sample_colors.get(str(preview_record["sample"]), "#4E79A7"),
                        plot_title=str(style["plot_title"]),
                        x_label=str(style["x_label"]),
                        y_label=str(style["y_label"]),
                        legend_title=str(style["legend_title"]),
                        show_legend=bool(style["show_legend"]),
                        x_min=float(effective_limits["x_min"]),
                        x_max=float(effective_limits["x_max"]),
                        y_min=float(effective_limits["y_min"]),
                        y_max=float(effective_limits["y_max"]),
                        linewidth=float(style["linewidth"]),
                        fig_width=float(style["fig_width"]),
                        fig_height=float(style["fig_height"]),
                        legend_position=str(style["legend_position"]),
                        legend_label_max_len=int(style["legend_label_max_len"]),
                        legend_columns=int(style["legend_columns"]),
                    )
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                    if adjusted:
                        st.caption("Axis range was expanded to keep data visible.")
                    st.dataframe(preview_summary, use_container_width=True, hide_index=True)
                    if not preview_cycles.empty:
                        with st.expander("Cycle summary", expanded=False):
                            st.dataframe(preview_cycles, use_container_width=True, hide_index=True)
        return

    st.markdown("### Final output")
    style = current_dqdv_style()
    sample_colors = current_dqdv_colors(style)
    selected_paths = {s: selected_dqdv_paths_for_sample(s, records_by_sample[s]) for s in selected_samples}
    signature = hashlib.sha1(
        repr(
            {
                "root_dir": str(root_dir),
                "output_dir": str(output_dir),
                "selected_samples": selected_samples,
                "selected_paths": selected_paths,
                "files": {sample: [file_record_signature(record) for record in records_by_sample[sample]] for sample in selected_samples},
                "cycles_override": cycles_override,
                "cycle_start": int(cycle_start),
                "cycle_step": int(cycle_step),
                "charge_step": charge_step,
                "discharge_step": discharge_step,
                "retention_cutoff": effective_retention_cutoff,
                "stop_at_retention_cutoff": bool(stop_at_retention_cutoff),
                "style": style,
                "colors": sample_colors,
                "implementation": "dqdv_final_v1",
            }
        ).encode("utf-8")
    ).hexdigest()
    cached = st.session_state.get("dqdv_final_output_cache")
    if cached and cached.get("signature") == signature:
        if st.button("Regenerate final outputs", type="primary"):
            st.session_state.pop("dqdv_final_output_cache", None)
        else:
            grid_cols = st.columns(2)
            for idx, item in enumerate(cached["rendered_outputs"]):
                safe_item_key = safe_filename(str(item["title"]))
                with grid_cols[idx % 2]:
                    st.markdown(f"#### {item['title']}")
                    fig = make_dqdv_figure(**item["figure_kwargs"])
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                    c1, c2 = st.columns(2)
                    c1.download_button("CSV", data=item["csv_bytes"], file_name=item["csv_file_name"], mime="text/csv", key=f"dqdv_csv_{idx}_{safe_item_key}")
                    c2.download_button("PNG", data=item["png_bytes"], file_name=item["png_file_name"], mime="image/png", key=f"dqdv_png_{idx}_{safe_item_key}")
            st.success(f"Batch dQ/dV analysis completed. Results saved to: `{cached['output_dir']}`")
            st.dataframe(cached["summary_df"], use_container_width=True, hide_index=True)
            st.download_button("Download all dQ/dV results ZIP", data=cached["zip_bytes"], file_name="dqdv_results.zip", mime="application/zip")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures_by_file"
    data_dir = output_dir / "plot_data_by_file"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    rendered_outputs = []
    summary_frames = []
    cycle_frames = []
    zip_buffer = io.BytesIO()
    output_records = [record for sample in selected_samples for record in selected_dqdv_records(sample)]

    progress = st.progress(0)
    status = st.empty()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, record in enumerate(output_records, start=1):
            title = f"{record['sample']} / {record['repeat']} / {record['source_file']}"
            status.write(f"Processing {title} ({idx}/{len(output_records)})...")
            plot_df, summary_df, cycle_summary = load_dqdv_records_for_plot([record])
            if not summary_df.empty:
                summary_frames.append(summary_df)
            if not cycle_summary.empty:
                cycle_frames.append(cycle_summary)
            if plot_df is None:
                st.warning(f"No valid dQ/dV data found for `{title}`.")
                progress.progress(idx / max(1, len(output_records)))
                continue
            safe_name = f"{safe_filename(str(record['sample']))}__{safe_filename(str(record['repeat']))}__{safe_filename(Path(str(record['source_file'])).stem)}"
            csv_path = data_dir / f"{safe_name}_plot_data.csv"
            png_path = figures_dir / f"{safe_name}_dqdv.png"
            summary_path = data_dir / f"{safe_name}_summary.csv"
            cycles_path = data_dir / f"{safe_name}_cycle_summary.csv"
            plot_df.to_csv(csv_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            cycle_summary.to_csv(cycles_path, index=False)
            effective_limits, numeric_df, adjusted = dqdv_figure_limits(plot_df, style)
            figure_kwargs = dict(
                plot_df=plot_df,
                sample_name=str(record["sample"]),
                repeat_name=str(record["repeat"]),
                source_file=str(record["source_file"]),
                color_hex=sample_colors.get(str(record["sample"]), "#4E79A7"),
                plot_title=str(style["plot_title"]),
                x_label=str(style["x_label"]),
                y_label=str(style["y_label"]),
                legend_title=str(style["legend_title"]),
                show_legend=bool(style["show_legend"]),
                x_min=float(effective_limits["x_min"]),
                x_max=float(effective_limits["x_max"]),
                y_min=float(effective_limits["y_min"]),
                y_max=float(effective_limits["y_max"]),
                linewidth=float(style["linewidth"]),
                fig_width=float(style["fig_width"]),
                fig_height=float(style["fig_height"]),
                legend_position=str(style["legend_position"]),
                legend_label_max_len=int(style["legend_label_max_len"]),
                legend_columns=int(style["legend_columns"]),
            )
            fig = make_dqdv_figure(**figure_kwargs)
            fig.canvas.draw()
            fig.savefig(png_path, dpi=int(style["dpi"]), bbox_inches="tight")
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=int(style["dpi"]), bbox_inches="tight")
            png_buffer.seek(0)
            png_bytes = png_buffer.getvalue()
            plt.close(fig)
            csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
            summary_bytes = summary_df.to_csv(index=False).encode("utf-8")
            cycle_bytes = cycle_summary.to_csv(index=False).encode("utf-8")
            zipf.writestr(f"{safe_name}/{safe_name}_plot_data.csv", csv_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_summary.csv", summary_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_cycle_summary.csv", cycle_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_dqdv.png", png_bytes)
            rendered_outputs.append(
                {
                    "title": title,
                    "csv_bytes": csv_bytes,
                    "png_bytes": png_bytes,
                    "csv_file_name": f"{safe_name}_plot_data.csv",
                    "png_file_name": f"{safe_name}_dqdv.png",
                    "figure_kwargs": figure_kwargs,
                    "adjusted_limits": adjusted,
                    "numeric_points": len(numeric_df),
                }
            )
            progress.progress(idx / max(1, len(output_records)))

    status.empty()
    progress.empty()
    if not rendered_outputs:
        st.warning("No valid dQ/dV results were generated.")
        return

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    cycle_summary_df = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    summary_df.to_csv(output_dir / "summary_by_file.csv", index=False)
    cycle_summary_df.to_csv(output_dir / "cycle_summary_by_file.csv", index=False)
    zip_buffer.seek(0)
    cache = {
        "signature": signature,
        "output_dir": str(output_dir),
        "rendered_outputs": rendered_outputs,
        "summary_df": summary_df,
        "zip_bytes": zip_buffer.getvalue(),
    }
    st.session_state["dqdv_final_output_cache"] = cache
    rerun_streamlit_app()


def render_placeholder_page(title: str) -> None:
    st.title(title)
    st.info("This module can be added later without changing the EIS fitting logic.")


def main() -> None:
    st.set_page_config(page_title="Battery Data Analysis", page_icon="🔋", layout="wide")

    st.sidebar.title("🔋 Battery Data Analysis")
    tool = st.sidebar.selectbox(
        "Choose analysis tool",
        [
            "EIS Fit",
            "EIS Fit Batch",
            "Cycling Analysis",
            "Stripping Overpotential",
            "dQ/dV Analysis",
        ],
    )

    if tool == "EIS Fit":
        render_eis_fit_page()
    elif tool == "EIS Fit Batch":
        render_eis_fit_batch_page()
    elif tool == "Cycling Analysis":
        render_cycling_analysis_page()
    elif tool == "Stripping Overpotential":
        render_stripping_analysis_page()
    elif tool == "dQ/dV Analysis":
        render_dqdv_analysis_page()


if __name__ == "__main__":
    main()
