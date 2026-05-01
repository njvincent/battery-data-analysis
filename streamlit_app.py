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
import re
import io
import tempfile
import zipfile
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
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
) -> tuple[pd.DataFrame | None, str | None]:
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
        st.session_state["cycling_workflow_step"] = "2. Style preview"


def set_cycling_workflow_step(step: str) -> None:
    """Set the cycling workflow step from a button callback."""
    st.session_state["cycling_workflow_step"] = step


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
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Sample", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Relative path", "Note"])
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
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

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
            full_label = Path(str(group["source_file"].iloc[0])).stem
        else:
            full_label = Path(str(group_id)).stem
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
    legend_rows = int(np.ceil(n_labels / ncol))

    # Avoid tight_layout here. It often cannot correctly reserve space for a
    # figure-level legend together with a twinx right y-axis.
    if legend_position == "Top":
        title_space = 0.08 if title.strip() else 0.0
        legend_space = min(0.22, 0.075 + 0.045 * legend_rows)
        top = max(0.58, 1.0 - title_space - legend_space - 0.02)
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
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

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
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

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

    def metric_text(value: object, suffix: str = "", digits: int = 3) -> str:
        text = format_metric_value(value, digits=digits)
        return f"{text}{suffix}" if text != "—" else text

    ice_text = metric_text(metrics.get("ICE (%)"), "%")
    life_text = metric_text(metrics.get("Cycle Life"), "", digits=4)
    ace_text = metric_text(metrics.get("ACE (%)"), "%")
    ace_cycle_text = metric_text(metrics.get("ACE cycle"), "", digits=4)
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

        st.markdown(
            f"""
            <div style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                        font-size:0.74rem; line-height:1.28; min-height:42px; margin-top:-0.25rem;">
                <div style="display:grid; grid-template-columns: 1fr 1fr; column-gap:0.55rem;">
                    <span><b>ICE</b>: {ice_text}</span>
                    <span><b>Life</b>: {life_text}</span>
                    <span><b>ACE</b>: {ace_text}</span>
                    <span><b>ACE cyc.</b>: {ace_cycle_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if note_text:
            st.markdown(
                f"<div style='font-size:0.72rem; color:rgba(120,120,120,0.95); min-height:18px;'>Note: {note_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div style='min-height:18px;'></div>", unsafe_allow_html=True)


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

        top_n_enabled = st.checkbox(
            "Only plot top N included files per sample",
            value=False,
            help="Applied after manual file selection. Files are ranked by the sum of capacity retention.",
        )

        top_n = st.number_input(
            "Top N files",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
            disabled=not top_n_enabled,
        )

    if root_dir is None or output_dir is None:
        st.info(input_mode)
        return

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
    top_n_value = int(top_n) if top_n_enabled else None

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
        "cycling_marker_size": 80,
        "cycling_fig_width": 9.5,
        "cycling_fig_height": 5.8,
        "cycling_dpi": 300,
    }
    for key, value in style_defaults.items():
        st.session_state.setdefault(key, value)

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

    def render_final_output_cache(cache: dict[str, object]) -> None:
        rendered_outputs = cache["rendered_outputs"]
        summary_df = cache["summary_df"]
        selected_file_summary_df = cache["selected_file_summary_df"]
        selection_rows = cache["selection_rows"]

        st.subheader("Final figures")
        output_cols = st.columns(2)
        for i, item in enumerate(rendered_outputs):
            safe_name = safe_filename(str(item["sample"]))
            with output_cols[i % 2]:
                st.markdown(f"#### {item['sample']}")
                display_fig = make_capacity_figure(**item["figure_kwargs"])
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
            display_cols = ["Sample", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Note"]
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

        preview_axis_mode = "Fixed common range"
        preview_min_retention = min_retention

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
            )
            raw_df, raw_error = load_raw_capacity_metrics_df(
                record=record,
                sample_name=inspect_sample,
                root_dir=root_dir,
                sheet_name=sheet_name,
                capacity_col=capacity_col,
                efficiency_col=efficiency_col,
                skip_initial_rows=int(skip_initial_rows),
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
            display_cols = ["ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Note"]
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
            preview_sample = st.selectbox(
                "Preview sample",
                options=selected_samples,
                key="cycling_preview_sample",
                help="This preview always uses real selected files. Final output processes all selected samples.",
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
                    )

            if preview_file_count == 0:
                st.warning(f"No selected Excel files found for preview sample `{preview_sample}`.")
            elif preview_df is None:
                st.warning(f"No valid cycling data found for preview sample `{preview_sample}`.")
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
                "sheet_name": sheet_name,
                "capacity_col": capacity_col,
                "efficiency_col": efficiency_col,
                "skip_initial_rows": int(skip_initial_rows),
                "min_retention": min_retention,
                "top_n_value": top_n_value,
                "style": style,
                "sample_colors": sample_colors,
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

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, sample_name in enumerate(selected_samples, start=1):
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
        else pd.DataFrame(columns=["Sample", "ICE (%)", "Cycle Life", "ACE (%)", "ACE cycle", "Time", "Operator", "File name", "Relative path", "Note"])
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
        render_placeholder_page("Stripping Overpotential")
    elif tool == "dQ/dV Analysis":
        render_placeholder_page("dQ/dV Analysis")


if __name__ == "__main__":
    main()
