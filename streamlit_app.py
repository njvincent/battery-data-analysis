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
) -> tuple[pd.DataFrame | None, int]:
    """
    Load and optionally filter all Excel cycling files for one sample.
    Returns (plot_df, number_of_excel_files_found).
    """
    excel_files = find_capacity_excel_files(sample_dir)

    if not excel_files:
        return None, 0

    sample_dfs = []

    for file_path in excel_files:
        one_df = read_one_capacity_file(
            file_path=file_path,
            sample_name=sample_name,
            root_dir=root_dir,
            sheet_name=sheet_name,
            capacity_col=capacity_col,
            efficiency_col=efficiency_col,
            skip_initial_rows=int(skip_initial_rows),
            min_capacity_retention=min_retention,
        )

        if one_df is not None:
            sample_dfs.append(one_df)

    if not sample_dfs:
        return None, len(excel_files)

    plot_df = pd.concat(sample_dfs, ignore_index=True)

    if top_n_value is not None:
        scores = (
            plot_df.groupby("source_file")["capacity_retention_percent"]
            .sum()
            .sort_values(ascending=False)
        )
        keep_files = scores.head(top_n_value).index
        plot_df = plot_df[plot_df["source_file"].isin(keep_files)].copy()

    if plot_df.empty:
        return None, len(excel_files)

    return plot_df, len(excel_files)


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

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    ax2 = ax1.twinx()

    color_rgb = hex_to_rgb_tuple(color_hex)

    for source_file, group in plot_df.groupby("source_file", sort=True):
        group = group.sort_values("cycle_index")
        full_label = Path(source_file).stem
        file_label = shorten_label(full_label, legend_label_max_len)

        ax1.scatter(
            group["cycle_index"],
            group["capacity_retention_percent"],
            color=color_rgb,
            marker="o",
            s=marker_size,
            alpha=1,
            zorder=3,
            label=file_label,
        )

        ax2.scatter(
            group["cycle_index"],
            group["coulombic_efficiency_percent"],
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
        st.header("Cycling input")

        root_dir_str = st.text_input(
            "Root data directory",
            value="",
            help="Folder containing sample subfolders. This must be accessible to the Streamlit process.",
        )

        output_dir_str = st.text_input(
            "Output directory",
            value="",
            help="Leave empty to save to <root_dir>/capacity_batch_results.",
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

        top_n_enabled = st.checkbox(
            "Only plot top N files per sample",
            value=False,
            help="Files are ranked by the sum of capacity retention.",
        )

        top_n = st.number_input(
            "Top N files",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            disabled=not top_n_enabled,
        )

    if not root_dir_str.strip():
        st.info("Enter a root data directory to start.")
        return

    root_dir = Path(root_dir_str).expanduser().resolve()

    if not root_dir.exists():
        st.error(f"Root directory does not exist: `{root_dir}`")
        return

    if not root_dir.is_dir():
        st.error(f"Root path is not a directory: `{root_dir}`")
        return

    if output_dir_str.strip():
        output_dir = Path(output_dir_str).expanduser().resolve()
    else:
        output_dir = root_dir / "capacity_batch_results"

    sample_folders = find_capacity_sample_folders(root_dir, output_dir)

    if not sample_folders:
        st.warning("No sample folders found under the root directory.")
        return

    sample_names = [folder.name for folder in sample_folders]
    folder_map = {folder.name: folder for folder in sample_folders}

    st.subheader("Customize and preview")
    st.caption("The preview updates automatically when you change settings on the left.")

    controls_col, preview_col = st.columns([1.0, 1.85], gap="large")

    with controls_col:
        st.markdown("### Samples")

        selected_samples = st.multiselect(
            "Samples to process",
            options=sample_names,
            default=sample_names,
        )

        if not selected_samples:
            st.warning("Select at least one sample.")
            return

        preview_sample = st.selectbox(
            "Preview sample",
            options=selected_samples,
            help="Only this sample is loaded for the real-data preview. The final processing step still processes all selected samples.",
        )

        preview_mode = st.radio(
            "Preview data source",
            ["Fast placeholder", "Real selected sample"],
            index=0,
            horizontal=True,
            help=(
                "Fast placeholder updates immediately and is best for layout/style. "
                "Real selected sample reads Excel files and can be slower."
            ),
        )

        st.markdown("### Plot text")

        plot_title = st.text_input(
            "Plot title",
            value="{sample}",
            help='Use "{sample}" to insert the sample folder name.',
        )

        x_label = st.text_input("X-axis label", value="Cycle Index")
        cap_y_label = st.text_input("Left Y-axis label", value="Capacity Retention (%)")
        ce_y_label = st.text_input("Right Y-axis label", value="Coulombic Efficiency (%)")

        st.markdown("### Legend")

        show_legend = st.checkbox("Show legend", value=True)

        legend_position = st.selectbox(
            "Legend position",
            ["Top", "Right", "Inside", "Hide"],
            index=0,
            disabled=not show_legend,
            help="Top is the default because it does not squeeze the plot horizontally.",
        )

        legend_title = st.text_input(
            "Legend title",
            value="Files",
            disabled=not show_legend or legend_position == "Hide",
        )

        legend_label_max_len = st.slider(
            "Max legend label length",
            min_value=8,
            max_value=60,
            value=24,
            step=1,
            disabled=not show_legend or legend_position == "Hide",
            help="Long file names are shortened only in the legend. Output CSV still keeps full source_file names.",
        )

        legend_columns = st.slider(
            "Legend columns",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            disabled=not show_legend or legend_position != "Top",
        )

        st.markdown("### Axis ranges")

        auto_x_range = st.checkbox("Auto X-axis range", value=True)

        x_min = st.number_input(
            "X min",
            value=0.0,
            step=10.0,
            disabled=auto_x_range,
        )

        x_max = st.number_input(
            "X max",
            value=500.0,
            step=10.0,
            disabled=auto_x_range,
        )

        cap_y_min = st.number_input("Capacity retention Y min", value=75.0, step=1.0)
        cap_y_max = st.number_input("Capacity retention Y max", value=110.0, step=1.0)

        ce_y_min = st.number_input("Coulombic efficiency Y min", value=90.0, step=0.5)
        ce_y_max = st.number_input("Coulombic efficiency Y max", value=100.5, step=0.5)

        st.markdown("### Style")

        palette_name = st.selectbox(
            "Default color palette",
            [
                "Set2 + Dark2 + tab20",
                "Set2",
                "Dark2",
                "tab10",
                "tab20",
                "tab20 + tab20b",
            ],
            index=0,
        )

        default_colors = palette_to_hex_colors(palette_name, len(sample_names))
        default_color_map = {
            sample: default_colors[i]
            for i, sample in enumerate(sample_names)
        }

        marker_size = st.slider(
            "Marker size",
            min_value=20,
            max_value=200,
            value=80,
            step=5,
        )

        fig_width = st.number_input(
            "Figure width",
            min_value=4.0,
            max_value=20.0,
            value=9.5,
            step=0.5,
            help="The default is wider than before to leave room for axis labels and top legends.",
        )

        fig_height = st.number_input(
            "Figure height",
            min_value=3.0,
            max_value=15.0,
            value=5.8,
            step=0.5,
        )

        dpi = st.number_input(
            "Saved figure DPI",
            min_value=72,
            max_value=600,
            value=300,
            step=50,
        )

        st.markdown("### Sample colors")

        sample_colors = {}
        with st.expander("Edit sample colors", expanded=True):
            for i, sample in enumerate(selected_samples, start=1):
                sample_colors[sample] = st.color_picker(
                    compact_widget_label("Color", i, sample),
                    value=default_color_map[sample],
                    key=f"cycling_color_{safe_filename(sample)}",
                    help=f"Full sample name: {sample}",
                )

        run = st.button("Process and save all selected samples", type="primary")

    min_retention = float(min_capacity_retention) if use_retention_filter else None
    top_n_value = int(top_n) if top_n_enabled else None

    with preview_col:
        st.markdown("### Live preview")

        preview_is_placeholder = preview_mode == "Fast placeholder"

        if preview_is_placeholder:
            preview_df = make_capacity_placeholder_plot_data(
                sample_name=preview_sample,
                n_files=int(top_n_value or 3),
                max_cycle=int(x_max if not auto_x_range else 500),
            )
            preview_file_count = preview_df["source_file"].nunique()
            st.caption(
                "Using lightweight placeholder data for instant style preview. "
                "Switch to **Real selected sample** when you want to inspect actual Excel data."
            )
        else:
            with st.spinner(f"Loading preview data for {preview_sample}..."):
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
                )

        if preview_file_count == 0:
            st.warning(f"No Excel files found for preview sample `{preview_sample}`.")
        elif preview_df is None:
            st.warning(f"No valid cycling data found for preview sample `{preview_sample}`.")
        else:
            preview_fig = make_capacity_figure(
                plot_df=preview_df,
                sample_name=preview_sample,
                color_hex=sample_colors[preview_sample],
                plot_title=plot_title,
                x_label=x_label,
                cap_y_label=cap_y_label,
                ce_y_label=ce_y_label,
                legend_title=legend_title,
                show_legend=show_legend,
                legend_position=legend_position,
                legend_label_max_len=int(legend_label_max_len),
                legend_columns=int(legend_columns),
                auto_x_range=auto_x_range,
                x_min=float(x_min),
                x_max=float(x_max),
                cap_y_min=float(cap_y_min),
                cap_y_max=float(cap_y_max),
                ce_y_min=float(ce_y_min),
                ce_y_max=float(ce_y_max),
                marker_size=int(marker_size),
                fig_width=float(fig_width),
                fig_height=float(fig_height),
            )
            st.pyplot(preview_fig, clear_figure=True)
            plt.close(preview_fig)

            c1, c2, c3 = st.columns(3)
            c1.metric("Preview files", preview_df["source_file"].nunique())
            c2.metric("Preview points", len(preview_df))
            c3.metric("Max cycle", _fmt_num(preview_df["cycle_index"].max()))

            if not preview_is_placeholder:
                with st.expander("Preview plotting data"):
                    st.dataframe(preview_df, use_container_width=True)

    if not run:
        st.info(
            "Preview changes live. Use **Fast placeholder** for layout/style, then click "
            "**Process and save all selected samples** when the figure style looks right."
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    zip_buffer = io.BytesIO()

    progress = st.progress(0)
    status = st.empty()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, sample_name in enumerate(selected_samples, start=1):
            status.write(f"Processing {sample_name} ({idx}/{len(selected_samples)})...")

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
            )

            if excel_file_count == 0:
                st.warning(f"No Excel files found for sample `{sample_name}`.")
                progress.progress(idx / len(selected_samples))
                continue

            if plot_df is None:
                st.warning(f"No valid cycling data found for sample `{sample_name}`.")
                progress.progress(idx / len(selected_samples))
                continue

            safe_name = safe_filename(sample_name)
            sample_output_dir = output_dir / safe_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = sample_output_dir / f"{safe_name}_plot_data.csv"
            png_path = sample_output_dir / f"{safe_name}_capacity_summary.png"

            plot_df.to_csv(csv_path, index=False)

            fig = make_capacity_figure(
                plot_df=plot_df,
                sample_name=sample_name,
                color_hex=sample_colors[sample_name],
                plot_title=plot_title,
                x_label=x_label,
                cap_y_label=cap_y_label,
                ce_y_label=ce_y_label,
                legend_title=legend_title,
                show_legend=show_legend,
                legend_position=legend_position,
                legend_label_max_len=int(legend_label_max_len),
                legend_columns=int(legend_columns),
                auto_x_range=auto_x_range,
                x_min=float(x_min),
                x_max=float(x_max),
                cap_y_min=float(cap_y_min),
                cap_y_max=float(cap_y_max),
                ce_y_min=float(ce_y_min),
                ce_y_max=float(ce_y_max),
                marker_size=int(marker_size),
                fig_width=float(fig_width),
                fig_height=float(fig_height),
            )

            fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=int(dpi), bbox_inches="tight")
            png_buffer.seek(0)

            csv_bytes = plot_df.to_csv(index=False).encode("utf-8")

            zipf.writestr(f"{safe_name}/{safe_name}_plot_data.csv", csv_bytes)
            zipf.writestr(f"{safe_name}/{safe_name}_capacity_summary.png", png_buffer.getvalue())

            st.markdown(f"## {sample_name}")
            st.pyplot(fig, clear_figure=True)

            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    f"Download {shorten_label(sample_name, 28)} CSV",
                    data=csv_bytes,
                    file_name=f"{safe_name}_plot_data.csv",
                    mime="text/csv",
                )

            with c2:
                st.download_button(
                    f"Download {shorten_label(sample_name, 28)} PNG",
                    data=png_buffer.getvalue(),
                    file_name=f"{safe_name}_capacity_summary.png",
                    mime="image/png",
                )

            with st.expander(f"Preview plotting data: {shorten_label(sample_name, 40)}"):
                st.dataframe(plot_df, use_container_width=True)

            plt.close(fig)

            summary_rows.append(
                {
                    "sample": sample_name,
                    "files_found": excel_file_count,
                    "files_plotted": plot_df["source_file"].nunique(),
                    "points_plotted": len(plot_df),
                    "color": sample_colors[sample_name],
                    "csv_path": str(csv_path),
                    "png_path": str(png_path),
                }
            )

            progress.progress(idx / len(selected_samples))

    status.empty()
    progress.empty()

    if not summary_rows:
        st.warning("No valid results were generated.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "capacity_batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    st.success(f"Batch cycling analysis completed. Results saved to: `{output_dir}`")

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True)

    zip_buffer.seek(0)

    st.download_button(
        "Download all cycling results ZIP",
        data=zip_buffer.getvalue(),
        file_name="capacity_batch_results.zip",
        mime="application/zip",
    )


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
