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

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
        render_placeholder_page("Cycling Analysis")
    elif tool == "Stripping Overpotential":
        render_placeholder_page("Stripping Overpotential")
    elif tool == "dQ/dV Analysis":
        render_placeholder_page("dQ/dV Analysis")


if __name__ == "__main__":
    main()
