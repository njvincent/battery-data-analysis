#!/usr/bin/env python3
"""
Battery Data Analysis Streamlit dashboard.

This file is intentionally UI-focused. It reuses the fitting/read/plot helpers
from eis_web_app.py, so the EIS fitting logic can stay unchanged.

Expected repo layout:
    battery-data-analysis/
    ├── streamlit_app.py              # this file, after renaming
    ├── eis_web_app.py                # EIS core logic/helper functions
    ├── requirements.txt
    └── ...

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Reuse the existing EIS logic. Keep this file in the same repo/folder.
from eis_web_app import (
    PARAM_ORDER,
    FitResultBundle,
    low_freq_zoom_figure,
    make_fit_bundle,
    make_zip_download,
    nyquist_figure,
    pack_params,
    read_uploaded_eis,
    read_zfit_xml_bytes,
)


# -----------------------------------------------------------------------------
# General UI helpers
# -----------------------------------------------------------------------------


def _fmt_num(x: float, unit: str = "", sig: int = 3) -> str:
    """Compact number formatting for metric cards and summary tables."""
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        x = float(x)
    except Exception:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{x:.{sig}g}{suffix}"


def _fit_quality_summary(bundle: FitResultBundle, low_freq_cutoff: float) -> dict[str, float | str | bool]:
    """Return concise fit-quality descriptors used by single and batch pages."""
    df = bundle.curve_df
    re = df["residual_real_ohm"].to_numpy(float)
    im = df["residual_minus_imag_ohm"].to_numpy(float)
    rmse_z = float(np.sqrt(np.nanmean(re**2 + im**2)))
    rmse_im = float(np.sqrt(np.nanmean(im**2)))

    low = df[df["freq_hz"] <= float(low_freq_cutoff)]
    if len(low):
        low_bias = float(low["residual_minus_imag_ohm"].mean())
        low_max_abs = float(low["residual_minus_imag_ohm"].abs().max())
        low_n = int(len(low))
    else:
        low_bias = np.nan
        low_max_abs = np.nan
        low_n = 0

    fit_params = dict(zip(bundle.params_df["parameter"], bundle.params_df["fit"]))
    arc_df = bundle.arc_df.copy()
    fusion_df = bundle.fusion_df.copy()

    # The final right intercept is a compact estimate of the fitted total real-axis span.
    total_right = float(arc_df["right_intercept_ohm"].iloc[-1]) if len(arc_df) else np.nan
    total_diameter = float(arc_df["diameter_ohm"].sum()) if len(arc_df) else np.nan

    arc3 = arc_df[arc_df["arc"] == 3]
    arc3_height = float(arc3["max_height_minus_im_ohm"].iloc[0]) if len(arc3) else np.nan
    arc3_depression = float(arc3["depression_ratio_height_over_radius"].iloc[0]) if len(arc3) else np.nan

    fusion23 = fusion_df[fusion_df["arc_pair"] == "2-3"]
    fusion23_val = (
        float(fusion23["fusion_index_overlap_over_narrower_FWHM"].iloc[0])
        if len(fusion23)
        else np.nan
    )

    s4 = float(fit_params.get("s4", np.nan))
    a4 = float(fit_params.get("a4", np.nan))
    low_note = "OK"
    if np.isfinite(low_bias) and abs(low_bias) > max(5.0, 0.05 * max(1.0, arc3_height)):
        low_note = "fit high at low-f" if low_bias > 0 else "fit low at low-f"
    if np.isfinite(s4) and abs(s4) < 1e-8:
        low_note = "Warburg inactive / arc 3 effective"
    if np.isfinite(a4) and (a4 > 0.995 or a4 < 0.055):
        low_note = "a4 near bound; check arc 3"

    return {
        "success": bool(getattr(bundle.result, "success", False)),
        "cost": float(getattr(bundle.result, "cost", np.nan)),
        "nfev": int(getattr(bundle.result, "nfev", -1)),
        "rmse_z_ohm": rmse_z,
        "rmse_minus_imag_ohm": rmse_im,
        "low_f_points": low_n,
        "low_f_bias_ohm": low_bias,
        "low_f_max_abs_ohm": low_max_abs,
        "R1_ohm": float(fit_params.get("R1", np.nan)),
        "R2_ohm": float(fit_params.get("R2", np.nan)),
        "R3_ohm": float(fit_params.get("R3", np.nan)),
        "R4_ohm": float(fit_params.get("R4", np.nan)),
        "s4": s4,
        "a4": a4,
        "total_diameter_ohm": total_diameter,
        "right_intercept_final_ohm": total_right,
        "arc3_height_ohm": arc3_height,
        "arc3_depression_ratio": arc3_depression,
        "fusion_2_3": fusion23_val,
        "low_f_note": low_note,
    }


def _summary_row(
    file_name: str,
    df: pd.DataFrame,
    bundle: FitResultBundle,
    low_freq_cutoff: float,
) -> dict[str, object]:
    q = _fit_quality_summary(bundle, low_freq_cutoff)
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
        "note": q["low_f_note"],
    }


def _display_metric_strip(bundle: FitResultBundle, df: pd.DataFrame, low_freq_cutoff: float) -> None:
    """Small, information-dense row of important fit results."""
    q = _fit_quality_summary(bundle, low_freq_cutoff)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Points", f"{len(df)}")
    c2.metric("Freq. range", f"{_fmt_num(df['freq_hz'].min())}–{_fmt_num(df['freq_hz'].max())} Hz")
    c3.metric("RMSE |Z|", _fmt_num(q["rmse_z_ohm"], "Ω"))
    c4.metric("Low-f bias", _fmt_num(q["low_f_bias_ohm"], "Ω"))
    c5.metric("Final intercept", _fmt_num(q["right_intercept_final_ohm"], "Ω"))
    c6.metric("Arc 2–3 fusion", _fmt_num(q["fusion_2_3"]))

    note = q["low_f_note"]
    if note != "OK":
        st.warning(f"Low-frequency check: {note}")
    else:
        st.success("Fit completed. No major low-frequency warning from the compact checks.")


def _format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, rounded table for screen display only."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.kind in "f":
            out[col] = out[col].map(lambda x: np.nan if pd.isna(x) else float(f"{x:.5g}"))
    return out


def _read_xml_params_from_sidebar(xml_file) -> dict[str, float]:
    if xml_file is None:
        return {}
    try:
        xml_params = read_zfit_xml_bytes(xml_file.getvalue())
        st.sidebar.success(f"Loaded {len(xml_params)} XML initial parameters.")
        return xml_params
    except Exception as exc:
        st.sidebar.warning(f"Could not read XML: {exc}")
        return {}


def _initial_parameter_editor(xml_params: dict[str, float], key_prefix: str) -> np.ndarray:
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


def _weights_to_run(primary_weight: str, compare_weights: bool) -> list[str]:
    return ["unit", "sqrt_modulus", "modulus"] if compare_weights else [primary_weight]


def _primary_bundle(bundles: list[FitResultBundle], primary_weight: str) -> FitResultBundle:
    return next((b for b in bundles if b.weight == primary_weight), bundles[0])


def _downloads_for_bundle(bundle: FitResultBundle, file_stem: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Fit curve CSV",
        bundle.curve_df.to_csv(index=False),
        file_name=f"{file_stem}_{bundle.weight}_fit_curve.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c2.download_button(
        "Fit params CSV",
        bundle.params_df.to_csv(index=False),
        file_name=f"{file_stem}_{bundle.weight}_fit_params.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        "Arc metrics CSV",
        bundle.arc_df.to_csv(index=False),
        file_name=f"{file_stem}_{bundle.weight}_arc_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c4.download_button(
        "Fusion CSV",
        bundle.fusion_df.to_csv(index=False),
        file_name=f"{file_stem}_{bundle.weight}_fusion_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# EIS single-file page
# -----------------------------------------------------------------------------


def render_eis_fit_page() -> None:
    st.title("EIS Fit")
    st.caption(
        "Single-file Nyquist fitting with compact preview. Current model: "
        "`R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4)`."
    )

    with st.sidebar:
        st.header("Input")
        data_file = st.file_uploader(
            "EIS data file (.mpr, .csv, .txt)",
            type=["mpr", "csv", "txt"],
            accept_multiple_files=False,
            key="single_data_file",
        )
        xml_file = st.file_uploader(
            "Optional EC-Lab ZFit XML",
            type=["xml"],
            accept_multiple_files=False,
            key="single_xml_file",
        )

        st.header("Fit options")
        primary_weight = st.selectbox(
            "Primary weighting",
            ["unit", "sqrt_modulus", "modulus"],
            index=0,
            key="single_primary_weight",
        )
        compare_weights = st.checkbox("Compare all three weightings", value=True, key="single_compare_weights")
        max_nfev = st.number_input(
            "Max function evaluations",
            min_value=1000,
            max_value=200000,
            value=50000,
            step=5000,
            key="single_max_nfev",
        )
        show_low_freq_labels = st.checkbox(
            "Label lowest-frequency points",
            value=False,
            key="single_low_f_labels",
        )
        low_freq_cutoff = st.number_input(
            "Low-frequency check cutoff / Hz",
            min_value=1e-6,
            max_value=1e6,
            value=0.1,
            format="%.6g",
            key="single_low_f_cutoff",
        )

        xml_params = _read_xml_params_from_sidebar(xml_file)
        p0 = _initial_parameter_editor(xml_params, "single")

    if data_file is None:
        st.info("Upload one `.mpr`, `.csv`, or `.txt` EIS file to start.")
        with st.expander("What this page reports", expanded=True):
            st.markdown(
                """
                - **Nyquist fit** and optional **low-frequency zoom**
                - Compact fit-quality cards: RMSE, low-frequency bias, final intercept, arc fusion
                - Full fit parameters, arc descriptors, fusion descriptors, and downloadable CSV outputs
                - No residual plot by default, to keep the preview focused
                """
            )
        return

    try:
        df = read_uploaded_eis(data_file).sort_values("freq_hz", ascending=False).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not read {data_file.name}: {exc}")
        return

    weights = _weights_to_run(primary_weight, compare_weights)
    bundles: list[FitResultBundle] = []
    with st.spinner(f"Fitting {data_file.name}..."):
        for w in weights:
            try:
                bundles.append(make_fit_bundle(data_file.name, df, p0, w, int(max_nfev)))
            except Exception as exc:
                st.error(f"Fit failed with weight={w}: {exc}")

    if not bundles:
        return

    primary = _primary_bundle(bundles, primary_weight)
    file_stem = Path(data_file.name).stem.replace(" ", "_")

    st.subheader(data_file.name)
    _display_metric_strip(primary, df, float(low_freq_cutoff))

    tab_overview, tab_params, tab_metrics, tab_data, tab_downloads = st.tabs(
        ["Overview", "Parameters", "Arc/fusion", "Data", "Downloads"]
    )

    with tab_overview:
        st.plotly_chart(
            nyquist_figure(df, bundles, show_low_freq_labels=show_low_freq_labels),
            use_container_width=True,
        )
        with st.expander("Low-frequency zoom", expanded=False):
            st.plotly_chart(
                low_freq_zoom_figure(df, bundles, float(low_freq_cutoff)),
                use_container_width=True,
            )

    with tab_params:
        st.caption("Primary table is shown first. Other weightings are available below if comparison is enabled.")
        st.markdown(f"**Primary weight = `{primary.weight}`**")
        st.dataframe(primary.params_df, use_container_width=True, hide_index=True)
        if len(bundles) > 1:
            with st.expander("Other weighting results", expanded=False):
                for b in bundles:
                    if b is primary:
                        continue
                    st.markdown(f"**weight = `{b.weight}`**")
                    st.dataframe(b.params_df, use_container_width=True, hide_index=True)

    with tab_metrics:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Arc geometry descriptors")
            st.dataframe(primary.arc_df, use_container_width=True, hide_index=True)
        with c2:
            st.caption("Fusion descriptors")
            st.dataframe(primary.fusion_df, use_container_width=True, hide_index=True)

    with tab_data:
        preview_cols = [
            "freq_hz",
            "z_real_ohm",
            "minus_z_imag_ohm",
            "fit_z_real_ohm",
            "fit_minus_z_imag_ohm",
        ]
        st.dataframe(primary.curve_df[preview_cols], use_container_width=True, hide_index=True)

    with tab_downloads:
        _downloads_for_bundle(primary, file_stem)
        if len(bundles) > 1:
            st.download_button(
                "Download all weighting outputs as ZIP",
                make_zip_download(bundles),
                file_name=f"{file_stem}_all_weightings.zip",
                mime="application/zip",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# EIS batch page
# -----------------------------------------------------------------------------


def _make_batch_zip(
    summary_df: pd.DataFrame,
    bundles_by_file: dict[str, list[FitResultBundle]],
) -> bytes:
    """Create a single ZIP with summary table plus all per-file outputs."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("batch_summary.csv", summary_df.to_csv(index=False))
        for file_name, bundles in bundles_by_file.items():
            safe_stem = Path(file_name).stem.replace(" ", "_")
            for b in bundles:
                stem = f"{safe_stem}_{b.weight}"
                zf.writestr(f"{stem}_fit_params.csv", b.params_df.to_csv(index=False))
                zf.writestr(f"{stem}_arc_metrics.csv", b.arc_df.to_csv(index=False))
                zf.writestr(f"{stem}_fusion_metrics.csv", b.fusion_df.to_csv(index=False))
                zf.writestr(f"{stem}_fit_curve.csv", b.curve_df.to_csv(index=False))
    return mem.getvalue()


def render_eis_fit_batch_page() -> None:
    st.title("EIS Fit Batch")
    st.caption(
        "Batch fitting for multiple uploaded EIS files. The preview is summarized first; "
        "select one file for detailed Nyquist and parameter inspection."
    )

    with st.sidebar:
        st.header("Input")
        data_files = st.file_uploader(
            "EIS data files (.mpr, .csv, .txt)",
            type=["mpr", "csv", "txt"],
            accept_multiple_files=True,
            key="batch_data_files",
        )
        xml_file = st.file_uploader(
            "Optional EC-Lab ZFit XML",
            type=["xml"],
            accept_multiple_files=False,
            key="batch_xml_file",
        )

        st.header("Fit options")
        primary_weight = st.selectbox(
            "Primary weighting",
            ["unit", "sqrt_modulus", "modulus"],
            index=0,
            key="batch_primary_weight",
        )
        compare_weights = st.checkbox(
            "Run all three weightings",
            value=False,
            key="batch_compare_weights",
            help="For many files, primary weighting only is faster and keeps the summary table cleaner.",
        )
        show_all_weights_in_summary = st.checkbox(
            "Show all weightings in summary",
            value=False,
            key="batch_show_all_weights_summary",
            disabled=not compare_weights,
        )
        max_nfev = st.number_input(
            "Max function evaluations",
            min_value=1000,
            max_value=200000,
            value=50000,
            step=5000,
            key="batch_max_nfev",
        )
        show_low_freq_labels = st.checkbox(
            "Label lowest-frequency points in preview",
            value=False,
            key="batch_low_f_labels",
        )
        low_freq_cutoff = st.number_input(
            "Low-frequency check cutoff / Hz",
            min_value=1e-6,
            max_value=1e6,
            value=0.1,
            format="%.6g",
            key="batch_low_f_cutoff",
        )

        xml_params = _read_xml_params_from_sidebar(xml_file)
        p0 = _initial_parameter_editor(xml_params, "batch")

    if not data_files:
        st.info("Upload multiple `.mpr`, `.csv`, or `.txt` files to run batch fitting.")
        st.markdown(
            """
            **Batch page layout**
            1. Upload many EIS files from the browser.
            2. Fit all files with the selected model and weighting.
            3. Review one compact summary table.
            4. Choose one file for detailed plot/parameter preview.
            5. Download the full batch output as a ZIP.
            """
        )
        return

    weights = _weights_to_run(primary_weight, compare_weights)
    dfs_by_file: dict[str, pd.DataFrame] = {}
    bundles_by_file: dict[str, list[FitResultBundle]] = {}
    summary_rows: list[dict[str, object]] = []
    errors: list[str] = []

    progress = st.progress(0.0, text="Preparing batch fitting...")
    total_jobs = max(1, len(data_files) * len(weights))
    done_jobs = 0

    for uploaded in data_files:
        file_name = uploaded.name
        try:
            df = read_uploaded_eis(uploaded).sort_values("freq_hz", ascending=False).reset_index(drop=True)
            dfs_by_file[file_name] = df
        except Exception as exc:
            errors.append(f"{file_name}: read failed — {exc}")
            done_jobs += len(weights)
            progress.progress(min(done_jobs / total_jobs, 1.0), text=f"Skipped {file_name}")
            continue

        bundles: list[FitResultBundle] = []
        for w in weights:
            try:
                b = make_fit_bundle(file_name, df, p0, w, int(max_nfev))
                bundles.append(b)
                if show_all_weights_in_summary or w == primary_weight:
                    summary_rows.append(_summary_row(file_name, df, b, float(low_freq_cutoff)))
            except Exception as exc:
                errors.append(f"{file_name}, weight={w}: fit failed — {exc}")
            finally:
                done_jobs += 1
                progress.progress(min(done_jobs / total_jobs, 1.0), text=f"Fitted {done_jobs}/{total_jobs} jobs")

        if bundles:
            bundles_by_file[file_name] = bundles

    progress.empty()

    if errors:
        with st.expander(f"Warnings / failed files ({len(errors)})", expanded=True):
            for msg in errors:
                st.warning(msg)

    if not summary_rows:
        st.error("No successful fits to display.")
        return

    summary_df = pd.DataFrame(summary_rows)

    st.subheader("Batch summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files uploaded", str(len(data_files)))
    c2.metric("Successful files", str(len(bundles_by_file)))
    c3.metric("Summary rows", str(len(summary_df)))
    c4.metric("Primary weight", primary_weight)

    concise_cols = [
        "file",
        "weight",
        "success",
        "points",
        "f_min_Hz",
        "f_max_Hz",
        "RMSE_Z_ohm",
        "low_f_bias_ohm",
        "R_total_span_ohm",
        "arc3_height_ohm",
        "fusion_2_3",
        "note",
    ]
    st.dataframe(
        _format_summary_table(summary_df[concise_cols]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download batch summary CSV",
        summary_df.to_csv(index=False),
        file_name="eis_batch_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Detailed preview")

    selectable_files = list(bundles_by_file.keys())
    selected_file = st.selectbox("Choose one file to preview", selectable_files, key="batch_preview_file")
    df = dfs_by_file[selected_file]
    bundles = bundles_by_file[selected_file]
    primary = _primary_bundle(bundles, primary_weight)

    _display_metric_strip(primary, df, float(low_freq_cutoff))

    tab_plot, tab_params, tab_metrics, tab_data, tab_downloads = st.tabs(
        ["Plot", "Parameters", "Arc/fusion", "Data", "Downloads"]
    )

    with tab_plot:
        st.plotly_chart(
            nyquist_figure(df, bundles, show_low_freq_labels=show_low_freq_labels),
            use_container_width=True,
        )
        with st.expander("Low-frequency zoom", expanded=False):
            st.plotly_chart(
                low_freq_zoom_figure(df, bundles, float(low_freq_cutoff)),
                use_container_width=True,
            )

    with tab_params:
        st.markdown(f"**Primary weight = `{primary.weight}`**")
        st.dataframe(primary.params_df, use_container_width=True, hide_index=True)
        if len(bundles) > 1:
            with st.expander("Other weighting results", expanded=False):
                for b in bundles:
                    if b is primary:
                        continue
                    st.markdown(f"**weight = `{b.weight}`**")
                    st.dataframe(b.params_df, use_container_width=True, hide_index=True)

    with tab_metrics:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Arc geometry descriptors")
            st.dataframe(primary.arc_df, use_container_width=True, hide_index=True)
        with c2:
            st.caption("Fusion descriptors")
            st.dataframe(primary.fusion_df, use_container_width=True, hide_index=True)

    with tab_data:
        preview_cols = [
            "freq_hz",
            "z_real_ohm",
            "minus_z_imag_ohm",
            "fit_z_real_ohm",
            "fit_minus_z_imag_ohm",
        ]
        st.dataframe(primary.curve_df[preview_cols], use_container_width=True, hide_index=True)

    with tab_downloads:
        file_stem = Path(selected_file).stem.replace(" ", "_")
        _downloads_for_bundle(primary, file_stem)
        st.download_button(
            "Download full batch outputs as ZIP",
            _make_batch_zip(summary_df, bundles_by_file),
            file_name="eis_batch_outputs.zip",
            mime="application/zip",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# App entry point
# -----------------------------------------------------------------------------


def render_placeholder_page(title: str, description: str) -> None:
    st.title(title)
    st.info(description)
    st.markdown(
        """
        This page is reserved for a future module. The dashboard structure is already ready:
        add a new `render_..._page()` function and connect it in the sidebar selector.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Battery Data Analysis",
        page_icon="🔋",
        layout="wide",
    )

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
        index=0,
    )

    if tool == "EIS Fit":
        render_eis_fit_page()
    elif tool == "EIS Fit Batch":
        render_eis_fit_batch_page()
    elif tool == "Cycling Analysis":
        render_placeholder_page("Cycling Analysis", "Cycling capacity/CE/voltage profile analysis will be added here.")
    elif tool == "Stripping Overpotential":
        render_placeholder_page("Stripping Overpotential", "Li/Na stripping overpotential analysis will be added here.")
    elif tool == "dQ/dV Analysis":
        render_placeholder_page("dQ/dV Analysis", "Differential capacity analysis will be added here.")


if __name__ == "__main__":
    main()
