#!/usr/bin/env python3
"""
Batch V-Q profile analysis for battery cycling Excel files.

This script keeps the same core data-processing logic as the provided example:
1. Read the active material area from the `test` sheet.
2. Read cycle, step, capacity, and voltage data from the `record` sheet.
3. Calculate areal capacity as Capacity(mAh) / area.
4. Extract selected charge/discharge cycles and plot Voltage vs. areal capacity.

Main differences from the original single-file style script:
- The data root folder is provided from the command line.
- Each top-level folder under the root is treated as one sample and assigned one color.
- One figure is generated for each source file so that files inside the same repeat are not overlaid.
- Raw plotting data are exported by sample and repeat.
- Summary tables are exported by sample and repeat.

Expected folder layout, flexible examples:

    root/
      Sample_A/
        Repeat_1/file.xlsx
        Repeat_2/file.xlsx
      Sample_B/
        file.xlsx

If a file is directly inside the sample folder, the file stem is used as the repeat name.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


# A muted, colorblind-friendly palette. The order is stable and works well for papers.
SAMPLE_PALETTE: list[tuple[float, float, float]] = [
    (0.121, 0.466, 0.705),  # blue
    (1.000, 0.498, 0.054),  # orange
    (0.172, 0.627, 0.172),  # green
    (0.839, 0.153, 0.157),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.811),  # cyan
]

DEFAULT_COLOR = (0.45, 0.45, 0.45)
REQUIRED_RECORD_COLUMNS = ["Cycle Index", "Step Type", "Capacity(mAh)", "Voltage(V)"]


@dataclass(frozen=True)
class FileMeta:
    """Metadata inferred from one Excel file path."""

    file_path: Path
    sample_folder: str
    sample_name: str
    repeat_name: str


@dataclass
class ProcessedFile:
    """Processed output from one Excel file."""

    meta: FileMeta
    status: str
    note: str
    area_cm2: float | None = None
    max_record_cycle: int | None = None
    initial_discharge_capacity: float | None = None
    raw_plot_data: pd.DataFrame | None = None
    cycle_summary: pd.DataFrame | None = None
    file_summary: dict | None = None


# -----------------------------------------------------------------------------
# Path and naming helpers
# -----------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Return a safe filename while preserving readable sample/repeat names."""
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unnamed"



def infer_sample_name(sample_folder: str) -> str:
    """
    Infer a cleaner sample name from the top-level folder.

    If the folder contains a pattern such as V1S2, use that. Otherwise use the
    full top-level folder name.
    """
    match = re.search(r"(V.*?S.*?)(?:_|$|\\|/)", sample_folder, re.IGNORECASE)
    return match.group(1).strip() if match else sample_folder



def infer_file_meta(file_path: Path, root_dir: Path) -> FileMeta:
    """Infer sample and repeat names from the file location."""
    rel_parts = file_path.relative_to(root_dir).parts

    if len(rel_parts) >= 2:
        sample_folder = rel_parts[0]
    else:
        sample_folder = root_dir.name

    sample_name = infer_sample_name(sample_folder)

    # root/sample/repeat/file.xlsx -> repeat
    # root/sample/file.xlsx        -> file stem
    if len(rel_parts) >= 3:
        repeat_name = rel_parts[1]
    else:
        repeat_name = file_path.stem

    return FileMeta(
        file_path=file_path,
        sample_folder=sample_folder,
        sample_name=sample_name,
        repeat_name=repeat_name,
    )



def find_excel_files(root_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Find Excel files under the root directory, excluding temporary files."""
    files: list[Path] = []
    output_dir_resolved = output_dir.resolve() if output_dir else None

    for path in root_dir.rglob("*.xlsx"):
        if path.name.startswith("~$"):
            continue
        if output_dir_resolved and output_dir_resolved in path.resolve().parents:
            continue
        files.append(path)

    return sorted(files)


# -----------------------------------------------------------------------------
# Data extraction helpers
# -----------------------------------------------------------------------------


def extract_area_cm2(file_path: Path, sheet_name: str = "test") -> float:
    """
    Extract active material area from the `test` sheet.

    The function searches the first 30 rows for a cell containing
    "active material" and then reads the numeric value two columns to the right,
    matching the logic in the provided example script.
    """
    df_test = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=30)

    for r in range(len(df_test)):
        for c in range(max(0, len(df_test.columns) - 2)):
            cell_value = str(df_test.iloc[r, c]).strip().lower()
            if "active material" not in cell_value:
                continue

            area_value = str(df_test.iloc[r, c + 2])
            match = re.search(r"([0-9]*\.?[0-9]+)", area_value)
            if match:
                area = float(match.group(1))
                if area > 0:
                    return area

    raise ValueError("Could not find a valid active material area in the test sheet.")



def read_record_sheet(file_path: Path, sheet_name: str = "record") -> pd.DataFrame:
    """Read and clean the record sheet."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    missing = [col for col in REQUIRED_RECORD_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in record sheet: {missing}")

    df = df.copy()
    df["Step Type"] = df["Step Type"].astype(str).str.strip()
    df["Cycle Index"] = pd.to_numeric(df["Cycle Index"], errors="coerce")
    df["Capacity(mAh)"] = pd.to_numeric(df["Capacity(mAh)"], errors="coerce")
    df["Voltage(V)"] = pd.to_numeric(df["Voltage(V)"], errors="coerce")

    df = df.dropna(subset=["Cycle Index", "Capacity(mAh)", "Voltage(V)"])
    df["Cycle Index"] = df["Cycle Index"].astype(int)
    return df



def parse_cycle_list(value: str | None) -> list[int] | None:
    """Parse a comma-separated cycle list, for example: '3,23,43,63'."""
    if value is None or not str(value).strip():
        return None

    cycles = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not cycles:
        return None
    return cycles



def build_default_cycles(start: int, step: int, max_cycle: int) -> list[int]:
    """Build the default cycle sequence used for V-Q plotting."""
    if start > max_cycle:
        return []
    return list(range(start, max_cycle + 1, step))



def get_step_data(df: pd.DataFrame, cycle: int, step_type: str) -> pd.DataFrame:
    """Return rows for one cycle and one step type."""
    return df[(df["Cycle Index"] == cycle) & (df["Step Type"] == step_type)].copy()


# -----------------------------------------------------------------------------
# Color and plotting helpers
# -----------------------------------------------------------------------------


def assign_sample_colors(sample_names: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    """Assign a stable color to each sample."""
    color_map: dict[str, tuple[float, float, float]] = {}
    for idx, sample_name in enumerate(sorted(set(sample_names))):
        color_map[sample_name] = SAMPLE_PALETTE[idx % len(SAMPLE_PALETTE)]
    return color_map



def faded_color(
    base_rgb: tuple[float, float, float],
    index: int,
    total: int,
    max_fade: float = 0.72,
) -> tuple[float, float, float]:
    """Fade a base color toward white to separate cycles within one sample."""
    if total <= 1:
        fade = 0.0
    else:
        fade = min(max_fade, max_fade * index / (total - 1))

    r, g, b = base_rgb
    return (
        r + (1.0 - r) * fade,
        g + (1.0 - g) * fade,
        b + (1.0 - b) * fade,
    )



def choose_auto_xmax(max_capacity: float) -> float:
    """Choose a clean x-axis maximum based on the plotted capacity range."""
    if not math.isfinite(max_capacity) or max_capacity <= 0:
        return 5.0
    if max_capacity <= 2.0:
        return 2.0
    if max_capacity <= 5.0:
        return 5.0
    if max_capacity <= 7.5:
        return 7.5
    return math.ceil(max_capacity * 1.05)



def apply_axis_style(ax: plt.Axes) -> None:
    """Apply a clean scientific-plot style."""
    ax.tick_params(axis="both", which="major", direction="in", labelsize=14, length=6, width=1.4)
    ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)



def save_file_plot(
    sample_name: str,
    repeat_name: str,
    source_file: str,
    file_raw: pd.DataFrame,
    sample_color: tuple[float, float, float],
    figures_dir: Path,
    voltage_min: float,
    voltage_max: float,
    x_max: str,
    dpi: int,
) -> Path | None:
    """Create one V-Q profile figure for one processed source file."""
    if file_raw.empty:
        return None

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cycles = sorted(file_raw["cycle_index"].dropna().astype(int).unique())
    cycle_to_color = {
        cycle: faded_color(sample_color, idx, len(cycles)) for idx, cycle in enumerate(cycles)
    }

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    for (_, cycle, step_type), group in file_raw.groupby(
        ["source_file", "cycle_index", "step_type"], sort=True
    ):
        group = group.sort_values("point_index")
        ax.plot(
            group["areal_capacity_mAh_cm2"],
            group["voltage_V"],
            color=cycle_to_color[int(cycle)],
            linestyle="-",
            linewidth=2.1,
            alpha=0.90,
        )

    legend_lines = [Line2D([0], [0], color=cycle_to_color[c], lw=2.4) for c in cycles]
    legend_labels = [str(c) for c in cycles]

    ax.set_xlabel("Capacity (mAh cm$^{-2}$)", fontsize=17)
    ax.set_ylabel("Voltage (V)", fontsize=17)
    ax.set_title(f"{sample_name} - {repeat_name} - {source_file}", fontsize=16, pad=10)
    ax.set_ylim(voltage_min, voltage_max)

    if x_max == "auto":
        xmax_value = choose_auto_xmax(float(file_raw["areal_capacity_mAh_cm2"].max()))
    else:
        xmax_value = float(x_max)
    ax.set_xlim(-0.05 * xmax_value, xmax_value)

    apply_axis_style(ax)

    ax.legend(
        legend_lines,
        legend_labels,
        loc="best",
        fontsize=10.5,
        frameon=False,
        title="Cycle Index",
        title_fontsize=11.5,
    )

    fig.tight_layout()

    sample_figures_dir = figures_dir / sanitize_filename(sample_name)
    sample_figures_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"{sanitize_filename(sample_name)}__{sanitize_filename(repeat_name)}__"
        f"{sanitize_filename(Path(source_file).stem)}_VQ_profiles"
    )
    output_png = sample_figures_dir / f"{base}.png"
    output_pdf = sample_figures_dir / f"{base}.pdf"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_png


# -----------------------------------------------------------------------------
# Processing logic
# -----------------------------------------------------------------------------


def process_one_file(
    meta: FileMeta,
    cycles_override: list[int] | None,
    cycle_start: int,
    cycle_step: int,
    charge_step: str,
    discharge_step: str,
    retention_cutoff: float | None,
    stop_at_retention_cutoff: bool,
) -> ProcessedFile:
    """Process one Excel file and return raw plotting data plus summaries."""
    try:
        area_cm2 = extract_area_cm2(meta.file_path)
        record = read_record_sheet(meta.file_path)
        record["areal_capacity_mAh_cm2"] = record["Capacity(mAh)"] / area_cm2

        if record.empty:
            raise ValueError("The record sheet has no valid numeric rows.")

        max_record_cycle = int(record["Cycle Index"].max())
        cycles = cycles_override or build_default_cycles(cycle_start, cycle_step, max_record_cycle)

        initial_discharge = get_step_data(record, cycle_start, discharge_step)
        if initial_discharge.empty:
            raise ValueError(f"Could not find {discharge_step!r} data for cycle {cycle_start}.")

        initial_discharge_capacity = float(initial_discharge["areal_capacity_mAh_cm2"].max())
        if not math.isfinite(initial_discharge_capacity) or initial_discharge_capacity <= 0:
            raise ValueError(f"Invalid initial discharge capacity for cycle {cycle_start}.")

        raw_frames: list[pd.DataFrame] = []
        cycle_summary_rows: list[dict] = []
        cutoff_trigger_cycle: int | None = None

        for cycle in cycles:
            charge = get_step_data(record, cycle, charge_step)
            discharge = get_step_data(record, cycle, discharge_step)

            charge_cap = float(charge["areal_capacity_mAh_cm2"].max()) if not charge.empty else float("nan")
            discharge_cap = (
                float(discharge["areal_capacity_mAh_cm2"].max()) if not discharge.empty else float("nan")
            )
            retention = (
                discharge_cap / initial_discharge_capacity * 100.0
                if math.isfinite(discharge_cap) and initial_discharge_capacity > 0
                else float("nan")
            )

            include_in_plot = not discharge.empty
            if (
                include_in_plot
                and retention_cutoff is not None
                and math.isfinite(retention)
                and retention < retention_cutoff
            ):
                cutoff_trigger_cycle = cycle
                include_in_plot = False

            cycle_summary_rows.append(
                {
                    "sample": meta.sample_name,
                    "repeat": meta.repeat_name,
                    "source_file": meta.file_path.name,
                    "source_path": str(meta.file_path),
                    "area_cm2": area_cm2,
                    "cycle_index": cycle,
                    "charge_capacity_mAh_cm2": charge_cap,
                    "discharge_capacity_mAh_cm2": discharge_cap,
                    "retention_percent_vs_initial": retention,
                    "charge_voltage_min_V": float(charge["Voltage(V)"].min()) if not charge.empty else float("nan"),
                    "charge_voltage_max_V": float(charge["Voltage(V)"].max()) if not charge.empty else float("nan"),
                    "discharge_voltage_min_V": float(discharge["Voltage(V)"].min()) if not discharge.empty else float("nan"),
                    "discharge_voltage_max_V": float(discharge["Voltage(V)"].max()) if not discharge.empty else float("nan"),
                    "included_in_plot": include_in_plot,
                }
            )

            if not include_in_plot:
                if cutoff_trigger_cycle is not None and stop_at_retention_cutoff:
                    break
                continue

            for step_label, step_df in (("charge", charge), ("discharge", discharge)):
                if step_df.empty:
                    continue

                out = step_df.copy()
                out = out.reset_index(drop=True).reset_index(names="point_index")
                out = out.rename(
                    columns={
                        "Cycle Index": "cycle_index",
                        "Step Type": "step_type",
                        "Capacity(mAh)": "capacity_mAh",
                        "Voltage(V)": "voltage_V",
                    }
                )
                out.insert(0, "sample", meta.sample_name)
                out.insert(1, "repeat", meta.repeat_name)
                out.insert(2, "source_file", meta.file_path.name)
                out.insert(3, "source_path", str(meta.file_path))
                out.insert(4, "step_label", step_label)
                out.insert(5, "area_cm2", area_cm2)
                out = out[
                    [
                        "sample",
                        "repeat",
                        "source_file",
                        "source_path",
                        "area_cm2",
                        "cycle_index",
                        "step_label",
                        "step_type",
                        "point_index",
                        "capacity_mAh",
                        "areal_capacity_mAh_cm2",
                        "voltage_V",
                    ]
                ]
                raw_frames.append(out)

        raw_plot_data = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        cycle_summary = pd.DataFrame(cycle_summary_rows)

        plotted_cycle_summary = cycle_summary[cycle_summary["included_in_plot"]].copy()
        if plotted_cycle_summary.empty:
            note = "No cycles passed the plotting criteria."
            status = "skipped"
        else:
            note = "OK"
            status = "ok"

        if cutoff_trigger_cycle is not None:
            note = (
                f"Stopped before cycle {cutoff_trigger_cycle} because retention fell "
                f"below {retention_cutoff:.1f}%."
            )

        file_summary = {
            "sample": meta.sample_name,
            "repeat": meta.repeat_name,
            "source_file": meta.file_path.name,
            "source_path": str(meta.file_path),
            "status": status,
            "note": note,
            "area_cm2": area_cm2,
            "max_record_cycle": max_record_cycle,
            "initial_cycle_index": cycle_start,
            "initial_discharge_capacity_mAh_cm2": initial_discharge_capacity,
            "n_selected_cycles_checked": len(cycle_summary),
            "n_cycles_included_in_plot": int(cycle_summary["included_in_plot"].sum()) if not cycle_summary.empty else 0,
            "first_plotted_cycle": int(plotted_cycle_summary["cycle_index"].min())
            if not plotted_cycle_summary.empty
            else None,
            "last_plotted_cycle": int(plotted_cycle_summary["cycle_index"].max())
            if not plotted_cycle_summary.empty
            else None,
            "last_plotted_discharge_capacity_mAh_cm2": float(
                plotted_cycle_summary.sort_values("cycle_index")["discharge_capacity_mAh_cm2"].iloc[-1]
            )
            if not plotted_cycle_summary.empty
            else None,
            "last_plotted_retention_percent": float(
                plotted_cycle_summary.sort_values("cycle_index")["retention_percent_vs_initial"].iloc[-1]
            )
            if not plotted_cycle_summary.empty
            else None,
            "raw_plot_points": len(raw_plot_data),
            "cutoff_trigger_cycle": cutoff_trigger_cycle,
        }

        return ProcessedFile(
            meta=meta,
            status=status,
            note=note,
            area_cm2=area_cm2,
            max_record_cycle=max_record_cycle,
            initial_discharge_capacity=initial_discharge_capacity,
            raw_plot_data=raw_plot_data,
            cycle_summary=cycle_summary,
            file_summary=file_summary,
        )

    except Exception as exc:
        file_summary = {
            "sample": meta.sample_name,
            "repeat": meta.repeat_name,
            "source_file": meta.file_path.name,
            "source_path": str(meta.file_path),
            "status": "error",
            "note": str(exc),
            "area_cm2": None,
            "max_record_cycle": None,
            "initial_cycle_index": cycle_start,
            "initial_discharge_capacity_mAh_cm2": None,
            "n_selected_cycles_checked": 0,
            "n_cycles_included_in_plot": 0,
            "first_plotted_cycle": None,
            "last_plotted_cycle": None,
            "last_plotted_discharge_capacity_mAh_cm2": None,
            "last_plotted_retention_percent": None,
            "raw_plot_points": 0,
            "cutoff_trigger_cycle": None,
        }
        return ProcessedFile(meta=meta, status="error", note=str(exc), file_summary=file_summary)



def write_outputs(
    processed: list[ProcessedFile],
    output_dir: Path,
    sample_colors: dict[str, tuple[float, float, float]],
    voltage_min: float,
    voltage_max: float,
    x_max: str,
    dpi: int,
) -> None:
    """Write raw data, summaries, and per-file figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_plot_data_by_sample_repeat"
    figures_dir = output_dir / "figures_by_file"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_all = pd.concat(
        [item.raw_plot_data for item in processed if item.raw_plot_data is not None and not item.raw_plot_data.empty],
        ignore_index=True,
    ) if any(item.raw_plot_data is not None and not item.raw_plot_data.empty for item in processed) else pd.DataFrame()

    cycle_summary_all = pd.concat(
        [item.cycle_summary for item in processed if item.cycle_summary is not None and not item.cycle_summary.empty],
        ignore_index=True,
    ) if any(item.cycle_summary is not None and not item.cycle_summary.empty for item in processed) else pd.DataFrame()

    file_summary_all = pd.DataFrame([item.file_summary for item in processed if item.file_summary is not None])

    raw_all.to_csv(output_dir / "plot_raw_data_all.csv", index=False)
    cycle_summary_all.to_csv(output_dir / "cycle_summary_by_sample_repeat.csv", index=False)
    file_summary_all.to_csv(output_dir / "summary_by_file.csv", index=False)

    if not file_summary_all.empty:
        numeric_summary_cols = [
            "area_cm2",
            "max_record_cycle",
            "initial_discharge_capacity_mAh_cm2",
            "n_cycles_included_in_plot",
            "first_plotted_cycle",
            "last_plotted_cycle",
            "last_plotted_retention_percent",
            "raw_plot_points",
        ]
        for col in numeric_summary_cols:
            if col in file_summary_all.columns:
                file_summary_all[col] = pd.to_numeric(file_summary_all[col], errors="coerce")

        summary_by_repeat = (
            file_summary_all
            .assign(
                ok_file=lambda d: d["status"].eq("ok"),
                error_file=lambda d: d["status"].eq("error"),
            )
            .groupby(["sample", "repeat"], as_index=False)
            .agg(
                n_files=("source_file", "count"),
                n_ok_files=("ok_file", "sum"),
                n_error_files=("error_file", "sum"),
                mean_area_cm2=("area_cm2", "mean"),
                max_record_cycle=("max_record_cycle", "max"),
                mean_initial_discharge_capacity_mAh_cm2=("initial_discharge_capacity_mAh_cm2", "mean"),
                total_cycles_included_in_plot=("n_cycles_included_in_plot", "sum"),
                total_raw_plot_points=("raw_plot_points", "sum"),
                first_plotted_cycle=("first_plotted_cycle", "min"),
                last_plotted_cycle=("last_plotted_cycle", "max"),
                mean_last_plotted_retention_percent=("last_plotted_retention_percent", "mean"),
            )
        )
    else:
        summary_by_repeat = pd.DataFrame()

    summary_by_repeat.to_csv(output_dir / "summary_by_sample_repeat.csv", index=False)

    if not raw_all.empty:
        # aggregated raw data by sample/repeat
        for (sample, repeat), group in raw_all.groupby(["sample", "repeat"], sort=True):
            sample_dir = raw_dir / sanitize_filename(sample)
            sample_dir.mkdir(parents=True, exist_ok=True)
            output_csv = sample_dir / f"{sanitize_filename(repeat)}_plot_raw_data.csv"
            group.to_csv(output_csv, index=False)

        # individual raw data by file to match the new figure granularity
        raw_file_dir = output_dir / "raw_plot_data_by_file"
        raw_file_dir.mkdir(parents=True, exist_ok=True)
        for (sample, repeat, source_file), group in raw_all.groupby(["sample", "repeat", "source_file"], sort=True):
            sample_dir = raw_file_dir / sanitize_filename(sample)
            sample_dir.mkdir(parents=True, exist_ok=True)
            output_csv = sample_dir / (
                f"{sanitize_filename(repeat)}__{sanitize_filename(Path(source_file).stem)}_plot_raw_data.csv"
            )
            group.to_csv(output_csv, index=False)

    if not raw_all.empty:
        for (sample, repeat, source_file), file_raw in raw_all.groupby(["sample", "repeat", "source_file"], sort=True):
            save_file_plot(
                sample_name=sample,
                repeat_name=repeat,
                source_file=source_file,
                file_raw=file_raw,
                sample_color=sample_colors.get(sample, DEFAULT_COLOR),
                figures_dir=figures_dir,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
                x_max=x_max,
                dpi=dpi,
            )

    try:
        with pd.ExcelWriter(output_dir / "vq_analysis_outputs.xlsx") as writer:
            file_summary_all.to_excel(writer, sheet_name="summary_by_file", index=False)
            summary_by_repeat.to_excel(writer, sheet_name="summary_by_repeat", index=False)
            cycle_summary_all.to_excel(writer, sheet_name="cycle_summary", index=False)
            if len(raw_all) <= 1_000_000:
                raw_all.to_excel(writer, sheet_name="plot_raw_data", index=False)
    except Exception as exc:
        print(f"Warning: could not write Excel workbook: {exc}")


# -----------------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-process battery V-Q profiles from Excel cycling data."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing one top-level folder per sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root_dir>/VQ_analysis_outputs.",
    )
    parser.add_argument(
        "--cycle-start",
        type=int,
        default=3,
        help="First cycle used for plotting and initial capacity. Default: 3.",
    )
    parser.add_argument(
        "--cycle-step",
        type=int,
        default=20,
        help="Cycle interval when --cycle-list is not provided. Default: 20.",
    )
    parser.add_argument(
        "--cycle-list",
        type=str,
        default=None,
        help="Comma-separated cycle list, for example '3,23,43,63'. Overrides --cycle-step.",
    )
    parser.add_argument(
        "--charge-step",
        type=str,
        default="CC Chg",
        help="Step Type value for charge curves. Default: 'CC Chg'.",
    )
    parser.add_argument(
        "--discharge-step",
        type=str,
        default="CC DChg",
        help="Step Type value for discharge curves. Default: 'CC DChg'.",
    )
    parser.add_argument(
        "--retention-cutoff",
        type=float,
        default=80.0,
        help="Stop plotting before cycles below this retention percentage. Use -1 to disable. Default: 80.",
    )
    parser.add_argument(
        "--keep-checking-after-cutoff",
        action="store_true",
        help="If set, cycles below the retention cutoff are skipped but later cycles are still checked.",
    )
    parser.add_argument(
        "--voltage-min",
        type=float,
        default=2.5,
        help="Y-axis minimum voltage. Default: 2.5.",
    )
    parser.add_argument(
        "--voltage-max",
        type=float,
        default=4.5,
        help="Y-axis maximum voltage. Default: 4.5.",
    )
    parser.add_argument(
        "--x-max",
        type=str,
        default="auto",
        help="X-axis maximum. Use 'auto' or a number such as 5 or 7.5. Default: auto.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution. Default: 300.",
    )
    return parser



def main() -> None:
    args = build_parser().parse_args()

    root_dir = args.root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist or is not a directory: {root_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root_dir / "VQ_analysis_outputs"
    )

    if args.cycle_step <= 0:
        raise ValueError("--cycle-step must be positive.")

    if args.x_max != "auto":
        try:
            x_value = float(args.x_max)
        except ValueError as exc:
            raise ValueError("--x-max must be 'auto' or a numeric value.") from exc
        if x_value <= 0:
            raise ValueError("--x-max must be positive.")

    cycles_override = parse_cycle_list(args.cycle_list)
    retention_cutoff = None if args.retention_cutoff < 0 else args.retention_cutoff
    stop_at_retention_cutoff = not args.keep_checking_after_cutoff

    files = find_excel_files(root_dir, output_dir=output_dir)
    print(f"Found {len(files)} Excel files under: {root_dir}")
    if not files:
        print("No Excel files found. Nothing to process.")
        return

    metas = [infer_file_meta(path, root_dir) for path in files]
    sample_colors = assign_sample_colors(meta.sample_name for meta in metas)

    processed: list[ProcessedFile] = []
    for idx, meta in enumerate(metas, start=1):
        result = process_one_file(
            meta=meta,
            cycles_override=cycles_override,
            cycle_start=args.cycle_start,
            cycle_step=args.cycle_step,
            charge_step=args.charge_step,
            discharge_step=args.discharge_step,
            retention_cutoff=retention_cutoff,
            stop_at_retention_cutoff=stop_at_retention_cutoff,
        )
        processed.append(result)
        print(f"[{idx}/{len(metas)}] {result.status.upper()}: {meta.sample_name} / {meta.repeat_name} / {meta.file_path.name} - {result.note}")

    write_outputs(
        processed=processed,
        output_dir=output_dir,
        sample_colors=sample_colors,
        voltage_min=args.voltage_min,
        voltage_max=args.voltage_max,
        x_max=args.x_max,
        dpi=args.dpi,
    )

    print("\nDone.")
    print(f"Figures: {output_dir / 'figures_by_file'}")
    print(f"Raw data: {output_dir / 'plot_raw_data_all.csv'}")
    print(f"Cycle summary: {output_dir / 'cycle_summary_by_sample_repeat.csv'}")
    print(f"Sample/repeat summary: {output_dir / 'summary_by_sample_repeat.csv'}")


if __name__ == "__main__":
    main()
