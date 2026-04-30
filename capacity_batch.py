#!/usr/bin/env python3
"""
Batch capacity processing and plotting for battery cycling data.

Directory structure expected:

root_directory/
    Sample_A/
        file1.xlsx
        file2.xlsx
        ...
    Sample_B/
        file3.xlsx
        file4.xlsx
        ...

Each folder directly under the root directory is treated as one sample.
For each sample, this script:

1. Recursively searches for Excel files.
2. Reads the "cycle" sheet from each file.
3. Extracts discharge capacity and coulombic efficiency.
4. Calculates capacity retention relative to the first valid discharge capacity.
5. Generates one plot per sample.
6. Exports the exact data used for plotting as a CSV file.

Default required columns:
    - DChg. Cap.(mAh)
    - Chg.-DChg. Eff(%)
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


DEFAULT_CAPACITY_COL = "DChg. Cap.(mAh)"
DEFAULT_EFFICIENCY_COL = "Chg.-DChg. Eff(%)"
DEFAULT_SHEET_NAME = "cycle"


def sanitize_filename(name: str) -> str:
    """
    Convert a sample name into a safe filename.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name.strip("_")


def get_sample_folders(root_dir: Path, output_dir: Path) -> list[Path]:
    """
    Return all valid sample folders directly under the root directory.
    """
    folders = []

    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue

        if item.name.lower() == "ignore":
            continue

        if item.resolve() == output_dir.resolve():
            continue

        folders.append(item)

    return folders


def build_color_map(sample_folders: list[Path]) -> dict[str, tuple]:
    """
    Assign one color to each sample folder.

    The palette combines several qualitative Matplotlib palettes so that
    different samples are visually distinguishable.
    """
    palette = (
        list(plt.cm.Set2.colors)
        + list(plt.cm.Dark2.colors)
        + list(plt.cm.tab20.colors)
        + list(plt.cm.tab20b.colors)
        + list(plt.cm.tab20c.colors)
    )

    color_map = {}

    for i, folder in enumerate(sample_folders):
        color_map[folder.name] = palette[i % len(palette)]

    return color_map


def find_excel_files(sample_dir: Path) -> list[Path]:
    """
    Recursively find Excel files in one sample directory.
    """
    excel_files = []

    for path in sample_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() in [".xlsx", ".xls"]:
            excel_files.append(path)

    return sorted(excel_files)


def detect_cycle_column(df: pd.DataFrame) -> str | None:
    """
    Try to detect a cycle index column.

    If no suitable column is found, the script will generate cycle indices
    automatically.
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


def read_one_cycling_file(
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
    Read one Excel cycling file and return a tidy DataFrame.

    The returned DataFrame contains the data used for plotting:
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
        print(f"  ⚠️ Could not read {file_path.name}: {exc}")
        return None

    missing_cols = [
        col for col in [capacity_col, efficiency_col] if col not in df.columns
    ]

    if missing_cols:
        print(f"  ⚠️ Skipping {file_path.name}: missing columns {missing_cols}")
        return None

    cycle_col = detect_cycle_column(df)

    data = df.iloc[skip_initial_rows:].copy()

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
        print(f"  ⚠️ Skipping {file_path.name}: no valid numeric data")
        return None

    initial_capacity = capacity.iloc[0]

    if initial_capacity == 0 or pd.isna(initial_capacity):
        print(f"  ⚠️ Skipping {file_path.name}: invalid initial capacity")
        return None

    capacity_retention = capacity / initial_capacity * 100

    result = pd.DataFrame(
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
        result = result[
            result["capacity_retention_percent"] >= min_capacity_retention
        ].reset_index(drop=True)

    if result.empty:
        print(
            f"  ⚠️ Skipping {file_path.name}: no data after retention filtering"
        )
        return None

    return result


def auto_x_limit(max_cycle: float) -> int:
    """
    Choose a clean x-axis upper limit based on the maximum cycle index.
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


def plot_one_sample(
    sample_name: str,
    plot_df: pd.DataFrame,
    color: tuple,
    output_path: Path,
    capacity_ylim: tuple[float, float],
    efficiency_ylim: tuple[float, float],
    dpi: int,
) -> None:
    """
    Generate one capacity-retention / coulombic-efficiency plot for one sample.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax1 = plt.subplots(figsize=(8.5, 5.8))
    ax2 = ax1.twinx()

    max_cycle = plot_df["cycle_index"].max()

    for source_file, group in plot_df.groupby("source_file", sort=True):
        group = group.sort_values("cycle_index")

        label = Path(source_file).stem

        ax1.scatter(
            group["cycle_index"],
            group["capacity_retention_percent"],
            color=color,
            marker="o",
            s=80,
            alpha=0.95,
            zorder=3,
            label=label,
        )

        ax2.scatter(
            group["cycle_index"],
            group["coulombic_efficiency_percent"],
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=80,
            linewidths=1.4,
            alpha=0.95,
            zorder=3,
        )

    ax1.set_xlim(0, auto_x_limit(max_cycle))
    ax1.set_ylim(*capacity_ylim)
    ax2.set_ylim(*efficiency_ylim)

    ax1.set_xlabel("Cycle Index", fontsize=18)
    ax1.set_ylabel("Capacity Retention (%)", fontsize=18)
    ax2.set_ylabel("Coulombic Efficiency (%)", fontsize=18)

    ax1.set_title(sample_name, fontsize=18, pad=12)

    for ax in [ax1, ax2]:
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            labelsize=14,
            length=6,
            width=1.5,
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

    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1.16, 0.5),
        title="Files",
        fontsize=10,
        title_fontsize=12,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_one_sample(
    sample_dir: Path,
    root_dir: Path,
    output_dir: Path,
    color: tuple,
    args: argparse.Namespace,
) -> None:
    """
    Process all cycling files under one sample directory.
    """
    sample_name = sample_dir.name
    print(f"\n📁 Processing sample: {sample_name}")

    excel_files = find_excel_files(sample_dir)

    if not excel_files:
        print(f"  ⚠️ No Excel files found in {sample_name}")
        return

    all_data = []

    for file_path in excel_files:
        one_file_df = read_one_cycling_file(
            file_path=file_path,
            sample_name=sample_name,
            root_dir=root_dir,
            sheet_name=args.sheet_name,
            capacity_col=args.capacity_col,
            efficiency_col=args.efficiency_col,
            skip_initial_rows=args.skip_initial_rows,
            min_capacity_retention=args.min_capacity_retention,
        )

        if one_file_df is not None:
            all_data.append(one_file_df)

    if not all_data:
        print(f"  ⚠️ No valid data found for sample {sample_name}")
        return

    plot_df = pd.concat(all_data, ignore_index=True)

    if args.top_n is not None and args.top_n > 0:
        scores = (
            plot_df.groupby("source_file")["capacity_retention_percent"]
            .sum()
            .sort_values(ascending=False)
        )

        selected_files = scores.head(args.top_n).index
        plot_df = plot_df[plot_df["source_file"].isin(selected_files)].copy()

    safe_sample_name = sanitize_filename(sample_name)
    sample_output_dir = output_dir / safe_sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sample_output_dir / f"{safe_sample_name}_plot_data.csv"
    fig_path = sample_output_dir / f"{safe_sample_name}_capacity_summary.png"

    plot_df.to_csv(csv_path, index=False)

    plot_one_sample(
        sample_name=sample_name,
        plot_df=plot_df,
        color=color,
        output_path=fig_path,
        capacity_ylim=(args.capacity_ymin, args.capacity_ymax),
        efficiency_ylim=(args.efficiency_ymin, args.efficiency_ymax),
        dpi=args.dpi,
    )

    print(f"  ✅ Saved plot: {fig_path}")
    print(f"  ✅ Saved plotting data: {csv_path}")


def save_color_map(
    color_map: dict[str, tuple],
    output_dir: Path,
) -> None:
    """
    Save the sample-to-color mapping as a CSV file.
    """
    rows = []

    for sample_name, color in color_map.items():
        rgb_255 = tuple(int(round(c * 255)) for c in color[:3])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_255)

        rows.append(
            {
                "sample": sample_name,
                "red": rgb_255[0],
                "green": rgb_255[1],
                "blue": rgb_255[2],
                "hex": hex_color,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "sample_color_map.csv", index=False)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Batch process battery cycling capacity data. "
            "Each folder directly under the root directory is treated as one sample."
        )
    )

    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing sample folders.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for plots and CSV files. "
            "Default: <root_dir>/capacity_batch_results"
        ),
    )

    parser.add_argument(
        "--sheet-name",
        type=str,
        default=DEFAULT_SHEET_NAME,
        help='Excel sheet name to read. Default: "cycle".',
    )

    parser.add_argument(
        "--capacity-col",
        type=str,
        default=DEFAULT_CAPACITY_COL,
        help=f'Discharge capacity column name. Default: "{DEFAULT_CAPACITY_COL}".',
    )

    parser.add_argument(
        "--efficiency-col",
        type=str,
        default=DEFAULT_EFFICIENCY_COL,
        help=f'Coulombic efficiency column name. Default: "{DEFAULT_EFFICIENCY_COL}".',
    )

    parser.add_argument(
        "--skip-initial-rows",
        type=int,
        default=2,
        help=(
            "Number of initial rows to skip in the cycle sheet. "
            "Default: 2, matching the example logic."
        ),
    )

    parser.add_argument(
        "--min-capacity-retention",
        type=float,
        default=80.0,
        help=(
            "Minimum capacity retention to keep for plotting. "
            "Use a negative value to disable this filter. Default: 80."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Only plot the top N files within each sample, ranked by the sum of "
            "capacity retention. Default: plot all valid files."
        ),
    )

    parser.add_argument(
        "--capacity-ymin",
        type=float,
        default=75.0,
        help="Minimum y-axis value for capacity retention. Default: 75.",
    )

    parser.add_argument(
        "--capacity-ymax",
        type=float,
        default=110.0,
        help="Maximum y-axis value for capacity retention. Default: 110.",
    )

    parser.add_argument(
        "--efficiency-ymin",
        type=float,
        default=90.0,
        help="Minimum y-axis value for coulombic efficiency. Default: 90.",
    )

    parser.add_argument(
        "--efficiency-ymax",
        type=float,
        default=100.5,
        help="Maximum y-axis value for coulombic efficiency. Default: 100.5.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution. Default: 300.",
    )

    args = parser.parse_args()

    if args.min_capacity_retention < 0:
        args.min_capacity_retention = None

    return args


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    if args.output_dir is None:
        output_dir = root_dir / "capacity_batch_results"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_folders = get_sample_folders(root_dir, output_dir)

    if not sample_folders:
        print("⚠️ No sample folders found under the root directory.")
        return

    color_map = build_color_map(sample_folders)
    save_color_map(color_map, output_dir)

    print(f"🔍 Root directory: {root_dir}")
    print(f"📤 Output directory: {output_dir}")
    print(f"🧪 Found {len(sample_folders)} sample folder(s).")

    for sample_dir in sample_folders:
        process_one_sample(
            sample_dir=sample_dir,
            root_dir=root_dir,
            output_dir=output_dir,
            color=color_map[sample_dir.name],
            args=args,
        )

    print("\n🎉 Batch capacity processing completed.")


if __name__ == "__main__":
    main()