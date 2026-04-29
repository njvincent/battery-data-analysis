#!/usr/bin/env python3
"""
Batch EIS fitting for all BioLogic/EC-Lab .mpr files in a directory.

This script uses eis_fit.py in the same directory. It can:
1. Find all .mpr files in a specified folder.
2. Fit each file with the same equivalent circuit:
       R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4)
3. Save per-file outputs:
       *_fit_params.csv
       *_arc_metrics.csv
       *_fusion_metrics.csv
       *_fit_curve.csv
       *_nyquist_fit.png
4. Save combined batch summaries:
       batch_fit_params_summary.csv
       batch_arc_metrics_summary.csv
       batch_fusion_metrics_summary.csv
       batch_status_summary.csv

Example:
    python eis_fit_batch.py "/path/to/eis_folder" \
        --xml "/path/to/zfit_initial.xml" \
        --outdir "/path/to/eis_fit_results" \
        --weight unit \
        --recursive

If you want to fit only files whose names contain C10:
    python eis_fit_batch.py "/path/to/eis_folder" \
        --xml initial.xml \
        --pattern "*C10*.mpr" \
        --outdir results
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# eis_fit.py must be in the same folder as this script, or in PYTHONPATH.
import eis_fit


def safe_name(path: Path) -> str:
    """
    Turn a filename into a safe output prefix.
    Keeps the sample identity readable while avoiding spaces/special chars.
    """
    stem = path.stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_") or "eis_file"


def find_mpr_files(folder: Path, pattern: str = "*.mpr", recursive: bool = False) -> list[Path]:
    if folder.is_file():
        if folder.suffix.lower() != ".mpr":
            raise ValueError(f"Input is a file but not .mpr: {folder}")
        return [folder]

    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    globber = folder.rglob if recursive else folder.glob
    files = sorted([p for p in globber(pattern) if p.is_file() and p.suffix.lower() == ".mpr"])
    return files


def fit_one_mpr(
    mpr_path: Path,
    xml_path: Path | None,
    outdir: Path,
    weight: str,
    overwrite: bool = True,
) -> dict:
    """
    Fit one .mpr file and save all outputs.

    Returns a status dictionary. On failure, returns {"status": "failed", ...}.
    """
    prefix = safe_name(mpr_path)
    sample_outdir = outdir / prefix
    sample_outdir.mkdir(parents=True, exist_ok=True)

    out_prefix = sample_outdir / prefix
    param_path = Path(f"{out_prefix}_fit_params.csv")
    arc_path = Path(f"{out_prefix}_arc_metrics.csv")
    fusion_path = Path(f"{out_prefix}_fusion_metrics.csv")
    curve_path = Path(f"{out_prefix}_fit_curve.csv")
    png_path = Path(f"{out_prefix}_nyquist_fit.png")

    if (not overwrite) and param_path.exists() and curve_path.exists() and png_path.exists():
        return {
            "file": str(mpr_path),
            "sample": prefix,
            "status": "skipped_existing",
            "weight": weight,
            "outdir": str(sample_outdir),
            "error": "",
        }

    try:
        df = eis_fit.read_biologic_mpr_eis(mpr_path)

        p0_dict = eis_fit.read_zfit_xml(xml_path) if xml_path else {}
        p0 = eis_fit.pack_params(p0_dict)

        p_fit, result = eis_fit.fit_eis(df, p0, weight=weight)

        params_df = pd.DataFrame({
            "file": str(mpr_path),
            "sample": prefix,
            "parameter": eis_fit.PARAM_ORDER,
            "initial": p0,
            "fit": p_fit,
        })
        params_df.loc[len(params_df)] = [str(mpr_path), prefix, "cost", np.nan, result.cost]
        params_df.loc[len(params_df)] = [str(mpr_path), prefix, "nfev", np.nan, result.nfev]
        params_df.to_csv(param_path, index=False)

        fmin, fmax = float(df["freq_hz"].min()), float(df["freq_hz"].max())
        metrics_df, fusion_df = eis_fit.arc_metrics(p_fit, fmin, fmax)
        metrics_df.insert(0, "sample", prefix)
        metrics_df.insert(0, "file", str(mpr_path))
        fusion_df.insert(0, "sample", prefix)
        fusion_df.insert(0, "file", str(mpr_path))
        metrics_df.to_csv(arc_path, index=False)
        fusion_df.to_csv(fusion_path, index=False)

        z_fit = eis_fit.circuit_z(p_fit, df["freq_hz"].to_numpy(float))
        curve_df = df.copy()
        curve_df.insert(0, "sample", prefix)
        curve_df.insert(0, "file", str(mpr_path))
        curve_df["fit_z_real_ohm"] = np.real(z_fit)
        curve_df["fit_z_imag_ohm"] = np.imag(z_fit)
        curve_df["fit_minus_z_imag_ohm"] = -np.imag(z_fit)
        curve_df["resid_z_real_ohm"] = curve_df["fit_z_real_ohm"] - curve_df["z_real_ohm"]
        curve_df["resid_minus_z_imag_ohm"] = curve_df["fit_minus_z_imag_ohm"] - curve_df["minus_z_imag_ohm"]
        curve_df.to_csv(curve_path, index=False)

        eis_fit.plot_fit(df, p_fit, png_path)

        return {
            "file": str(mpr_path),
            "sample": prefix,
            "status": "ok",
            "weight": weight,
            "n_points": len(df),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "R1": float(p_fit[0]),
            "R2": float(p_fit[3]),
            "R3": float(p_fit[6]),
            "R4": float(p_fit[9]),
            "a2": float(p_fit[2]),
            "a3": float(p_fit[5]),
            "a4": float(p_fit[8]),
            "s4": float(p_fit[10]),
            "outdir": str(sample_outdir),
            "param_csv": str(param_path),
            "arc_csv": str(arc_path),
            "fusion_csv": str(fusion_path),
            "curve_csv": str(curve_path),
            "plot_png": str(png_path),
            "error": "",
        }

    except Exception as exc:
        return {
            "file": str(mpr_path),
            "sample": prefix,
            "status": "failed",
            "weight": weight,
            "outdir": str(sample_outdir),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def combine_csvs(status_df: pd.DataFrame, outdir: Path):
    """
    Combine per-file parameter/arc/fusion CSVs into batch-level summary CSVs.
    """
    ok = status_df[status_df["status"] == "ok"].copy()

    def read_existing(paths):
        frames = []
        for p in paths:
            if isinstance(p, str) and p and Path(p).exists():
                frames.append(pd.read_csv(p))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    params = read_existing(ok.get("param_csv", []))
    arcs = read_existing(ok.get("arc_csv", []))
    fusion = read_existing(ok.get("fusion_csv", []))

    if not params.empty:
        params.to_csv(outdir / "batch_fit_params_summary.csv", index=False)
    if not arcs.empty:
        arcs.to_csv(outdir / "batch_arc_metrics_summary.csv", index=False)
    if not fusion.empty:
        fusion.to_csv(outdir / "batch_fusion_metrics_summary.csv", index=False)

    status_df.to_csv(outdir / "batch_status_summary.csv", index=False)

    return params, arcs, fusion


def main():
    parser = argparse.ArgumentParser(description="Batch fit all .mpr EIS files in a directory.")
    parser.add_argument("input", help="folder containing .mpr files, or a single .mpr file")
    parser.add_argument("--xml", default=None, help="optional EC-Lab ZFit XML file for initial values")
    parser.add_argument("--outdir", default="eis_fit_results", help="output directory")
    parser.add_argument("--pattern", default="*.mpr", help='filename pattern, e.g. "*.mpr" or "*C10*.mpr"')
    parser.add_argument("--recursive", action="store_true", help="search subfolders recursively")
    parser.add_argument("--weight", default="unit", choices=["unit", "modulus", "sqrt_modulus"])
    parser.add_argument("--no-overwrite", action="store_true", help="skip files that already have output")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve() if args.xml else None
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    files = find_mpr_files(input_path, pattern=args.pattern, recursive=args.recursive)

    if not files:
        print(f"No .mpr files found in {input_path} with pattern {args.pattern}")
        return

    print(f"Found {len(files)} .mpr file(s).")
    print(f"Output directory: {outdir}")
    print(f"Weighting: {args.weight}")

    statuses = []
    for i, mpr_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] fitting {mpr_path.name} ...", flush=True)
        status = fit_one_mpr(
            mpr_path=mpr_path,
            xml_path=xml_path,
            outdir=outdir,
            weight=args.weight,
            overwrite=not args.no_overwrite,
        )
        statuses.append(status)
        if status["status"] == "ok":
            print(f"    ok: cost={status['cost']:.6g}, plot={status['plot_png']}")
        elif status["status"] == "skipped_existing":
            print("    skipped: existing output")
        else:
            print(f"    failed: {status['error']}")

    status_df = pd.DataFrame(statuses)
    combine_csvs(status_df, outdir)

    n_ok = int((status_df["status"] == "ok").sum())
    n_failed = int((status_df["status"] == "failed").sum())
    n_skipped = int((status_df["status"] == "skipped_existing").sum())

    print("\nBatch finished.")
    print(f"Successful: {n_ok}")
    print(f"Failed: {n_failed}")
    print(f"Skipped: {n_skipped}")
    print(f"Status summary: {outdir / 'batch_status_summary.csv'}")
    if n_ok:
        print(f"Parameter summary: {outdir / 'batch_fit_params_summary.csv'}")
        print(f"Arc summary: {outdir / 'batch_arc_metrics_summary.csv'}")
        print(f"Fusion summary: {outdir / 'batch_fusion_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
