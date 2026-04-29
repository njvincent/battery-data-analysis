#!/usr/bin/env python3
"""
Fit BioLogic/EC-Lab EIS data with the ZFit circuit:

    R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4)

Notation:
- Q is a constant phase element: Z_Q = 1 / (Q * (j omega)^alpha)
- "/" means parallel connection.
- W is a semi-infinite Warburg element: Z_W = sigma / sqrt(j omega)

Outputs:
- <prefix>_fit_params.csv
- <prefix>_arc_metrics.csv
- <prefix>_fusion_metrics.csv
- <prefix>_fit_curve.csv
- <prefix>_nyquist_fit.png

Example:
    python eis_fit.py "V1S2 Sym LHCE R1_C10.mpr" \
        --xml "V1S2 Sym LHCE R1_C10-Minimize-2.xml" \
        --prefix V1S2_fit \
        --weight unit
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


PARAM_ORDER = ["R1", "Q2", "a2", "R2", "Q3", "a3", "R3", "Q4", "a4", "R4", "s4"]
DEFAULT_P0 = {
    "R1": 1.0,
    "Q2": 1e-6, "a2": 0.9, "R2": 100.0,
    "Q3": 1e-4, "a3": 0.8, "R3": 300.0,
    "Q4": 1e-3, "a4": 0.8, "R4": 100.0,
    "s4": 1.0,
}


def _module_slices_mpr(raw: bytes):
    """
    Minimal BioLogic .mpr module reader.

    BioLogic's .mpr is not a public stable text format, but common EC-Lab files
    use a modular container with a 65-byte module header. This function is only
    intended to locate the VMP data block; it does not try to decode every column.
    """
    positions = [i for i in range(len(raw)) if raw.startswith(b"MODULE", i)]
    modules = []
    for pos in positions:
        header = raw[pos:pos + 65]
        if len(header) < 65:
            continue
        short_name = header[6:16].decode("latin1", errors="ignore").strip()
        long_name = header[16:41].decode("latin1", errors="ignore").strip()
        length = struct.unpack_from("<I", header, 45)[0]
        body_start = pos + 65
        body_end = body_start + length
        if body_end <= len(raw):
            modules.append((short_name, long_name, body_start, body_end))
    return modules


def read_biologic_mpr_eis(path: str | Path) -> pd.DataFrame:
    """
    Read the first five EIS columns from a BioLogic .mpr VMP data block.

    For the uploaded EC-Lab EIS file, the first five float32 columns are:
        freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, |Z|/Ohm, Phase(Z)/deg

    The function detects the row start by checking monotonic frequency and the
    consistency |Z| ~= sqrt(Re(Z)^2 + (-Im(Z))^2).
    """
    raw = Path(path).read_bytes()
    modules = _module_slices_mpr(raw)
    data_blocks = [(s, e) for short, long, s, e in modules if "data" in short.lower() or "data" in long.lower()]
    if not data_blocks:
        raise ValueError("No VMP data module found. Export the EIS data to CSV/TXT and use read_csv_eis instead.")

    body = raw[data_blocks[0][0]:data_blocks[0][1]]
    if len(body) < 32:
        raise ValueError("VMP data block is too small.")

    n_rows = struct.unpack_from("<I", body, 0)[0]
    n_cols = struct.unpack_from("<H", body, 4)[0]
    if n_rows <= 2 or n_rows > 1_000_000 or n_cols <= 0 or n_cols > 300:
        raise ValueError(f"Unreasonable data shape in .mpr: rows={n_rows}, columns={n_cols}")

    min_start = 6 + 2 * n_cols
    candidates = []
    # EC-Lab usually puts the numeric table after a binary metadata area.
    for start in range(min_start, min(len(body) - 20, 4000)):
        remain = len(body) - start
        if remain <= 0 or remain % n_rows != 0:
            continue
        row_size = remain // n_rows
        if row_size < 20 or row_size > 512:
            continue
        try:
            f = np.array([struct.unpack_from("<f", body, start + i * row_size + 0)[0] for i in range(n_rows)])
            zr = np.array([struct.unpack_from("<f", body, start + i * row_size + 4)[0] for i in range(n_rows)])
            zim_neg = np.array([struct.unpack_from("<f", body, start + i * row_size + 8)[0] for i in range(n_rows)])
            zmod = np.array([struct.unpack_from("<f", body, start + i * row_size + 12)[0] for i in range(n_rows)])
            phase = np.array([struct.unpack_from("<f", body, start + i * row_size + 16)[0] for i in range(n_rows)])
        except struct.error:
            continue

        if not (np.isfinite(f).all() and np.isfinite(zr).all() and np.isfinite(zim_neg).all()):
            continue
        if np.any(f <= 0) or f.max() / f.min() < 10:
            continue

        dfreq = np.diff(f)
        monotone_fraction = max(np.mean(dfreq < 0), np.mean(dfreq > 0))
        if monotone_fraction < 0.8:
            continue

        zmod_calc = np.sqrt(zr**2 + zim_neg**2)
        zmod_err = np.nanmedian(np.abs(zmod - zmod_calc) / np.maximum(1.0, zmod_calc))
        phase_calc = -np.degrees(np.arctan2(zim_neg, zr))
        phase_err = np.nanmedian(np.abs(phase - phase_calc))
        # Smaller is better. Weight monotonicity heavily.
        score = zmod_err + 0.01 * phase_err + (1 - monotone_fraction)
        candidates.append((score, start, row_size, f, zr, zim_neg, zmod, phase))

    if not candidates:
        raise ValueError(
            "Could not detect EIS table inside .mpr. Try exporting from EC-Lab as CSV/TXT "
            "with columns freq, Re(Z), -Im(Z)."
        )

    _, start, row_size, f, zr, zim_neg, zmod, phase = sorted(candidates, key=lambda x: x[0])[0]
    df = pd.DataFrame({
        "freq_hz": f.astype(float),
        "z_real_ohm": zr.astype(float),
        "minus_z_imag_ohm": zim_neg.astype(float),
        "z_imag_ohm": (-zim_neg).astype(float),
        "z_mod_ohm": zmod.astype(float),
        "phase_deg": phase.astype(float),
    })
    # Keep the original EC-Lab order, usually high frequency to low frequency.
    df.attrs["mpr_table_start"] = start
    df.attrs["mpr_row_size"] = row_size
    return df


def read_csv_eis(path: str | Path) -> pd.DataFrame:
    """
    Read CSV/TXT EIS data. The function tries to recognize common column names:
    frequency, Re(Z), Im(Z), -Im(Z).
    """
    path = Path(path)
    # sep=None lets pandas sniff comma/tab/semicolon delimiters.
    df0 = pd.read_csv(path, sep=None, engine="python")
    cols_lower = {c.lower().strip(): c for c in df0.columns}

    def find_col(keys):
        for low, original in cols_lower.items():
            if all(k in low for k in keys):
                return original
        return None

    fcol = find_col(["freq"]) or find_col(["frequency"]) or find_col(["f/"])
    rcol = find_col(["re"]) or find_col(["real"])
    negicol = find_col(["-im"]) or find_col(["minus", "im"]) or find_col(["-z"])
    icol = find_col(["im"]) or find_col(["imag"])

    if fcol is None or rcol is None or (negicol is None and icol is None):
        raise ValueError("CSV needs frequency, Re(Z), and Im(Z) or -Im(Z) columns.")

    f = pd.to_numeric(df0[fcol], errors="coerce").to_numpy(float)
    zr = pd.to_numeric(df0[rcol], errors="coerce").to_numpy(float)
    if negicol is not None:
        zim_neg = pd.to_numeric(df0[negicol], errors="coerce").to_numpy(float)
        zi = -zim_neg
    else:
        zi = pd.to_numeric(df0[icol], errors="coerce").to_numpy(float)
        zim_neg = -zi

    out = pd.DataFrame({
        "freq_hz": f,
        "z_real_ohm": zr,
        "minus_z_imag_ohm": zim_neg,
        "z_imag_ohm": zi,
    }).dropna()
    return out[out["freq_hz"] > 0].reset_index(drop=True)


def read_zfit_xml(path: str | Path) -> dict[str, float]:
    """Extract initial parameter values from an EC-Lab ZFit XML file."""
    root = ET.fromstring(Path(path).read_text(errors="ignore"))
    params = {}
    for elem in root.iter("Param"):
        name = elem.attrib.get("name")
        value = elem.attrib.get("value")
        if name and value is not None:
            try:
                params[name] = float(value)
            except ValueError:
                pass
    return params


def pack_params(params: dict[str, float]) -> np.ndarray:
    p = DEFAULT_P0.copy()
    p.update(params)
    return np.array([p[k] for k in PARAM_ORDER], dtype=float)


def unpack_params(p: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(PARAM_ORDER, p)}


def z_cpe(Q: float, alpha: float, omega: np.ndarray) -> np.ndarray:
    return 1.0 / (Q * (1j * omega) ** alpha)


def z_warburg(sigma: float, omega: np.ndarray) -> np.ndarray:
    return sigma / np.sqrt(1j * omega)


def parallel(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 / z1 + 1.0 / z2)


def circuit_z(p: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    R1, Q2, a2, R2, Q3, a3, R3, Q4, a4, R4, s4 = p
    omega = 2 * np.pi * freq_hz
    z2 = parallel(z_cpe(Q2, a2, omega), R2)
    z3 = parallel(z_cpe(Q3, a3, omega), R3)
    z4 = parallel(z_cpe(Q4, a4, omega), R4 + z_warburg(s4, omega))
    return R1 + z2 + z3 + z4


def component_zs(p: np.ndarray, freq_hz: np.ndarray) -> list[np.ndarray]:
    R1, Q2, a2, R2, Q3, a3, R3, Q4, a4, R4, s4 = p
    omega = 2 * np.pi * freq_hz
    return [
        parallel(z_cpe(Q2, a2, omega), R2),
        parallel(z_cpe(Q3, a3, omega), R3),
        parallel(z_cpe(Q4, a4, omega), R4 + z_warburg(s4, omega)),
    ]


def fit_eis(df: pd.DataFrame, p0: np.ndarray, weight: str = "unit") -> tuple[np.ndarray, object]:
    freq = df["freq_hz"].to_numpy(float)
    z_exp = df["z_real_ohm"].to_numpy(float) + 1j * df["z_imag_ohm"].to_numpy(float)

    def residual(p):
        z_fit = circuit_z(p, freq)
        err_re = np.real(z_fit - z_exp)
        err_im = np.imag(z_fit - z_exp)
        if weight == "modulus":
            w = np.maximum(np.abs(z_exp), 1e-12)
            err_re = err_re / w
            err_im = err_im / w
        elif weight == "sqrt_modulus":
            w = np.sqrt(np.maximum(np.abs(z_exp), 1e-12))
            err_re = err_re / w
            err_im = err_im / w
        elif weight != "unit":
            raise ValueError("weight must be unit, modulus, or sqrt_modulus")
        return np.r_[err_re, err_im]

    lower = np.array([0, 1e-15, 0.05, 0, 1e-15, 0.05, 0, 1e-15, 0.05, 0, 0], dtype=float)
    upper = np.array([np.inf, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf, np.inf], dtype=float)

    result = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        max_nfev=50_000,
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    return result.x, result


def _fwhm_log_band(freq: np.ndarray, y: np.ndarray):
    ymax = float(np.nanmax(y))
    if not np.isfinite(ymax) or ymax <= 0:
        return np.nan, np.nan, ymax
    idx = np.where(y >= 0.5 * ymax)[0]
    if len(idx) == 0:
        return np.nan, np.nan, ymax
    return float(np.log10(freq[idx[0]])), float(np.log10(freq[idx[-1]])), ymax


def arc_metrics(p: np.ndarray, fmin: float, fmax: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute geometry-like descriptors for each fitted arc.

    For an ideal R||C semicircle:
        diameter = R
        radius = R/2
        height = R/2
        depression_ratio = 1

    For CPE arcs, height < R/2. This script reports an effective radius R/2
    and a depression_ratio = numerical max(-Im(component)) / (R/2).
    """
    f_grid = np.logspace(np.log10(fmin), np.log10(fmax), 20_000)
    comps = component_zs(p, f_grid)
    R1, Q2, a2, R2, Q3, a3, R3, Q4, a4, R4, s4 = p
    Rs = [R2, R3, R4]
    Qs = [Q2, Q3, Q4]
    alphas = [a2, a3, a4]

    rows = []
    bands = []
    left = R1
    for i, zc in enumerate(comps, start=1):
        y = -np.imag(zc)
        peak_idx = int(np.nanargmax(y))
        diameter = float(Rs[i - 1])
        effective_radius = diameter / 2.0
        height = float(y[peak_idx])
        f_peak = float(f_grid[peak_idx])
        band_lo, band_hi, _ = _fwhm_log_band(f_grid, y)
        rows.append({
            "arc": i,
            "left_intercept_ohm": float(left),
            "right_intercept_ohm": float(left + diameter),
            "diameter_ohm": diameter,
            "effective_radius_ohm": effective_radius,
            "max_height_minus_im_ohm": height,
            "peak_frequency_hz": f_peak,
            "depression_ratio_height_over_radius": height / effective_radius if effective_radius > 0 else np.nan,
            "Q": float(Qs[i - 1]),
            "alpha": float(alphas[i - 1]),
            "note": "Arc 3 has Warburg if s4 is nonzero; intercept/radius are effective descriptors."
                    if i == 3 and abs(s4) > 1e-12 else "",
        })
        bands.append((band_lo, band_hi))
        left += diameter

    fusion_rows = []
    for i in range(len(bands) - 1):
        a0, a1 = bands[i]
        b0, b1 = bands[i + 1]
        if np.any(pd.isna([a0, a1, b0, b1])):
            overlap_min_width = np.nan
            overlap_union = np.nan
        else:
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            min_width = min(a1 - a0, b1 - b0)
            union = max(a1, b1) - min(a0, b0)
            overlap_min_width = overlap / min_width if min_width > 0 else np.nan
            overlap_union = overlap / union if union > 0 else np.nan
        fusion_rows.append({
            "arc_pair": f"{i + 1}-{i + 2}",
            "fusion_index_overlap_over_narrower_FWHM": overlap_min_width,
            "fusion_index_overlap_over_union_FWHM": overlap_union,
            "definition": "0 means separated; 1 means the narrower arc's half-height log-frequency band is fully covered.",
        })

    return pd.DataFrame(rows), pd.DataFrame(fusion_rows)


def plot_fit(df: pd.DataFrame, p: np.ndarray, out_png: str | Path):
    freq = df["freq_hz"].to_numpy(float)
    z_fit = circuit_z(p, freq)
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(df["z_real_ohm"], df["minus_z_imag_ohm"], label="data")
    plt.plot(np.real(z_fit), -np.imag(z_fit), label="fit")
    plt.xlabel("Z' / ohm")
    plt.ylabel("-Z'' / ohm")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help=".mpr or CSV/TXT EIS data file")
    parser.add_argument("--xml", default=None, help="optional EC-Lab ZFit XML file for initial values")
    parser.add_argument("--prefix", default="eis_fit", help="output file prefix")
    parser.add_argument("--weight", default="unit", choices=["unit", "modulus", "sqrt_modulus"])
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix.lower() == ".mpr":
        df = read_biologic_mpr_eis(data_path)
    else:
        df = read_csv_eis(data_path)

    p0_dict = read_zfit_xml(args.xml) if args.xml else {}
    p0 = pack_params(p0_dict)

    p_fit, result = fit_eis(df, p0, weight=args.weight)

    params_df = pd.DataFrame({
        "parameter": PARAM_ORDER,
        "initial": p0,
        "fit": p_fit,
    })
    params_df.loc[len(params_df)] = ["cost", np.nan, result.cost]
    params_df.loc[len(params_df)] = ["nfev", np.nan, result.nfev]
    params_df.to_csv(f"{args.prefix}_fit_params.csv", index=False)

    fmin, fmax = float(df["freq_hz"].min()), float(df["freq_hz"].max())
    metrics_df, fusion_df = arc_metrics(p_fit, fmin, fmax)
    metrics_df.to_csv(f"{args.prefix}_arc_metrics.csv", index=False)
    fusion_df.to_csv(f"{args.prefix}_fusion_metrics.csv", index=False)

    z_fit = circuit_z(p_fit, df["freq_hz"].to_numpy(float))
    curve_df = df.copy()
    curve_df["fit_z_real_ohm"] = np.real(z_fit)
    curve_df["fit_z_imag_ohm"] = np.imag(z_fit)
    curve_df["fit_minus_z_imag_ohm"] = -np.imag(z_fit)
    curve_df.to_csv(f"{args.prefix}_fit_curve.csv", index=False)

    plot_fit(df, p_fit, f"{args.prefix}_nyquist_fit.png")

    print("Fit completed.")
    print(params_df.to_string(index=False))
    print("\nArc metrics:")
    print(metrics_df.to_string(index=False))
    print("\nFusion metrics:")
    print(fusion_df.to_string(index=False))


if __name__ == "__main__":
    main()
