# Battery Data Analysis

Python tools for battery electrochemistry data analysis. The repository includes a Streamlit dashboard for interactive workflows and standalone command-line scripts for batch processing.

## Features

- Fit EIS data from BioLogic/EC-Lab `.mpr` files or exported `.csv`/`.txt` files.
- Batch fit EIS files with the equivalent circuit `R1 + Q2/R2 + Q3/R3 + Q4/(R4 + W4)`.
- Analyze cycling capacity retention and coulombic efficiency from Excel cycling data.
- Analyze stripping-cell overpotential and capacity-normalized voltage profiles.
- Generate V-Q / dQ/dV style voltage-capacity profiles from cycling Excel files.
- Export publication-ready figures and CSV summaries for downstream plotting or reporting.

## Repository Structure

```text
.
├── streamlit_app.py       # Interactive dashboard
├── eis_fit.py             # Single-file EIS fitting
├── eis_fit_batch.py       # Batch EIS fitting
├── capacity_batch.py      # Batch cycling capacity analysis
├── stripping_batch.py     # Batch stripping-cell analysis
├── dqdv_batch.py          # Batch V-Q / dQ/dV profile analysis
├── requirements.txt       # Python dependencies
└── eis_fit_results/       # Example or generated EIS output files
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- `streamlit`
- `numpy`
- `pandas`
- `scipy`
- `plotly`
- `openpyxl`
- `matplotlib`

## Interactive Dashboard

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The dashboard contains four main analysis pages:

- `EIS Fit`
- `Cycling Analysis`
- `Stripping Overpotential`
- `dQ/dV Analysis`

For large datasets, use the local/server folder path mode. Browser upload is best for small demo files only.

## Expected Data Layout

Most batch tools expect one first-level folder per sample:

```text
root_directory/
    Sample_A/
        file_1.xlsx
        file_2.xlsx
    Sample_B/
        Repeat_1/
            file_3.xlsx
        Repeat_2/
            file_4.xlsx
```

Files directly under a sample folder are treated as individual repeats or file-level records. Files inside repeat folders inherit the repeat name from the folder.

## Command-Line Usage

### Single EIS Fit

Fit one `.mpr`, `.csv`, or `.txt` EIS file:

```bash
python eis_fit.py "path/to/eis_file.mpr" \
    --xml "path/to/zfit_initial.xml" \
    --prefix results/sample_fit \
    --weight unit
```

Outputs:

- `<prefix>_fit_params.csv`
- `<prefix>_arc_metrics.csv`
- `<prefix>_fusion_metrics.csv`
- `<prefix>_fit_curve.csv`
- `<prefix>_nyquist_fit.png`

### Batch EIS Fit

Fit all `.mpr` files in a folder:

```bash
python eis_fit_batch.py "path/to/eis_folder" \
    --xml "path/to/zfit_initial.xml" \
    --outdir eis_fit_results \
    --recursive \
    --weight unit
```

Useful options:

- `--pattern "*.mpr"`: choose which files to fit.
- `--recursive`: search subfolders.
- `--weight unit|modulus|sqrt_modulus`: choose fitting residual weighting.
- `--no-overwrite`: skip files that already have outputs.

Batch outputs include per-file folders plus:

- `batch_fit_params_summary.csv`
- `batch_arc_metrics_summary.csv`
- `batch_fusion_metrics_summary.csv`
- `batch_status_summary.csv`

### Cycling Capacity Analysis

Process Excel cycling files and plot capacity retention / coulombic efficiency:

```bash
python capacity_batch.py "path/to/root_directory" \
    --output-dir "path/to/capacity_batch_results" \
    --sheet-name cycle \
    --min-capacity-retention 80
```

Default required columns in the `cycle` sheet:

- `DChg. Cap.(mAh)`
- `Chg.-DChg. Eff(%)`

The script skips the first two rows by default and normalizes retention to the first valid discharge capacity.

### Stripping Overpotential Analysis

Process stripping-cell Excel files by sample folder:

```bash
python stripping_batch.py "path/to/root_directory" \
    --output-dir-name stripping_outputs \
    --area 1.27 \
    --normalization area \
    --show-legend
```

Main outputs are written under:

```text
root_directory/stripping_outputs/
    figures/
    plot_data/
    summary/
```

The summary includes nucleation voltage, plateau voltage, overpotential, and areal capacity metrics.

### V-Q / dQ/dV Profile Analysis

Process voltage-capacity profiles from cycling Excel files:

```bash
python dqdv_batch.py "path/to/root_directory" \
    --output-dir "path/to/VQ_analysis_outputs" \
    --cycle-start 3 \
    --cycle-step 20 \
    --retention-cutoff 80
```

To use explicit cycles instead of an interval:

```bash
python dqdv_batch.py "path/to/root_directory" \
    --cycle-list "3,23,43,63"
```

The script reads active material area from the `test` sheet and cycling records from the `record` sheet. Required `record` columns are:

- `Cycle Index`
- `Step Type`
- `Capacity(mAh)`
- `Voltage(V)`

## Notes on EIS Input

For `.mpr` files, `eis_fit.py` includes a minimal BioLogic/EC-Lab binary reader designed to locate the VMP data block and extract common EIS columns. If a file cannot be decoded, export the EIS data from EC-Lab as CSV or TXT with frequency, real impedance, and imaginary impedance columns.

Optional EC-Lab ZFit XML files can be supplied to initialize circuit parameters. If no XML is provided, built-in defaults are used.

## Generated Files

Generated outputs may include CSV summaries, fit curves, PNG figures, and cached parsed data. These outputs are useful for checking and reporting results, but they can become large for full experimental datasets.

Common generated folders include:

- `eis_fit_results/`
- `capacity_batch_results/`
- `stripping_outputs/`
- `VQ_analysis_outputs/`

## Troubleshooting

- If Streamlit cannot find a local data path, make sure the path exists on the machine running Streamlit, not only on the browser machine.
- If Excel files fail to parse, confirm that the expected sheets and column names are present.
- If EIS `.mpr` parsing fails, export the data to CSV/TXT and rerun the fit.
- For large cloud-drive folders, start with a small subset of files before running full batch processing.
