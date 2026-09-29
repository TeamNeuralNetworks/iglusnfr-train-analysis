# Quickstart

Two common starting points: reproducing the published figures from the
released dataset, and re-running the extraction pipeline on your own
recordings with different settings.

## What do I need to generate the paper figures?

1. **Data** — download the dataset from Zenodo,
   DOI [10.5281/zenodo.21554049](https://doi.org/10.5281/zenodo.21554049),
   and extract it. The archive contains one folder, `release/`, with the
   consolidated CSV tables described in
   [`dataset_tools/SCHEMA.md`](dataset_tools/SCHEMA.md).

2. **Code** — clone or download this repository (for exact reproduction of
   the published figures, use the tagged release linked from the Zenodo record).

3. **Python environment**
   ```bash
   conda create -n glusnfr python=3.11
   conda activate glusnfr
   pip install -r requirements.txt
   ```

4. **Point the code at your data.** Either:
   - extract/copy the `release/` folder so it ends up at
     `<repo_root>/PPR_DATA_FINAL/release/` (no path edits needed), **or**
   - set the `GLUSNFR_DATA_ROOT` environment variable to wherever you put the
     data (the folder that directly *contains* `release/`):
     ```powershell
     $env:GLUSNFR_DATA_ROOT = "C:\path\to\your\data"
     ```
     ```bash
     export GLUSNFR_DATA_ROOT="/path/to/your/data"
     ```

   Both `Support_figure.ipynb` and the `Feature_extraction`/`Model_Calibration`
   demo scripts read this same variable, so you only need to set it once.

5. **Run the notebook.** Open
   [`Support_figure.ipynb`](Support_figure.ipynb) in Jupyter or VS Code, select
   the `glusnfr` kernel, and run all cells in order. Figures and statistics
   tables are written to `<data_root>/output/`.

   [`docs/model_fitting/Model_Fitting_Gallery.ipynb`](docs/model_fitting/Model_Fitting_Gallery.ipynb)
   and [`docs/nnls_lecture/NNLS_Lecture_Demo.ipynb`](docs/nnls_lecture/NNLS_Lecture_Demo.ipynb)
   are self-contained and don't require the dataset.

**Scope note:** the Zenodo deposit ships the *converted* recordings
(`release/*.csv`, including `release/raw/<uid>.csv` and
`release/boutons_manifest.csv`) — not the original raw per-condition `.xlsx`
files or the `.tif` microscopy stacks (see `SCHEMA.md` §4). That's sufficient
both to reproduce every figure **and** to run the `Feature_extraction`/
`Model_Calibration` demo scripts, since those read `release/raw/` + the
manifest, not the private `.xlsx` tree. You only need your own raw recordings
if you want to *add* data or regenerate `release/` from scratch (next section).

## How do I re-extract data differently?

Use this if you want to change extraction settings (fitting model, template
options, thresholds, ...) and regenerate the derived tables, rather than just
reproducing the published figures.

**Case A — you only have the public deposit, just want different settings.**
`release/` already has everything the demo scripts need
(`release/raw/<uid>.csv` + `release/boutons_manifest.csv`), so no raw data or
manifest-building step is required:

1. Point `GLUSNFR_DATA_ROOT` at your extracted `release/`'s parent folder, as
   above.
2. Edit the `options={}` dict / preset name and re-run the extraction, either
   for one recording
   ([`Feature_extraction/demo_single_file.py`](Feature_extraction/demo_single_file.py)
   — set `TARGET_LEGACY_ID` or `TARGET_UID` to the recording you want, looked
   up in `release/boutons_manifest.csv`) or the whole dataset
   ([`Feature_extraction/demo_batch_process.py`](Feature_extraction/demo_batch_process.py)).
   `demo_batch_process.py` has `WRITE_TIDY=True` by default, so it writes the
   tidy tables straight into `<DATA_ROOT>/release/` in the same run — no
   separate consolidation step needed.
3. Re-run `Support_figure.ipynb` against the regenerated `release/` folder —
   no notebook changes needed, since it reads through
   `dataset_tools/release_io.py` the same way regardless of how `release/`
   was produced.

**Case B — you have your own private raw recordings to add.** These need to
become "well-organized raw" first, via the one dedicated converter, before any
extraction script touches them:

```bash
# writes release/boutons_manifest.csv
python dataset_tools/build_manifest.py   --data-root <DATA_ROOT>
# writes release/raw/<uid>.csv + release/f0.csv, and the saturation tables
python dataset_tools/reorganize_raw.py   --data-root <DATA_ROOT> --out <DATA_ROOT>/release
python dataset_tools/convert_saturation.py --xlsx <DATA_ROOT>/Saturation_data.xlsx --out <DATA_ROOT>/release
```

Your raw recordings need to be organized as `<DATA_ROOT>/<condition>/<recording>.xlsx`
(one subfolder per experimental condition) for `build_manifest.py` to find them.
See [`dataset_tools/SCHEMA.md`](dataset_tools/SCHEMA.md) §5 for what each step
produces and how the columns are defined. Once this has run, proceed as in
Case A — the extraction scripts don't know or care whether `release/raw/` came
from the Zenodo deposit or your own conversion.
