# HD88986 notes

Working directory: `/work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal`

---

## Layout

| Path | Contents |
|---|---|
| `ESO archive/` | Original ESO-derived RDB products + their processed outputs/figures |
| `data/` | Current analysis inputs: paper-table RDB mocks **+ APF** |
| `data/cds_J_A+A_681_A55/` | CDS machine-readable Tables D.1–D.3 (`J/A+A/681/A55`) |
| `data/processed_data/` | PyORBIT `.dat` files from `HD88986_data.py` |
| `figures/` | QC / LS figures from `HD88986_data.py` and `HD88986_figures_jz.py` |

---

## Scripts

| Script | Role |
|---|---|
| `HD88986_data.py` | Load RDBs → outlier clip → write PyORBIT `.dat` + timeseries plots |
| `HD88986_figures_jz.py` | Load processed `.dat` → activity + Lomb–Scargle figures |
| `build_paper_rdbs.py` | Rebuild **paper** RDBs from CDS only (does **not** overwrite APF) |
| `scripts_HD88986_emcee/setup_HD88986_emcee.sh` | Generate YAML + LSF scripts for all-instr **no-GP** emcee (1p–3p) |
| `scripts_HD88986_dynesty/setup_HD88986_dynesty.sh` | Generate YAML + LSF scripts for all-instr **no-GP** dynesty (1p–3p) |
| `scripts_HD88986_gp_emcee/setup_HD88986_gp_emcee.sh` | Generate YAML + LSF scripts for all-instr **sophie_gp** emcee (1p–3p) |
| `scripts_HD88986_gp_dynesty/setup_HD88986_gp_dynesty.sh` | Generate YAML + LSF scripts for all-instr **sophie_gp** dynesty (1p–3p) |

Run order:

```bash
python build_paper_rdbs.py   # only if regenerating paper RDBs (leaves APF alone)
python HD88986_data.py
python HD88986_figures_jz.py
```

---

## What was done (2026-09-25)

### 1. Pipeline adapted from HD102365
- Created `HD88986_data.py` (header-driven RDB loader, per-dataset σ-clip, PyORBIT export, plots).
- Created / adapted `HD88986_figures_jz.py` for HD88986 processed `.dat` files (including Na index).

### 2. ESO archive products
- Original ESO RDB set moved/kept under `ESO archive/` (APF, ELODIE, HIRES*, SOPHIE*).
- ESO-specific handling that was tested earlier:
  - Skip `#rjd` / `weight=0` rows
  - ELODIE `sig_fwhm` ÷ 1000 before writing FWHM errors
- **These ESO RDBs are not identical to the published paper tables** (different reductions / activity columns; SOPHIE+ incomplete vs Table D.1).

### 3. Consistency check (ESO RDB vs paper screenshots)
- **ELODIE (D.3):** mostly consistent (times/σ match; ~mm/s RV offset; paper includes two `#` / weight=0 epochs).
- **SOPHIE-old (D.1):** consistent with `SOPHIE_1` (km/s ↔ m/s).
- **SOPHIE+ (D.1):** same epochs roughly, but RVs differ by several m/s; paper has BIS/S/Na, ESO has FWHM/bis_span/rhk; paper extends later than ESO RDB.
- **HIRES (D.2):** `HIRES_1` shares epochs; S-index matches if ESO `s_mw/1000`; RVs/σ differ. PRE/POST are a different reduction.

### 4. Paper-table RDB mocks
Built from Heidari+2024 Appendix D / CDS `J/A+A/681/A55` (cross-checked vs screenshots):

| File | Rows | Table |
|---|---|---|
| `data/HD88986_SOPHIE_1.rdb` | 12 | D.1 SOPHIE (RV only) |
| `data/HD88986_SOPHIE-PLUS_1.rdb` | 378 | D.1 SOPHIE+ (RV, BIS, S-index, Na) |
| `data/HD88986_HIRES_1.rdb` | 17 | D.2 HIRES |
| `data/HD88986_HIRES-PLUS_1.rdb` | 33 | D.2 HIRES+ |
| `data/HD88986_ELODIE_1.rdb` | 28 | D.3 ELODIE |

Processing rules for paper mocks:
- SOPHIE RV / σ / BIS: **km/s → m/s (×1000)**
- Invalid activity sentinel **`999.0` → blank** (RV row kept; S/Na omitted for that epoch)
- Placeholder tiny errors (`0.000001`) where the paper has no published σ

### 5. APF added (not used in the paper)
- Source: ESO product, same as `ESO archive/data/HD88986_APF_1.rdb`
- Analysis copy: `data/HD88986_APF_1.rdb` (**19** RV epochs; RV-only)
- **Not** in Heidari+2024 Appendix D; included here as an extra instrument for the NEID proposal / multi-instrument fit
- `build_paper_rdbs.py` does not regenerate or overwrite this file
- `HD88986_data.py` uses a **3σ** outlier threshold for `HD88986_APF_1`

### 6. Pipeline status (paper set + APF)
- Current `data/` inputs: SOPHIE, SOPHIE+, HIRES, HIRES+, ELODIE, **APF**
- `HD88986_data.py` and `HD88986_figures_jz.py` run on this combined set
- Example activity coverage after dropping `999.0`: SOPHIE+ SHK ~322 pts, Na ~267 pts (of 378 RV epochs)

---

## PyORBIT emcee (first step: no GP)

Generator: `scripts_HD88986_emcee/setup_HD88986_emcee.sh`

```bash
bash scripts_HD88986_emcee/setup_HD88986_emcee.sh   # (re)create YAMLs + LSF scripts
cd scripts_HD88986_emcee
./submit_all_emcee.sh                               # or ./submit_1p_emcee.sh etc.
```

| Axis | Choice |
|---|---|
| Instruments | all (`APF`, `ELODIE`, `HIRES`, `HIRES-PLUS`, `SOPHIE`, `SOPHIE-PLUS`) |
| Activity / GP | **none** (`no_gp`) |
| Planets | `1p` (b ~146 d), `2p` (b + outer c ~116 yr), `3p` (b + c + free d) — no `0p` (no GP + no planets ⇒ no fit) |
| Sampler | emcee |

Results under `results_HD88986_emcee/all_instr/no_gp/{1,2,3}p/`.
LS files under `out_HD88986_emcee/`.

---

## PyORBIT dynesty (first step: no GP)

Generator: `scripts_HD88986_dynesty/setup_HD88986_dynesty.sh`

Same scientific setup as emcee (all instruments, no GP, 1p–3p; no 0p). Sampler is dynesty; jobs use **32 cores** (`cpu_threads` / `nthreads` = 31), walltime 48:00.

```bash
bash scripts_HD88986_dynesty/setup_HD88986_dynesty.sh   # (re)create YAMLs + LSF scripts
cd scripts_HD88986_dynesty
./submit_all_dynesty.sh                                 # or ./submit_1p_dynesty.sh etc.
```

| Axis | Choice |
|---|---|
| Instruments | all (`APF`, `ELODIE`, `HIRES`, `HIRES-PLUS`, `SOPHIE`, `SOPHIE-PLUS`) |
| Activity / GP | **none** (`no_gp`) |
| Planets | `1p` (b ~146 d), `2p` (b + outer c ~116 yr), `3p` (b + c + free d) — no `0p` |
| Sampler | dynesty |
| Resources | 32 cores, 2GB/core, walltime 48:00 |

Results under `results_HD88986_dynesty/all_instr/no_gp/{1,2,3}p/`.
LS files under `out_HD88986_dynesty/`.

### Dynesty model comparison (1p–3p)

Script: `post_analysis/compare_dynesty_models_HD88986.py`  
(Adapted from `PyORBIT_ESSP/post_analysis/compare_dynesty_models.py`; star-specific HD102365 copy was not in-tree.)

```bash
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal
python post_analysis/compare_dynesty_models_HD88986.py
```

Compares `all_instr` / `no_gp` dynesty runs for **1p, 2p, 3p** (skips any missing/incomplete logs).  
Outputs go to `post_analysis/dynesty_model_comparison/` (CSV, HTML, best-model planetFit CSV).

Orbital / planetFit values are **posterior median with 15–84 percentile uncertainties**
(`median^{+upper}_{-lower}`, ≈68% CI) taken from the PyORBIT log `(15-84 p)` columns.
`t0` in planetFit CSVs is derived from median `mean_long` / `ω` / `P` only (no invented error).

---

## Emcee model comparison (1p–3p)

Script: `post_analysis/compare_emcee_models_HD88986.py`  
(Adapted from `PyORBIT_ESSP/post_analysis/compare_emcee_models.py`; star-specific HD102365 copy was not in-tree.)

```bash
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal
python post_analysis/compare_emcee_models_HD88986.py
```

Compares `all_instr` / `no_gp` emcee runs for **1p, 2p, 3p** (skips any missing/incomplete logs).  
Reads Median BIC / AIC / AICc + Gelman–Rubin; ranks with ΔBIC/ΔAIC and lnZ proxy (−BIC/2).  
Outputs go to `post_analysis/results_emcee_all_instr_no_gp/` (CSV, HTML, metric/Δ plots, planetFit CSVs).

Orbital / planetFit uncertainties use the same `median^{+upper}_{-lower}` format from the
**last** PyORBIT stats block that reports `(15-84 p)` (later median-only dumps are skipped).
`t0` is median-derived only.

---

## PyORBIT GP (first step: SOPHIE-PLUS RV + BIS)

Separate folders from the no-GP setups (do **not** mix GP into `scripts_HD88986_emcee/` / `scripts_HD88986_dynesty/`).

Analogous to HD102365 `all_instr_espresso_gp`: all instruments enter the RV model; GP is applied only to **SOPHIE-PLUS RV + BIS**; other instruments remain Keplerian-only.

| Axis | Choice |
|---|---|
| Instruments | all RV (`APF`, `ELODIE`, `HIRES`, `HIRES-PLUS`, `SOPHIE`, `SOPHIE-PLUS`) + `HD88986_SOPHIE-PLUS_BIS.dat` |
| Activity / GP | `sophie_gp` — `spleaf_multidimensional_esp` on SOPHIE-PLUS RV + BIS only |
| Prot prior | G2; paper Prot = 25^{+8}_{-6} d → Gaussian(25, 7), bounds [10, 50] |
| Planets | `1p` / `2p` / `3p` (same bounds as no-GP) |
| Naming | `HD88986_all_instr_sophie_gp_{1,2,3}p_{emcee,dynesty}` |

### Emcee (GP)

Generator: `scripts_HD88986_gp_emcee/setup_HD88986_gp_emcee.sh`  
Resources: **32 cores**, 4GB/core (limit 5GB), walltime 72:00.

```bash
bash scripts_HD88986_gp_emcee/setup_HD88986_gp_emcee.sh   # (re)create YAMLs + LSF scripts
cd scripts_HD88986_gp_emcee
./submit_all_emcee.sh                                     # or ./submit_1p_emcee.sh etc.
# or: ./submit_all_instr_sophie_gp_emcee.sh
```

Results under `results_HD88986_gp_emcee/all_instr/sophie_gp/{1,2,3}p/`.  
Logs under `out_HD88986_gp_emcee/`.

### Dynesty (GP)

Generator: `scripts_HD88986_gp_dynesty/setup_HD88986_gp_dynesty.sh`  
Resources: **32 cores** (`cpu_threads` / `nthreads` = 31), 4GB/core (limit 5GB), walltime 72:00.

```bash
bash scripts_HD88986_gp_dynesty/setup_HD88986_gp_dynesty.sh   # (re)create YAMLs + LSF scripts
cd scripts_HD88986_gp_dynesty
./submit_all_dynesty.sh                                       # or ./submit_1p_dynesty.sh etc.
# or: ./submit_all_instr_sophie_gp_dynesty.sh
```

Results under `results_HD88986_gp_dynesty/all_instr/sophie_gp/{1,2,3}p/`.  
Logs under `out_HD88986_gp_dynesty/`.