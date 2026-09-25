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
| `scripts_HD88986_emcee/setup_HD88986_emcee.sh` | Generate YAML + LSF scripts for all-instr **no-GP** emcee (0p–3p) |

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
./submit_all_emcee.sh                               # or ./submit_0p_emcee.sh etc.
```

| Axis | Choice |
|---|---|
| Instruments | all (`APF`, `ELODIE`, `HIRES`, `HIRES-PLUS`, `SOPHIE`, `SOPHIE-PLUS`) |
| Activity / GP | **none** (`no_gp`) |
| Planets | `0p`, `1p` (b ~146 d), `2p` (b + outer c ~116 yr), `3p` (b + c + free d) |
| Sampler | emcee |

Results under `results_HD88986_emcee/all_instr/no_gp/{0,1,2,3}p/`.
LS files under `out_HD88986_emcee/`.
