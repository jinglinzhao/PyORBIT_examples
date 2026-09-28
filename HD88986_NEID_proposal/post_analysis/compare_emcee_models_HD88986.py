#!/usr/bin/env python3
"""
Compare HD88986 emcee models (all_instr / no_gp: 1p, 2p, 3p).

Adapted from the repo-root post_analysis/compare_emcee_models.py workflow
used for HD102365 (the star-specific path
HD102365/post_analysis/compare_emcee_models_HD102365.py is not present
in this checkout).

Extracts Median BIC / AIC / AICc, Gelman–Rubin convergence, orbital
parameters; builds Δ metrics and a BIC-based lnZ proxy (lnZ ≈ −BIC/2);
writes CSV / HTML / comparison plots. Incomplete runs are skipped with
clear messages.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / model catalogue
# ---------------------------------------------------------------------------
STAR = "HD88986"
INSTR = "all_instr"
GP = "no_gp"
MODELS = ("1p", "2p", "3p")  # no 0p for this no-GP setup

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS_ROOT = PROJECT / f"results_{STAR}_emcee" / INSTR / GP
OUTPUT_DIR = HERE / "results_emcee_all_instr_no_gp"

# Reference epoch used when exporting planet-fit CSVs (eMJD); adjust if needed.
REFERENCE_EPOCH = 59334.700184


def model_run_dir(planets: str) -> Path:
    name = f"{STAR}_{INSTR}_{GP}_{planets}_emcee"
    return RESULTS_ROOT / planets / name


def model_log_path(planets: str) -> Path:
    name = f"{STAR}_{INSTR}_{GP}_{planets}_emcee"
    return model_run_dir(planets) / f"configuration_file_emcee_run_{name}.log"


def calculate_t0_from_mean_long(mean_long_deg, omega_deg, period_days, reference_epoch=0.0):
    mean_long_rad = np.deg2rad(mean_long_deg)
    omega_rad = np.deg2rad(omega_deg)
    mean_anomaly_rad = mean_long_rad - omega_rad
    n = 2.0 * np.pi / period_days
    return reference_epoch - mean_anomaly_rad / n


def format_param_unc(entry, placeholder="—"):
    """Format a parsed PyORBIT posterior entry as median^{+up}_{-lo} (15–84 p ≈ 68% CI)."""
    if not entry or entry.get("value") is None:
        return placeholder
    med = entry.get("value_str")
    if med is None:
        med = f"{entry['value']}"
    lo = entry.get("lower_error")
    hi = entry.get("upper_error")
    if lo is None or hi is None:
        return med
    lo_str = entry.get("lower_error_str")
    hi_str = entry.get("upper_error_str")
    lo_mag = (lo_str or f"{abs(lo)}").lstrip("+-")
    hi_mag = (hi_str or f"{hi}").lstrip("+-")
    return f"{med}^{{+{hi_mag}}}_{{-{lo_mag}}}"


def _param_entry(median_str, lower_str=None, upper_str=None, gelman_rubin=None):
    entry = {
        "value": float(median_str),
        "value_str": median_str,
        "gelman_rubin": gelman_rubin,
    }
    if lower_str is not None and upper_str is not None:
        entry["lower_error"] = float(lower_str)
        entry["lower_error_str"] = lower_str
        entry["upper_error"] = float(upper_str)
        entry["upper_error_str"] = upper_str
    return entry


def parse_log_file(filepath: Path, planets_label: str):
    """Parse a PyORBIT emcee log. Returns None if Median BIC is missing."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        print(f"  SKIP {planets_label}: log not found — {filepath}")
        return None
    except OSError as exc:
        print(f"  SKIP {planets_label}: cannot read log ({exc})")
        return None

    bic_match = re.search(r"Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)", content)
    aic_match = re.search(r"Median AIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)", content)
    aicc_match = re.search(r"Median AICc\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)", content)

    if not bic_match:
        print(f"  SKIP {planets_label}: Median BIC not found (run incomplete?) — {filepath}")
        return None

    median_bic = float(bic_match.group(1))
    median_aic = float(aic_match.group(1)) if aic_match else None
    median_aicc = float(aicc_match.group(1)) if aicc_match else None
    # Laplace / BIC evidence proxy used for rough model ranking alongside BIC.
    lnz_proxy = -0.5 * median_bic

    # Allow hyphens in instrument names (e.g. HIRES-PLUS).
    gr_pattern = (
        r"Gelman-Rubin:\s+(\d+)\s+([\d\.]+)\s+"
        r"([A-Za-z0-9_][A-Za-z0-9_\-]*)\s*$"
    )
    gr_matches = re.findall(gr_pattern, content, re.MULTILINE)
    gr_dict = {name: float(val) for _, val, name in gr_matches}
    gelman_rubin_values = list(gr_dict.values())

    orbital_parameters = {}
    activity_parameters = {}

    lines = content.split("\n")
    stats_idxs = [
        i
        for i, line in enumerate(lines)
        if "Statistics on the model parameters obtained from the posteriors samples" in line
    ]

    # Prefer the last stats block that includes (15-84 p) uncertainties.
    # Later blocks are often median-only MAP-style dumps without errors.
    stats_idx = -1
    for i in reversed(stats_idxs):
        end = next((j for j in stats_idxs if j > i), len(lines))
        chunk = "\n".join(lines[i:end])
        if "(15-84 p)" in chunk:
            stats_idx = i
            break
    if stats_idx == -1 and stats_idxs:
        stats_idx = stats_idxs[-1]

    if stats_idx != -1:
        current_planet = None
        in_activity_section = False
        for i in range(stats_idx, len(lines)):
            line = lines[i]
            if "Statistics on the derived parameters" in line:
                break

            planet_match = re.search(r"----- common model:\s+([a-z])\s*$", line)
            if planet_match:
                current_planet = planet_match.group(1)
                in_activity_section = False
                orbital_parameters.setdefault(current_planet, {})
                continue

            if "----- common model:  activity" in line:
                in_activity_section = True
                current_planet = None
                continue

            if "----- common model:" in line:
                current_planet = None
                in_activity_section = False
                continue

            unc_match = re.match(
                r"^([A-Za-z_]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([\d\.]+).*\(15-84 p\)",
                line.strip(),
            )
            med_match = re.match(r"^([A-Za-z_]+)\s+([-\d\.]+)\s*$", line.strip())
            if unc_match:
                param_name = unc_match.group(1)
                entry = _param_entry(
                    unc_match.group(2).strip(),
                    unc_match.group(3).strip(),
                    unc_match.group(4).strip(),
                )
            elif med_match:
                param_name = med_match.group(1)
                entry = _param_entry(med_match.group(2).strip())
            else:
                continue

            if in_activity_section:
                full = f"activity_{param_name}"
                entry["gelman_rubin"] = gr_dict.get(full)
                activity_parameters[param_name] = entry
            elif current_planet:
                full = f"{current_planet}_{param_name}"
                entry["gelman_rubin"] = gr_dict.get(full)
                orbital_parameters[current_planet][param_name] = entry

        for planet in orbital_parameters:
            for key in ("sre_coso", "sre_sino"):
                full = f"{planet}_{key}"
                if full in gr_dict:
                    orbital_parameters[planet].setdefault(key, {})
                    orbital_parameters[planet][key]["gelman_rubin"] = gr_dict[full]

    if gelman_rubin_values:
        converged_count = sum(1 for gr in gelman_rubin_values if gr < 1.1)
        convergence_pct = 100.0 * converged_count / len(gelman_rubin_values)
        max_gr = max(gelman_rubin_values)
    else:
        convergence_pct = 0.0
        max_gr = None

    return {
        "Star": STAR,
        "Configuration": f"{INSTR}/{GP}",
        "Planets": planets_label,
        "Median BIC": median_bic,
        "Median AIC": median_aic,
        "Median AICc": median_aicc,
        "lnZ_proxy": lnz_proxy,
        "Convergence %": convergence_pct,
        "Max GR": max_gr,
        "Orbital Parameters": orbital_parameters,
        "Activity Parameters": activity_parameters,
        "File": filepath.name,
        "Directory": str(filepath.parent.resolve()),
        "N_GR": len(gelman_rubin_values),
    }


def export_planet_fit_csv(row, output_dir: Path):
    orbital_params = row["Orbital Parameters"] or {}
    planets = sorted(orbital_params.keys())
    columns = ["K [m/s]", "P [d]", "t0 [eMJD]", "e", "w [deg]"]
    planet_rows = []
    for planet in planets:
        pp = orbital_params[planet]
        K = pp.get("K", {}).get("value")
        P = pp.get("P", {}).get("value")
        if K is None or P is None:
            continue
        e = pp.get("e", {}).get("value")
        omega = pp.get("omega", {}).get("value")
        mean_long = pp.get("mean_long", {}).get("value")
        if mean_long is not None:
            t0 = calculate_t0_from_mean_long(
                mean_long, 0.0 if omega is None else omega, P, REFERENCE_EPOCH
            )
            t0_str = f"{t0:.10g}"
        else:
            t0_str = ""
        e_fmt = format_param_unc(pp.get("e"), placeholder="")
        if not e_fmt:
            e_fmt = "0" if e is None else str(e)
        omega_fmt = format_param_unc(pp.get("omega"), placeholder="")
        if not omega_fmt:
            omega_fmt = "0" if omega is None else str(omega)
        planet_rows.append(
            {
                "K [m/s]": format_param_unc(pp.get("K"), placeholder=str(K)),
                "P [d]": format_param_unc(pp.get("P"), placeholder=str(P)),
                "t0 [eMJD]": t0_str,  # derived from medians only
                "e": e_fmt,
                "w [deg]": omega_fmt,
            }
        )
    df = pd.DataFrame(planet_rows, columns=columns)
    fname = f"{STAR}_{INSTR}_{GP}_{row['Planets']}_emcee_planetFit.csv"
    path = output_dir / fname
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def make_comparison_plots(df: pd.DataFrame, output_dir: Path, stamp: str):
    """Bar charts for BIC / AIC / AICc / lnZ proxy and Δ metrics."""
    order = [m for m in MODELS if m in set(df["Planets"])]
    d = df.set_index("Planets").loc[order]

    metrics = [
        ("Median BIC", "BIC (lower better)", "bic"),
        ("Median AIC", "AIC (lower better)", "aic"),
        ("Median AICc", "AICc (lower better)", "aicc"),
        ("lnZ_proxy", "lnZ proxy = −BIC/2 (higher better)", "lnz_proxy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    x = np.arange(len(order))
    for ax, (col, title, _) in zip(axes, metrics):
        vals = d[col].astype(float).values
        colors = ["#2e7d32" if v == (vals.max() if col == "lnZ_proxy" else vals.min()) else "#5c6bc0" for v in vals]
        if col == "lnZ_proxy":
            best = vals.max()
            colors = ["#2e7d32" if v == best else "#5c6bc0" for v in vals]
        else:
            best = vals.min()
            colors = ["#2e7d32" if v == best else "#5c6bc0" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{STAR} emcee — {INSTR} / {GP}", fontsize=13)
    fig.tight_layout()
    path = output_dir / f"emcee_model_comparison_metrics_{stamp}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Δ relative to best (min for IC, max for lnZ proxy)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.22
    for i, (col, label, key) in enumerate(
        [
            ("ΔBIC", "ΔBIC", "dbic"),
            ("ΔAIC", "ΔAIC", "daic"),
            ("ΔAICc", "ΔAICc", "daicc"),
            ("ΔlnZ_proxy", "ΔlnZ proxy", "dlnz"),
        ]
    ):
        if col not in d.columns:
            continue
        ax.bar(x + (i - 1.5) * width, d[col].astype(float).values, width, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Δ relative to preferred model")
    ax.set_title(f"{STAR} emcee Δ metrics ({INSTR}/{GP})")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path2 = output_dir / f"emcee_model_comparison_deltas_{stamp}.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    return path, path2


def write_html(df: pd.DataFrame, output_path: Path):
    preferred = df.loc[df["Preferred_BIC"].astype(bool)].iloc[0]
    rows_html = []
    for _, row in df.iterrows():
        cls = ' class="best"' if row["Preferred_BIC"] else ""
        max_gr = f"{row['Max GR']:.4f}" if pd.notna(row["Max GR"]) else "—"
        aic = f"{row['Median AIC']:.2f}" if pd.notna(row["Median AIC"]) else "—"
        aicc = f"{row['Median AICc']:.2f}" if pd.notna(row["Median AICc"]) else "—"
        daic = f"{row['ΔAIC']:.2f}" if pd.notna(row.get("ΔAIC")) else "—"
        daicc = f"{row['ΔAICc']:.2f}" if pd.notna(row.get("ΔAICc")) else "—"
        rows_html.append(
            f"<tr{cls}>"
            f"<td>{row['Planets']}</td>"
            f"<td>{row['Median BIC']:.2f}</td><td>{row['ΔBIC']:.2f}</td>"
            f"<td>{aic}</td><td>{daic}</td>"
            f"<td>{aicc}</td><td>{daicc}</td>"
            f"<td>{row['lnZ_proxy']:.2f}</td><td>{row['ΔlnZ_proxy']:.2f}</td>"
            f"<td>{row['Convergence %']:.1f}%</td><td>{max_gr}</td>"
            f"</tr>"
        )

    # Orbital summary for each model
    orb_sections = []
    for _, row in df.iterrows():
        orb = row["Orbital Parameters"] or {}
        if not orb:
            orb_sections.append(f"<h3>{row['Planets']}</h3><p>No orbital parameters.</p>")
            continue
        lines = [
            f"<h3>{row['Planets']}</h3>",
            "<table><tr><th>Planet</th><th>P [d]</th><th>K [m/s]</th>"
            "<th>e</th><th>ω [deg]</th><th>mean_long [deg]</th></tr>",
        ]
        for planet in sorted(orb.keys()):
            pp = orb[planet]
            lines.append(
                f"<tr><td>{planet}</td>"
                f"<td>{format_param_unc(pp.get('P'))}</td>"
                f"<td>{format_param_unc(pp.get('K'))}</td>"
                f"<td>{format_param_unc(pp.get('e'))}</td>"
                f"<td>{format_param_unc(pp.get('omega'))}</td>"
                f"<td>{format_param_unc(pp.get('mean_long'))}</td></tr>"
            )
        lines.append("</table>")
        orb_sections.append("\n".join(lines))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{STAR} emcee model comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 24px; background: #fafafa; color: #2c3e50; }}
h1,h2,h3 {{ color: #1a237e; }}
table {{ border-collapse: collapse; background: white; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #cfd8dc; padding: 8px 12px; text-align: right; }}
th {{ background: #eceff1; text-align: center; }}
td:first-child, th:first-child {{ text-align: center; }}
tr.best td {{ background: #e8f5e9; font-weight: 600; }}
.note {{ color: #546e7a; font-size: 0.95em; }}
</style></head><body>
<h1>{STAR} emcee model comparison</h1>
<p>Configuration: <b>{INSTR} / {GP}</b> &nbsp;|&nbsp; Models: {", ".join(MODELS)}</p>
<p class="note">lnZ proxy = −BIC/2 (rough Laplace/BIC evidence ranking; not a nested-sampling lnZ).
Green row = lowest Median BIC.</p>
<p><b>Preferred (BIC):</b> {preferred['Planets']}
(BIC = {preferred['Median BIC']:.2f}, lnZ proxy = {preferred['lnZ_proxy']:.2f})</p>
<h2>Information criteria</h2>
<table>
<tr><th>Planets</th><th>BIC</th><th>ΔBIC</th><th>AIC</th><th>ΔAIC</th>
<th>AICc</th><th>ΔAICc</th><th>lnZ proxy</th><th>ΔlnZ proxy</th>
<th>Conv %</th><th>Max GR</th></tr>
{''.join(rows_html)}
</table>
<h2>Orbital parameters (posterior median ± 15–84% ≈ 68% CI)</h2>
<p class="note">Format: median^{{+upper}}_{{-lower}} from the last PyORBIT stats block that reports (15-84 p).</p>
{''.join(orb_sections)}
</body></html>
"""
    output_path.write_text(html, encoding="utf-8")


def print_orbital_table(rows):
    print("\n" + "=" * 100)
    print("ORBITAL PARAMETERS (median^{+up}_{-lo}, 15–84 p ≈ 68% CI)")
    print("=" * 100)
    hdr = (
        f"{'Model':<6} {'Pl':<4} {'P [d]':<28} {'K [m/s]':<24} "
        f"{'e':<24} {'ω [deg]':<24} {'λ [deg]':<24}"
    )
    print(hdr)
    print("-" * 140)
    for row in rows:
        orb = row["Orbital Parameters"] or {}
        if not orb:
            print(f"{row['Planets']:<6} {'—':<4}")
            continue
        for planet in sorted(orb.keys()):
            pp = orb[planet]
            print(
                f"{row['Planets']:<6} {planet:<4} "
                f"{format_param_unc(pp.get('P'), '—'):<28} "
                f"{format_param_unc(pp.get('K'), '—'):<24} "
                f"{format_param_unc(pp.get('e'), '—'):<24} "
                f"{format_param_unc(pp.get('omega'), '—'):<24} "
                f"{format_param_unc(pp.get('mean_long'), '—'):<24}"
            )


def main() -> int:
    print(f"HD88986 emcee model comparison")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Models: {', '.join(MODELS)}")
    print()

    if not RESULTS_ROOT.is_dir():
        print(f"ERROR: results directory does not exist: {RESULTS_ROOT}")
        return 1

    rows = []
    for planets in MODELS:
        log_path = model_log_path(planets)
        print(f"Parsing {planets}: {log_path}")
        if not log_path.is_file():
            print(f"  SKIP {planets}: missing log file")
            continue
        data = parse_log_file(log_path, planets)
        if data is None:
            continue
        print(
            f"  OK  BIC={data['Median BIC']:.2f}  "
            f"AIC={data['Median AIC']}  AICc={data['Median AICc']}  "
            f"lnZ_proxy={data['lnZ_proxy']:.2f}  "
            f"conv={data['Convergence %']:.1f}%  maxGR={data['Max GR']}"
        )
        rows.append(data)

    if not rows:
        print("\nNo complete emcee runs found to compare.")
        print("Expected logs under:")
        for planets in MODELS:
            print(f"  {model_log_path(planets)}")
        return 1

    if len(rows) < 2:
        print(f"\nOnly {len(rows)} complete model(s); writing partial summary anyway.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(rows)
    # Preferred = lowest BIC among available models
    min_bic = df["Median BIC"].min()
    max_lnz = df["lnZ_proxy"].max()
    df["ΔBIC"] = df["Median BIC"] - min_bic
    df["Preferred_BIC"] = df["Median BIC"] == min_bic
    df["ΔlnZ_proxy"] = df["lnZ_proxy"] - max_lnz
    if df["Median AIC"].notna().any():
        min_aic = df["Median AIC"].min()
        df["ΔAIC"] = df["Median AIC"] - min_aic
        df["Preferred_AIC"] = df["Median AIC"] == min_aic
    else:
        df["ΔAIC"] = np.nan
        df["Preferred_AIC"] = False
    if df["Median AICc"].notna().any():
        min_aicc = df["Median AICc"].min()
        df["ΔAICc"] = df["Median AICc"] - min_aicc
        df["Preferred_AICc"] = df["Median AICc"] == min_aicc
    else:
        df["ΔAICc"] = np.nan
        df["Preferred_AICc"] = False

    cat = pd.Categorical(df["Planets"], categories=list(MODELS), ordered=True)
    df = df.assign(Planets=cat).sort_values("Planets").reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"MODEL COMPARISON — {STAR} / {INSTR} / {GP}")
    print("=" * 90)
    display_cols = [
        "Planets",
        "Median BIC",
        "ΔBIC",
        "Median AIC",
        "ΔAIC",
        "Median AICc",
        "ΔAICc",
        "lnZ_proxy",
        "ΔlnZ_proxy",
        "Convergence %",
        "Max GR",
    ]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    best = df.loc[df["Preferred_BIC"]].iloc[0]
    print(
        f"\nPreferred (lowest BIC): {best['Planets']}  "
        f"BIC={best['Median BIC']:.2f}  lnZ_proxy={best['lnZ_proxy']:.2f}"
    )
    if best["Convergence %"] < 90:
        print(
            f"NOTE: preferred model has Convergence % = {best['Convergence %']:.1f}% "
            f"(Max GR = {best['Max GR']}); treat ranking with caution."
        )

    print_orbital_table(rows)

    # Exports (drop bulky nested dicts from the flat CSV)
    export_df = df[
        [
            "Star",
            "Configuration",
            "Planets",
            "Median BIC",
            "ΔBIC",
            "Median AIC",
            "ΔAIC",
            "Median AICc",
            "ΔAICc",
            "lnZ_proxy",
            "ΔlnZ_proxy",
            "Convergence %",
            "Max GR",
            "Preferred_BIC",
            "Preferred_AIC",
            "Preferred_AICc",
            "File",
            "Directory",
        ]
    ].copy()
    csv_path = OUTPUT_DIR / f"model_comparison_{stamp}.csv"
    export_df.to_csv(csv_path, index=False, encoding="utf-8")

    html_path = OUTPUT_DIR / f"model_comparison_{stamp}.html"
    write_html(df, html_path)

    plot_paths = make_comparison_plots(df, OUTPUT_DIR, stamp)

    planet_csvs = []
    for _, row in df.iterrows():
        planet_csvs.append(export_planet_fit_csv(row, OUTPUT_DIR))

    best_dirs = OUTPUT_DIR / "best_model_directory.txt"
    best_dirs.write_text(best["Directory"] + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print("Outputs written to:", OUTPUT_DIR)
    print(f"  CSV:   {csv_path}")
    print(f"  HTML:  {html_path}")
    print(f"  Plots: {plot_paths[0]}")
    print(f"         {plot_paths[1]}")
    for p in planet_csvs:
        print(f"  PlanetFit: {p}")
    print(f"  Best dir: {best_dirs}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
