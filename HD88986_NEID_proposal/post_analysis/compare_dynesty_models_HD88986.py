#!/usr/bin/env python3
"""
HD88986 dynesty model comparison (all_instr / no_gp; 1p–3p).

Adapted from PyORBIT_ESSP/post_analysis/compare_dynesty_models.py
(the HD102365-named script path was not present; this preserves the same
evidence / ΔlnZ / Bayes-factor / table / HTML comparison logic).

Usage (from anywhere):
  python post_analysis/compare_dynesty_models_HD88986.py

Optional:
  python post_analysis/compare_dynesty_models_HD88986.py --results-root /path/to/results_HD88986_dynesty
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

STAR = "HD88986"
DATA_CONFIG = "all_instr"
GP_CONFIG = "no_gp"
PLANET_MODELS = ("1p", "2p", "3p")  # no 0p for this setup
SAMPLER = "dynesty"

# PyORBIT Tref (JD) from YAML → eMJD = JD - 2400000.5
TREF_JD = 2450420.109460
REFERENCE_EPOCH_EMJD = TREF_JD - 2400000.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAR_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_RESULTS_ROOT = os.path.join(STAR_ROOT, f"results_{STAR}_{SAMPLER}")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"{SAMPLER}_model_comparison")


def calculate_t0_from_mean_long(mean_long_deg, omega_deg, period_days, reference_epoch=0.0):
    """t0 = t_ref - (mean_long - omega) / n, with n = 2π/P."""
    mean_long_rad = np.deg2rad(mean_long_deg)
    omega_rad = np.deg2rad(omega_deg)
    mean_anomaly_rad = mean_long_rad - omega_rad
    n = 2.0 * np.pi / period_days
    return reference_epoch - (mean_anomaly_rad / n)


def format_param_unc(entry, placeholder="—"):
    """
    Format a parsed PyORBIT posterior entry as median^{+up}_{-lo}.

    Errors come from the log's (15-84 p) columns (≈68% CI). Lower error in the
    log is already signed negative; we display magnitudes with explicit signs.
    """
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


def expected_job_name(planets: str) -> str:
    return f"{STAR}_{DATA_CONFIG}_{GP_CONFIG}_{planets}_{SAMPLER}"


def expected_log_path(results_root: str, planets: str) -> str:
    job = expected_job_name(planets)
    return os.path.join(
        results_root,
        DATA_CONFIG,
        GP_CONFIG,
        planets,
        job,
        f"configuration_file_{SAMPLER}_run_{job}.log",
    )


def discover_models(results_root: str):
    """Return (found_logs, missing_info) for 1p/2p/3p under all_instr/no_gp."""
    found = []
    missing = []
    for planets in PLANET_MODELS:
        log_path = expected_log_path(results_root, planets)
        job_dir = os.path.dirname(log_path)
        if not os.path.isdir(job_dir):
            missing.append((planets, "job directory missing", job_dir))
            print(f"[SKIP] {planets}: job directory not found:\n       {job_dir}")
            continue
        if not os.path.isfile(log_path):
            missing.append((planets, "log file missing", log_path))
            print(f"[SKIP] {planets}: log file not found:\n       {log_path}")
            continue
        found.append((planets, log_path))
        print(f"[FOUND] {planets}: {log_path}")
    return found, missing


def parse_dynesty_log_file(filepath: str, planets_hint: str | None = None):
    """
    Parse a PyORBIT dynesty log for logZ, BIC, efficiency, and parameters.

    Naming expected after stripping prefixes:
      HD88986_all_instr_no_gp_{1,2,3}p_dynesty
    """
    logz = None
    logz_err = None
    median_bic = None
    efficiency = None
    ncall = None
    niter = None
    orbital_parameters = {}
    activity_parameters = {}

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    logz_match = re.search(r"logz:\s*(-?[\d\.]+)\s*\+/-\s*([\d\.]+)", content)
    if logz_match:
        logz = float(logz_match.group(1))
        logz_err = float(logz_match.group(2))

    median_bic_match = re.search(
        r"Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)", content
    )
    if median_bic_match:
        median_bic = float(median_bic_match.group(1))

    eff_match = re.search(r"eff\(%\):\s*([\d\.]+)", content)
    if eff_match:
        efficiency = float(eff_match.group(1))

    ncall_match = re.search(r"ncall:\s*(\d+)", content)
    if ncall_match:
        ncall = int(ncall_match.group(1))

    niter_match = re.search(r"niter:\s*(\d+)", content)
    if niter_match:
        niter = int(niter_match.group(1))

    lines = content.split("\n")
    first_stats_idx = -1
    for i, line in enumerate(lines):
        if "Statistics on the model parameters obtained from the posteriors samples" in line:
            first_stats_idx = i
            break

    if first_stats_idx != -1:
        current_planet = None
        in_activity_section = False
        for i in range(first_stats_idx, len(lines)):
            line = lines[i]
            if "Statistics on the derived parameters" in line or "Parameters corresponding to" in line:
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

            param_match = re.match(
                r"^([A-Za-z_]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([\d\.]+).*\(15-84 p\)",
                line.strip(),
            )
            if param_match:
                param_name = param_match.group(1)
                median_str = param_match.group(2).strip()
                lower_error_str = param_match.group(3).strip()
                upper_error_str = param_match.group(4).strip()
                entry = {
                    "value": float(median_str),
                    "value_str": median_str,
                    "lower_error": float(lower_error_str),
                    "lower_error_str": lower_error_str,
                    "upper_error": float(upper_error_str),
                    "upper_error_str": upper_error_str,
                }
                if in_activity_section:
                    activity_parameters[param_name] = entry
                elif current_planet:
                    orbital_parameters[current_planet][param_name] = entry

    if logz is None:
        print(f"[SKIP] Incomplete / no logZ yet: {filepath}")
        return None

    basename = os.path.basename(filepath)
    cleaned = (
        basename.replace(f"configuration_file_{SAMPLER}_run_", "")
        .replace("configuration_file_run_", "")
        .replace(".log", "")
    )

    # HD88986_all_instr_no_gp_2p_dynesty
    match = re.match(rf"{STAR}_(.+)_(\dp)_{SAMPLER}$", cleaned)
    if match:
        config_name = match.group(1)
        num_planets = match.group(2)
    elif planets_hint in PLANET_MODELS:
        config_name = f"{DATA_CONFIG}_{GP_CONFIG}"
        num_planets = planets_hint
    else:
        config_name = cleaned
        num_planets = planets_hint or "N/A"

    return {
        "Configuration": config_name,
        "Dataset": STAR,
        "Planets": num_planets,
        "log(Z)": logz,
        "log(Z) error": logz_err,
        "Median BIC": median_bic,
        "Efficiency %": efficiency,
        "N calls": ncall,
        "N iter": niter,
        "Orbital Parameters": orbital_parameters,
        "Activity Parameters": activity_parameters,
        "File": basename,
        "Directory": os.path.dirname(os.path.abspath(filepath)),
    }


def evidence_strength(delta_logz: float) -> str:
    """Strength label for Δlog(Z) relative to the best model (≤ 0)."""
    if delta_logz > -1.0:
        return "Weak"
    if delta_logz > -2.5:
        return "Moderate"
    if delta_logz > -5.0:
        return "Strong"
    return "Decisive"


def export_planet_fit_csv(df, output_dir, reference_epoch=REFERENCE_EPOCH_EMJD):
    """Export best-logZ planet-fit CSV for each configuration (median^{+up}_{-lo})."""
    exported = []
    for config_name, group in df.groupby("Configuration"):
        group = group.copy()
        group["Planets"] = pd.Categorical(
            group["Planets"], categories=list(PLANET_MODELS), ordered=True
        )
        group = group.sort_values("Planets")
        best = group.loc[group["log(Z)"].idxmax()]
        orbital_params = best["Orbital Parameters"] or {}
        planet_rows = []
        for planet in sorted(orbital_params.keys()):
            p = orbital_params[planet]
            K = p.get("K", {}).get("value")
            P = p.get("P", {}).get("value")
            if K is None or P is None:
                continue
            e = p.get("e", {}).get("value")
            omega = p.get("omega", {}).get("value")
            mean_long = p.get("mean_long", {}).get("value")

            if mean_long is not None and omega is not None:
                t0 = calculate_t0_from_mean_long(mean_long, omega, P, reference_epoch)
            elif mean_long is not None:
                t0 = calculate_t0_from_mean_long(mean_long, 0.0, P, reference_epoch)
            else:
                t0 = None

            # t0 is derived from medians only — no invented uncertainty.
            t0_str = f"{t0:.10g}" if t0 is not None else ""
            e_fmt = format_param_unc(p.get("e"), placeholder="")
            if not e_fmt:
                e_fmt = "0" if e == 0.0 else ("" if e is None else str(e))
            omega_fmt = format_param_unc(p.get("omega"), placeholder="")
            if not omega_fmt:
                omega_fmt = "0" if omega == 0.0 else ("" if omega is None else str(omega))

            planet_rows.append(
                {
                    "K [m/s]": format_param_unc(p.get("K"), placeholder=str(K)),
                    "P [d]": format_param_unc(p.get("P"), placeholder=str(P)),
                    "t0 [eMJD]": t0_str,
                    "e": e_fmt,
                    "w [deg]": omega_fmt,
                }
            )

        columns = ["K [m/s]", "P [d]", "t0 [eMJD]", "e", "w [deg]"]
        planet_df = pd.DataFrame(planet_rows, columns=columns)
        filename = f"{STAR}_DTU-Padova-PSU_dynesty_{config_name}_planetFit.csv"
        filepath = os.path.join(output_dir, filename)
        planet_df.to_csv(filepath, index=False, encoding="utf-8")
        exported.append(filepath)
        print(f"  Exported: {filepath} ({len(planet_rows)} planet(s); best={best['Planets']})")
    return exported


def write_html(df, output_dir, timestamp):
    """Write a formatted HTML comparison (same metrics as the reference script)."""
    html_filepath = os.path.join(output_dir, f"dynesty_model_comparison_{timestamp}.html")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{STAR} Dynesty Model Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0; padding: 20px; background: #fafafa; color: #2c3e50; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ text-align: center; }}
.subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 24px; }}
.legend, .section {{ background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                     padding: 20px 25px; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ padding: 10px 14px; border-bottom: 1px solid #ecf0f1; text-align: left; }}
th {{ background: #f8f9fa; text-transform: uppercase; font-size: 12px; letter-spacing: 0.4px; }}
.highlight-logz {{ background: #e8f5e9 !important; font-weight: 600; color: #2e7d32; }}
.highlight-bic {{ background: #fff3e0 !important; font-weight: 600; color: #e65100; }}
.summary {{ margin-top: 12px; padding: 12px 16px; background: #f8f9fa; border-radius: 8px; }}
.badge-green {{ background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 4px; }}
.badge-orange {{ background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>{STAR} Dynesty Model Comparison</h1>
  <div class="subtitle">all_instr / no_gp — Bayesian evidence (nested sampling)</div>
  <div class="legend">
    <div><span class="badge-green">Green</span> = best log(Z) &nbsp;
         <span class="badge-orange">Orange</span> = best BIC</div>
    <div><strong>Δlog(Z)</strong> = difference from best model (0 = best)</div>
    <div>Evidence: |Δlog(Z)| &gt; 5 decisive; &gt; 2.5 strong; &gt; 1 moderate; &lt; 1 weak</div>
  </div>
"""

    for config_name, group in df.groupby("Configuration"):
        group = group.copy()
        group["Planets"] = pd.Categorical(
            group["Planets"], categories=list(PLANET_MODELS), ordered=True
        )
        group = group.sort_values("Planets")
        max_logz_idx = group["log(Z)"].idxmax()
        max_logz = group.loc[max_logz_idx, "log(Z)"]
        bic_ok = group["Median BIC"].notna()
        min_bic_idx = group.loc[bic_ok, "Median BIC"].idxmin() if bic_ok.any() else None
        min_bic = group.loc[min_bic_idx, "Median BIC"] if min_bic_idx is not None else np.nan

        html += f'  <div class="section">\n    <h2>Configuration: {config_name}</h2>\n'
        html += "    <table>\n"
        html += "      <tr><th>Planets</th><th>log(Z)</th><th>Δlog(Z)</th>"
        html += "<th>Median BIC</th><th>ΔBIC</th><th>Efficiency %</th><th>N calls</th></tr>\n"

        for idx, row in group.iterrows():
            dlogz = row["log(Z)"] - max_logz
            dbic = (row["Median BIC"] - min_bic) if pd.notna(row["Median BIC"]) and pd.notna(min_bic) else np.nan
            lz_cls = "highlight-logz" if idx == max_logz_idx else ""
            bic_cls = "highlight-bic" if min_bic_idx is not None and idx == min_bic_idx else ""
            bic_str = f"{row['Median BIC']:.2f}" if pd.notna(row["Median BIC"]) else "—"
            dbic_str = f"{dbic:.2f}" if pd.notna(dbic) else "—"
            eff = row["Efficiency %"]
            eff_str = f"{eff:.2f}%" if pd.notna(eff) else "—"
            ncall = row["N calls"]
            ncall_str = f"{int(ncall):,}" if pd.notna(ncall) else "—"
            html += "      <tr>\n"
            html += f"        <td>{row['Planets']}</td>\n"
            html += f'        <td class="{lz_cls}">{row["log(Z)"]:.3f} ± {row["log(Z) error"]:.3f}</td>\n'
            html += f'        <td class="{lz_cls}">{dlogz:.3f}</td>\n'
            html += f'        <td class="{bic_cls}">{bic_str}</td>\n'
            html += f'        <td class="{bic_cls}">{dbic_str}</td>\n'
            html += f"        <td>{eff_str}</td>\n"
            html += f"        <td>{ncall_str}</td>\n"
            html += "      </tr>\n"
        html += "    </table>\n"

        best_p = group.loc[max_logz_idx, "Planets"]
        html += '    <div class="summary">\n'
        html += f"      <strong>Best by log(Z):</strong> {best_p} (log(Z) = {max_logz:.3f})<br>\n"
        if min_bic_idx is not None:
            html += f"      <strong>Best by BIC:</strong> {group.loc[min_bic_idx, 'Planets']} (BIC = {min_bic:.2f})\n"
        html += "    </div>\n"

        # Orbital params for best model (median^{+up}_{-lo} from 15–84 p)
        orbital = group.loc[max_logz_idx, "Orbital Parameters"] or {}
        html += "    <h3>Best-model orbital parameters</h3>\n"
        html += '    <p style="color:#7f8c8d;font-size:0.95em;">Posterior median with '
        html += "15–84 percentile uncertainties (≈68% CI), from PyORBIT log.</p>\n"
        html += "    <table>\n"
        html += "      <tr><th>Planet</th><th>P (d)</th><th>K (m/s)</th><th>mean_long (°)</th><th>e</th><th>ω (°)</th></tr>\n"
        if not orbital:
            html += "      <tr><td colspan='6'>—</td></tr>\n"
        else:
            for planet in sorted(orbital.keys()):
                p = orbital[planet]
                html += "      <tr>\n"
                html += f"        <td>{planet}</td>\n"
                for key in ("P", "K", "mean_long", "e", "omega"):
                    html += f"        <td>{format_param_unc(p.get(key))}</td>\n"
                html += "      </tr>\n"
        html += "    </table>\n"
        html += "  </div>\n"

    html += "</div>\n</body>\n</html>\n"
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return html_filepath


def analyze_and_display(all_data, output_dir, missing):
    if not all_data:
        print("\nNo complete dynesty models available to compare.")
        if missing:
            print("Missing / incomplete:")
            for planets, reason, path in missing:
                print(f"  - {planets}: {reason} ({path})")
        return None

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print(f"{STAR} DYNESTY MODEL COMPARISON — BAYESIAN EVIDENCE")
    print(f"Data: {DATA_CONFIG} / {GP_CONFIG} | Models requested: {', '.join(PLANET_MODELS)}")
    print("=" * 100)
    print("Interpretation:")
    print("  • Δlog(Z) > 5.0  : Decisive evidence for better model")
    print("  • Δlog(Z) > 2.5  : Strong evidence")
    print("  • Δlog(Z) > 1.0  : Moderate evidence")
    print("  • Δlog(Z) < 1.0  : Weak/inconclusive evidence")
    print("  • Lower BIC is better (rule of thumb: ΔBIC > 10 is strong)")
    print("=" * 100)

    summary_rows = []

    for config_name, group in df.groupby("Configuration"):
        print(f"\n--- Configuration: {config_name} ---\n")
        group = group.copy()
        group["Planets"] = pd.Categorical(
            group["Planets"], categories=list(PLANET_MODELS), ordered=True
        )
        group = group.sort_values("Planets")

        max_logz_idx = group["log(Z)"].idxmax()
        max_logz = group.loc[max_logz_idx, "log(Z)"]
        bic_ok = group["Median BIC"].notna()
        min_bic_idx = group.loc[bic_ok, "Median BIC"].idxmin() if bic_ok.any() else None
        min_bic = group.loc[min_bic_idx, "Median BIC"] if min_bic_idx is not None else np.nan

        display = group[
            ["Planets", "log(Z)", "log(Z) error", "Median BIC", "Efficiency %", "N calls"]
        ].copy()
        display["Δlog(Z)"] = group["log(Z)"] - max_logz
        display["ΔBIC"] = group["Median BIC"] - min_bic if pd.notna(min_bic) else np.nan
        display["Bayes Factor"] = display["Δlog(Z)"].apply(lambda x: f"{np.exp(x):.2e}")

        print(
            display[
                ["Planets", "log(Z)", "Δlog(Z)", "Median BIC", "ΔBIC", "Efficiency %", "Bayes Factor"]
            ].to_string(index=False)
        )

        best_p = group.loc[max_logz_idx, "Planets"]
        print(
            f"\nBest by log(Z): {best_p} "
            f"(log(Z) = {max_logz:.3f} ± {group.loc[max_logz_idx, 'log(Z) error']:.3f})"
        )
        if min_bic_idx is not None:
            print(f"Best by BIC:    {group.loc[min_bic_idx, 'Planets']} (BIC = {min_bic:.2f})")

        if len(group) > 1:
            print("\nEvidence interpretation:")
            for idx, row in group.iterrows():
                if idx == max_logz_idx:
                    continue
                dlogz = row["log(Z)"] - max_logz
                bf = np.exp(-dlogz)
                strength = evidence_strength(dlogz)
                print(
                    f"  {row['Planets']} vs {best_p}: Δlog(Z) = {dlogz:.3f}, "
                    f"BF = {bf:.2e} → {strength} evidence for {best_p}"
                )

        # Orbital summary for best model (median^{+up}_{-lo})
        orbital = group.loc[max_logz_idx, "Orbital Parameters"] or {}
        print("\nBest-model orbital parameters (median^{+up}_{-lo}, 15–84 p ≈ 68% CI):")
        print(f"{'Planet':<8} {'P (d)':<28} {'K (m/s)':<24} {'mean_long':<24} {'e':<24} {'ω':<24}")
        print("-" * 130)
        if not orbital:
            print("(none)")
        else:
            for planet in sorted(orbital.keys()):
                p = orbital[planet]
                print(
                    f"{planet:<8} "
                    f"{format_param_unc(p.get('P'), '-'):<28} "
                    f"{format_param_unc(p.get('K'), '-'):<24} "
                    f"{format_param_unc(p.get('mean_long'), '-'):<24} "
                    f"{format_param_unc(p.get('e'), '-'):<24} "
                    f"{format_param_unc(p.get('omega'), '-'):<24}"
                )

        for _, row in group.iterrows():
            dlogz = row["log(Z)"] - max_logz
            summary_rows.append(
                {
                    "Dataset": STAR,
                    "Configuration": config_name,
                    "Planets": row["Planets"],
                    "log(Z)": row["log(Z)"],
                    "log(Z) error": row["log(Z) error"],
                    "Δlog(Z)": dlogz,
                    "Bayes Factor vs best": np.exp(dlogz),
                    "Median BIC": row["Median BIC"],
                    "ΔBIC": (row["Median BIC"] - min_bic) if pd.notna(row["Median BIC"]) and pd.notna(min_bic) else np.nan,
                    "Efficiency %": row["Efficiency %"],
                    "N calls": row["N calls"],
                    "N iter": row["N iter"],
                    "Preferred_logZ": row["Planets"] == best_p,
                    "Preferred_BIC": min_bic_idx is not None and row.name == min_bic_idx,
                    "File": row["File"],
                    "Directory": row["Directory"],
                }
            )

    if missing:
        print("\nModels not included in comparison:")
        for planets, reason, path in missing:
            print(f"  - {planets}: {reason}")
            print(f"    {path}")

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, f"dynesty_model_comparison_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    html_path = write_html(df, output_dir, timestamp)
    print("\nExporting best-fit planet CSV (best logZ model)...")
    planet_files = export_planet_fit_csv(df, output_dir)

    best_dirs = summary_df.loc[summary_df["Preferred_logZ"], ["Configuration", "Planets", "Directory"]]
    dir_csv = os.path.join(output_dir, f"best_model_directories_{timestamp}.csv")
    best_dirs.to_csv(dir_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 100)
    print("Results exported to:")
    print(f"  • CSV:  {csv_path}")
    print(f"  • HTML: {html_path}")
    print(f"  • Best dirs: {dir_csv}")
    for pf in planet_files:
        print(f"  • Planet fit: {pf}")
    print(f"  (reference_epoch for t0 = {REFERENCE_EPOCH_EMJD:.6f} eMJD from Tref JD {TREF_JD})")
    print("=" * 100)

    return {
        "csv": csv_path,
        "html": html_path,
        "dir_csv": dir_csv,
        "planet_files": planet_files,
        "summary_df": summary_df,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"{STAR} dynesty model comparison")
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root of dynesty results (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    print(f"Star:          {STAR}")
    print(f"Results root:  {args.results_root}")
    print(f"Data / GP:     {DATA_CONFIG} / {GP_CONFIG}")
    print(f"Planet models: {', '.join(PLANET_MODELS)}")
    print(f"Output dir:    {args.output_dir}\n")

    if not os.path.isdir(args.results_root):
        print(f"ERROR: results root does not exist: {args.results_root}")
        sys.exit(1)

    found, missing = discover_models(args.results_root)
    if not found:
        print("\nDry check: no dynesty logs available yet for any of 1p/2p/3p.")
        sys.exit(2)

    all_data = []
    incomplete = list(missing)
    for planets, log_path in found:
        data = parse_dynesty_log_file(log_path, planets_hint=planets)
        if data is None:
            incomplete.append((planets, "log present but incomplete (no logZ)", log_path))
        else:
            all_data.append(data)
            print(
                f"  parsed {planets}: log(Z) = {data['log(Z)']:.3f} ± {data['log(Z) error']:.3f}, "
                f"BIC = {data['Median BIC']}"
            )

    result = analyze_and_display(all_data, args.output_dir, incomplete)
    if result is None:
        sys.exit(3)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
