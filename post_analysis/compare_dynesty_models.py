import re
import os
import pandas as pd
from datetime import datetime
import numpy as np

def calculate_t0_from_mean_long(mean_long_deg, omega_deg, period_days, reference_epoch=0.0):
    """
    Calculate time of periastron (t0) from mean longitude.

    Formula: t0 = t_ref - (mean_long - omega) / n
    where n = 2π/P is the mean motion

    Args:
        mean_long_deg (float): Mean longitude in degrees
        omega_deg (float): Argument of periastron in degrees
        period_days (float): Orbital period in days
        reference_epoch (float): Reference epoch in eMJD (default: 0.0)

    Returns:
        float: Time of periastron in eMJD
    """
    # Convert degrees to radians
    mean_long_rad = np.deg2rad(mean_long_deg)
    omega_rad = np.deg2rad(omega_deg)

    # Calculate mean anomaly at reference epoch
    mean_anomaly_rad = mean_long_rad - omega_rad

    # Calculate mean motion (radians per day)
    n = 2.0 * np.pi / period_days

    # Calculate time since periastron at reference epoch
    dt = mean_anomaly_rad / n

    # Time of periastron
    t0 = reference_epoch - dt

    return t0

def export_planet_fit_csv(datasets, df, group_name="DTU-Padova-PSU", reference_epoch=0.0, output_dir="."):
    """
    Export Best-Fit Planet Parameters CSV files for ESSP submission.

    Creates CSV files named: <<Dataset>>_<<Group Name>>_<<Method Name>>_planetFit.csv
    with columns: K [m/s], P [d], t0 [eMJD], e, w [deg]

    Args:
        datasets (list): List of dataset names
        df (DataFrame): DataFrame containing parsed log file data
        group_name (str): Group name for file naming (default: "DTU-Padova-PSU")
        reference_epoch (float): Reference epoch in eMJD for t0 calculation (default: 0.0)
        output_dir (str): Output directory path (default: ".")
    """
    exported_files = []

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')

        for config_name, group in grouped:
            # Find the best model (highest log(Z))
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            max_logz_idx = group['log(Z)'].idxmax()
            best_model_data = group.loc[max_logz_idx]

            orbital_params = best_model_data['Orbital Parameters']

            # Prepare data for CSV
            planet_rows = []
            planets = sorted(orbital_params.keys()) if orbital_params else []

            # Process planets if they exist
            if planets:
                for planet in planets:
                    planet_params = orbital_params[planet]

                    # Extract parameters - use original string values from log file to preserve exact format
                    K_str = planet_params.get('K', {}).get('value_str')
                    P_str = planet_params.get('P', {}).get('value_str')
                    e_str = planet_params.get('e', {}).get('value_str')
                    omega_str = planet_params.get('omega', {}).get('value_str')
                    mean_long_str = planet_params.get('mean_long', {}).get('value_str')
                    
                    # Also get float values for calculations
                    K = planet_params.get('K', {}).get('value')
                    P = planet_params.get('P', {}).get('value')
                    e = planet_params.get('e', {}).get('value')
                    omega = planet_params.get('omega', {}).get('value')
                    mean_long = planet_params.get('mean_long', {}).get('value')

                    # Skip if essential parameters are missing
                    if K is None or P is None:
                        continue

                    # Calculate t0 from mean longitude (using float values for calculation)
                    if mean_long is not None and omega is not None:
                        t0 = calculate_t0_from_mean_long(mean_long, omega, P, reference_epoch)
                    elif mean_long is not None:
                        # If omega is missing but mean_long exists, assume circular orbit (omega = 0)
                        t0 = calculate_t0_from_mean_long(mean_long, 0.0, P, reference_epoch)
                    else:
                        # If mean_long is missing, we can't calculate t0
                        # Set to None (will be written as empty)
                        t0 = None

                    # Use original string values from log file, preserving exact format
                    # For calculated t0, convert to string (will format appropriately)
                    # For parameters with defaults, use string representation
                    if e_str is None:
                        e_str = '0' if e == 0.0 else str(e)
                    if omega_str is None:
                        omega_str = '0' if omega == 0.0 else str(omega)
                    
                    # Format t0 as string (preserve precision, remove trailing zeros only if they're truly unnecessary)
                    if t0 is not None:
                        t0_str = f"{t0:.10g}"  # Use g format to remove unnecessary trailing zeros
                    else:
                        t0_str = ''

                    # Create row with string values to preserve exact format from log file
                    row = {
                        'K [m/s]': K_str if K_str is not None else str(K),
                        'P [d]': P_str if P_str is not None else str(P),
                        't0 [eMJD]': t0_str,
                        'e': e_str,
                        'w [deg]': omega_str
                    }
                    planet_rows.append(row)

            # Create DataFrame with headers (even if empty for 0-planet models)
            # Define column order for CSV
            columns = ['K [m/s]', 'P [d]', 't0 [eMJD]', 'e', 'w [deg]']
            
            if planet_rows:
                planet_df = pd.DataFrame(planet_rows)
            else:
                # Create empty DataFrame with headers for 0-planet models
                planet_df = pd.DataFrame(columns=columns)

            # Create filename: DS1_DTU-Padova-PSU_dynesty_<config_name>_planetFit.csv
            filename = f"{dataset}_{group_name}_{config_name}_planetFit.csv"
            filepath = os.path.join(output_dir, filename)

            # Export to CSV - values are already strings preserving exact format from log file
            # For empty DataFrames, this will create a file with just headers
            planet_df.to_csv(filepath, index=False, encoding='utf-8')
            exported_files.append(filepath)
            
            num_planets = len(planet_rows)
            if num_planets == 0:
                print(f"  Exported: {filepath} (0-planet model, headers only)")
            else:
                print(f"  Exported: {filepath} ({num_planets} planet(s))")

    return exported_files

def export_best_model_directories(datasets, df, output_filename=None, output_dir="."):
    """
    Export a simple list of directory names where best models (highest log(Z)) were found.
    
    Args:
        datasets (list): List of dataset names
        df (DataFrame): DataFrame containing parsed log file data
        output_filename (str, optional): Output filename. If None, uses timestamp-based name.
        output_dir (str): Output directory path (default: ".")
    
    Returns:
        str: Filepath of the exported CSV file
    """
    directory_names = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')
        
        for config_name, group in grouped:
            # Find the best model (highest log(Z))
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            max_logz_idx = group['log(Z)'].idxmax()
            best_model_data = group.loc[max_logz_idx]
            
            # Get directory name
            directory_name = best_model_data.get('Directory', '')
            directory_names.append(directory_name)
    
    # Create DataFrame with just directory names
    if directory_names:
        dir_df = pd.DataFrame({'Directory': directory_names})
        
        # Use provided filename or fall back to timestamp-based name
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_model_directories_{timestamp}.csv"
        else:
            filename = output_filename
        
        filepath = os.path.join(output_dir, filename)
        dir_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"  Exported: {filepath} ({len(directory_names)} directory name(s))")
        return filepath
    return None

def parse_dynesty_log_file(filepath):
    """
    Parses a PyORBIT dynesty log file to extract logZ, BIC, efficiency, and parameters.

    Args:
        filepath (str): The full path to the log file.

    Returns:
        dict: A dictionary containing the model name, logZ, BIC, efficiency, and parameters.
              Returns None if the required lines are not found.
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
        with open(filepath, 'r') as f:
            content = f.read()

            # Extract log-evidence (logZ)
            logz_match = re.search(r'logz:\s*(-?[\d\.]+)\s*\+/-\s*([\d\.]+)', content)
            if logz_match:
                logz = float(logz_match.group(1))
                logz_err = float(logz_match.group(2))

            # Extract Median BIC (changed from MAP BIC)
            median_bic_match = re.search(r'Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)', content)
            if median_bic_match:
                median_bic = float(median_bic_match.group(1))

            # Extract efficiency
            eff_match = re.search(r'eff\(%\):\s*([\d\.]+)', content)
            if eff_match:
                efficiency = float(eff_match.group(1))

            # Extract number of calls
            ncall_match = re.search(r'ncall:\s*(\d+)', content)
            if ncall_match:
                ncall = int(ncall_match.group(1))

            # Extract number of iterations
            niter_match = re.search(r'niter:\s*(\d+)', content)
            if niter_match:
                niter = int(niter_match.group(1))

            # Extract orbital and activity parameters from the FIRST "Statistics on the model parameters" section
            lines = content.split('\n')

            # Find the FIRST occurrence of "Statistics on the model parameters obtained from the posteriors samples"
            first_stats_idx = -1
            for i, line in enumerate(lines):
                if "Statistics on the model parameters obtained from the posteriors samples" in line:
                    first_stats_idx = i
                    break  # Stop at first occurrence

            if first_stats_idx != -1:
                # Now parse from this section onwards
                current_planet = None
                in_activity_section = False

                for i in range(first_stats_idx, len(lines)):
                    line = lines[i]

                    # Stop if we hit the next major section
                    if "Statistics on the derived parameters" in line or "Parameters corresponding to" in line:
                        break

                    # Check for planet section headers (e.g., "----- common model:  b")
                    planet_match = re.search(r'----- common model:\s+([a-z])\s*$', line)
                    if planet_match:
                        current_planet = planet_match.group(1)
                        in_activity_section = False
                        if current_planet not in orbital_parameters:
                            orbital_parameters[current_planet] = {}
                        continue

                    # Check for activity section header
                    if "----- common model:  activity" in line:
                        in_activity_section = True
                        current_planet = None
                        continue

                    # Parse parameter lines (format: "P                         30.0         -1.0          1.0    (15-84 p)")
                    # Format: parameter_name, median_value, lower_error (negative), upper_error (positive), (15-84 p)
                    # Use flexible whitespace matching (\s+) to handle variable spacing
                    param_match = re.match(r'^([A-Za-z_]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([\d\.]+).*\(15-84 p\)', line.strip())
                    if param_match:
                        param_name = param_match.group(1)
                        # Store original string values to preserve decimal places
                        median_str = param_match.group(2).strip()
                        lower_error_str = param_match.group(3).strip()
                        upper_error_str = param_match.group(4).strip()
                        # Also store as floats for calculations
                        median_value = float(median_str)
                        lower_error = float(lower_error_str)  # Already negative
                        upper_error = float(upper_error_str)  # Positive

                        if in_activity_section:
                            activity_parameters[param_name] = {
                                'value': median_value,
                                'value_str': median_str,  # Original string representation
                                'lower_error': lower_error,
                                'lower_error_str': lower_error_str,  # Original string representation
                                'upper_error': upper_error,
                                'upper_error_str': upper_error_str  # Original string representation
                            }
                        elif current_planet:
                            orbital_parameters[current_planet][param_name] = {
                                'value': median_value,
                                'value_str': median_str,  # Original string representation
                                'lower_error': lower_error,
                                'lower_error_str': lower_error_str,  # Original string representation
                                'upper_error': upper_error,
                                'upper_error_str': upper_error_str  # Original string representation
                            }

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    if logz is not None:
        # Extract model name details from the filename
        basename = os.path.basename(filepath)
        # Remove common prefixes and .log extension
        cleaned_name = basename.replace('configuration_file_emcee_run_', '').replace('configuration_file_run_', '').replace('.log', '')

        # Extract dataset (DS1, DS2, etc.) and number of planets
        match = re.match(r'(DS\d+)_(\dp)_(.*)', cleaned_name)
        if match:
            dataset = match.group(1)
            num_planets = match.group(2)
            config_name = match.group(3)
        else:
            # Fallback for other naming patterns
            dataset = 'Unknown'
            num_planets = 'N/A'
            config_name = cleaned_name

        # Extract full directory path from filepath
        directory_name = os.path.dirname(os.path.abspath(filepath))
        
        return {
            'Configuration': config_name,
            'Dataset': dataset,
            'Planets': num_planets,
            'log(Z)': logz,
            'log(Z) error': logz_err,
            'Median BIC': median_bic,
            'Efficiency %': efficiency,
            'N calls': ncall,
            'N iter': niter,
            'Orbital Parameters': orbital_parameters,
            'Activity Parameters': activity_parameters,
            'File': basename,
            'Directory': directory_name
        }
    return None

def analyze_and_display_dynesty(log_files, search_directory=None):
    """
    Analyzes a list of dynesty log files and prints a formatted comparison table.
    Also exports results to CSV with highlighting for preferred models.

    Args:
        log_files (list): A list of paths to the log files.
        search_directory (str, optional): The directory path where log files were searched.
                                          Used to name output files.
    """
    # Create output directory based on search_directory
    output_dir = "."
    if search_directory:
        folder_name = os.path.basename(os.path.normpath(search_directory))
        output_dir = folder_name
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}\n")
    
    all_data = []
    failed_files = []

    print(f"Processing {len(log_files)} dynesty log files...")
    for log_file in log_files:
        data = parse_dynesty_log_file(log_file)
        if data:
            all_data.append(data)
        else:
            failed_files.append(log_file)

    if failed_files:
        print(f"\nWarning: {len(failed_files)} files could not be parsed (missing logZ data):")
        for f in failed_files:
            print(f"  - {f}")
        print()

    if not all_data:
        print("No data could be extracted from any log files found.")
        return

    # Create a DataFrame and group by configuration and dataset
    df = pd.DataFrame(all_data)

    # Group by dataset first, then by configuration
    datasets = sorted(df['Dataset'].unique())

    # --- Display Results ---
    print("=" * 100)
    print("PyORBIT DYNESTY MODEL COMPARISON - BAYESIAN EVIDENCE ANALYSIS")
    print("=" * 100)
    print("\nInterpretation Guide:")
    print("  • Δlog(Z) > 5.0  : Decisive evidence for better model")
    print("  • Δlog(Z) > 2.5  : Strong evidence")
    print("  • Δlog(Z) > 1.0  : Moderate evidence")
    print("  • Δlog(Z) < 1.0  : Weak/inconclusive evidence")
    print("  • Lower BIC is better (rule of thumb: ΔBIC > 10 is strong)")
    print("=" * 100)

    # Prepare data for CSV export
    export_data = []

    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')

        print(f"\n{'='*100}")
        print(f"DATASET: {dataset}")
        print(f"{'='*100}\n")

        # Store best models info for parameters summary
        best_models_info = []

        for config_name, group in grouped:
            print(f"--- Configuration: {config_name} ---\n")

            # Sort by number of planets
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')

            # Find the best model (highest log(Z))
            max_logz_idx = group['log(Z)'].idxmax()
            max_logz_value = group.loc[max_logz_idx, 'log(Z)']

            # Find minimum BIC
            min_bic_idx = group['Median BIC'].idxmin()
            min_bic_value = group.loc[min_bic_idx, 'Median BIC']

            # Create display dataframe
            display_group = group[['Planets', 'log(Z)', 'log(Z) error', 'Median BIC', 'Efficiency %', 'N calls']].copy()

            # Add Δlog(Z) and ΔBIC columns
            display_group['Δlog(Z)'] = group['log(Z)'] - max_logz_value
            display_group['ΔBIC'] = group['Median BIC'] - min_bic_value

            # Calculate Bayes factors
            display_group['Bayes Factor'] = display_group['Δlog(Z)'].apply(lambda x: f"{np.exp(x):.2e}")

            # Add preferred model indicators
            group_copy = group.copy()
            group_copy['Preferred_logZ'] = group_copy.index == max_logz_idx
            group_copy['Preferred_BIC'] = group_copy.index == min_bic_idx
            group_copy['Δlog(Z)'] = group_copy['log(Z)'] - max_logz_value
            group_copy['ΔBIC'] = group_copy['Median BIC'] - min_bic_value
            group_copy['Dataset'] = dataset
            group_copy['Configuration'] = config_name

            # Reorder columns for export
            export_group = group_copy[['Dataset', 'Configuration', 'Planets', 'log(Z)', 'log(Z) error',
                                      'Δlog(Z)', 'Median BIC', 'ΔBIC', 'Efficiency %',
                                      'N calls', 'N iter', 'Preferred_logZ', 'Preferred_BIC', 'File']]
            export_data.append(export_group)

            # Print table with key columns
            print(display_group[['Planets', 'log(Z)', 'Δlog(Z)', 'Median BIC', 'ΔBIC', 'Efficiency %']].to_string(index=False))

            print(f"\n📊 Best Model by log(Z): {group.loc[max_logz_idx, 'Planets']} (log(Z) = {max_logz_value:.2f} ± {group.loc[max_logz_idx, 'log(Z) error']:.2f})")
            print(f"📊 Best Model by BIC:    {group.loc[min_bic_idx, 'Planets']} (BIC = {min_bic_value:.2f})")

            # Evidence interpretation
            if len(group) > 1:
                print(f"\n🔬 Evidence Interpretation:")
                for idx, row in group.iterrows():
                    if idx != max_logz_idx:
                        delta_logz = row['log(Z)'] - max_logz_value
                        bf = np.exp(-delta_logz)  # Bayes factor against best model

                        if delta_logz > -1.0:
                            strength = "Weak"
                        elif delta_logz > -2.5:
                            strength = "Moderate"
                        elif delta_logz > -5.0:
                            strength = "Strong"
                        else:
                            strength = "Decisive"

                        print(f"   {row['Planets']} vs {group.loc[max_logz_idx, 'Planets']}: "
                              f"Δlog(Z) = {delta_logz:.2f}, BF = {bf:.2e} → {strength} evidence for {group.loc[max_logz_idx, 'Planets']}")

            print("\n" + "-"*100 + "\n")

            # Store best model info for summary table
            best_model_data = group.loc[max_logz_idx]
            best_models_info.append({
                'Configuration': config_name,
                'Best Model': best_model_data['Planets'],
                'log(Z)': best_model_data['log(Z)'],
                'BIC': best_model_data['Median BIC'],
                'Orbital Parameters': best_model_data['Orbital Parameters'],
                'Activity Parameters': best_model_data['Activity Parameters']
            })

        # Display orbital & activity parameters summary table for this dataset
        if best_models_info:
            print(f"\n{'='*140}")
            print(f"PARAMETERS SUMMARY FOR {dataset} (BEST MODELS BY log(Z))")
            print(f"{'='*140}\n")

            # Table 1: ORBITAL PARAMETERS
            print("TABLE 1: ORBITAL PARAMETERS\n")
            print(f"{'Config':<20} {'Model':<8} {'Planet':<8} {'P (days)':<12} {'K (m/s)':<10} {'mean_long (°)':<14} {'e':<10} {'ω (deg)':<10}")
            print("-" * 140)

            for model_info in best_models_info:
                config_name = model_info['Configuration']
                best_model = model_info['Best Model']
                orbital_params = model_info['Orbital Parameters']

                planets = sorted(orbital_params.keys()) if orbital_params else []

                if not planets:
                    # 0-planet model
                    print(f"{config_name:<20} {best_model:<8} {'-':<8} {'-':<12} {'-':<10} {'-':<14} {'-':<10} {'-':<10}")
                else:
                    for planet in planets:
                        planet_params = orbital_params[planet]

                        # Use original string values to preserve decimal places from log file
                        p_val = planet_params.get('P', {}).get('value_str', '-') if 'P' in planet_params else '-'
                        k_val = planet_params.get('K', {}).get('value_str', '-') if 'K' in planet_params else '-'
                        ml_val = planet_params.get('mean_long', {}).get('value_str', '-') if 'mean_long' in planet_params else '-'
                        e_val_str = planet_params.get('e', {}).get('value_str') if 'e' in planet_params else None
                        omega_val_str = planet_params.get('omega', {}).get('value_str') if 'omega' in planet_params else None

                        e_str = e_val_str if e_val_str is not None else '-'
                        omega_str = omega_val_str if omega_val_str is not None else '-'

                        print(f"{config_name:<20} {best_model:<8} {planet:<8} {p_val:<12} {k_val:<10} {ml_val:<14} {e_str:<10} {omega_str:<10}")

            print()

            # Table 2: Activity Parameters
            print("TABLE 2: ACTIVITY PARAMETERS\n")
            print(f"{'Config':<20} {'Model':<8} {'Prot (days)':<15} {'Pdec (days)':<15} {'Oamp':<10}")
            print("-" * 80)

            for model_info in best_models_info:
                config_name = model_info['Configuration']
                best_model = model_info['Best Model']
                activity_params = model_info['Activity Parameters']

                # Use original string values to preserve decimal places from log file
                prot_val = activity_params.get('Prot', {}).get('value_str', '-') if 'Prot' in activity_params else '-'
                pdec_val = activity_params.get('Pdec', {}).get('value_str', '-') if 'Pdec' in activity_params else '-'
                oamp_val = activity_params.get('Oamp', {}).get('value_str', '-') if 'Oamp' in activity_params else '-'

                print(f"{config_name:<20} {best_model:<8} {prot_val:<15} {pdec_val:<15} {oamp_val:<10}")

            print("\n" + "="*140 + "\n")

    # Export to HTML
    if export_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export HTML with formatting
        html_filename = f"dynesty_model_comparison_{timestamp}.html"
        html_filepath = os.path.join(output_dir, html_filename)

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PyORBIT Dynesty Model Comparison</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
            color: #2c3e50;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .dataset-section {
            background-color: white;
            margin: 30px 0;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }
        .dataset-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 30px;
            font-size: 22px;
            font-weight: 600;
        }
        .config-section {
            margin: 0;
            padding: 25px 30px;
            border-bottom: 1px solid #ecf0f1;
        }
        .config-section:last-child {
            border-bottom: none;
        }
        h2 { 
            color: #34495e;
            margin: 0 0 15px 0;
            font-size: 18px;
            font-weight: 600;
        }
        h3 {
            color: #2c3e50;
            margin: 25px 0 10px 0;
            font-size: 16px;
            font-weight: 600;
            background-color: #ecf0f1;
            padding: 10px 15px;
            border-radius: 5px;
        }
        table { 
            border-collapse: collapse; 
            margin: 15px 0;
            width: 100%;
            background-color: white;
        }
        .params-summary-section {
            margin: 30px;
            padding: 25px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .params-table {
            border-collapse: collapse;
            margin: 15px 0;
            width: 100%;
            background-color: white;
            border: 2px solid #3498db;
            font-size: 12px;
        }
        .params-table th {
            background-color: #3498db;
            color: white;
            padding: 10px 6px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #2980b9;
            font-size: 11px;
        }
        .params-table td {
            padding: 8px 6px;
            border: 1px solid #bdc3c7;
            text-align: center;
        }
        .params-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .params-table tr:hover {
            background-color: #e8f4f8;
        }
        th, td { 
            padding: 12px 16px; 
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th { 
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #34495e;
            font-size: 14px;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background-color: #f8f9fa;
        }
        .highlight-logz {
            background-color: #e8f5e9 !important;
            font-weight: 600;
            color: #2e7d32;
        }
        .highlight-bic {
            background-color: #fff3e0 !important;
            font-weight: 600;
            color: #e65100;
        }
        .evidence-strong {
            color: #2e7d32;
            font-weight: 600;
        }
        .evidence-moderate {
            color: #f57c00;
            font-weight: 600;
        }
        .evidence-weak {
            color: #c62828;
            font-weight: 600;
        }
        .summary {
            margin: 15px 0 0 0;
            padding: 15px 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            font-size: 14px;
            color: #34495e;
        }
        .summary strong {
            color: #2c3e50;
        }
        .legend {
            margin: 0 0 30px 0;
            padding: 20px 25px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .legend h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 18px;
            font-weight: 600;
            background: none;
            padding: 0;
        }
        .legend-item {
            margin: 8px 0;
            color: #34495e;
            font-size: 14px;
            line-height: 1.6;
        }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
        }
        .badge-green {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .badge-orange {
            background-color: #fff3e0;
            color: #e65100;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyORBIT Dynesty Model Comparison</h1>
        <div class="subtitle">Bayesian Evidence Analysis via Nested Sampling</div>
        
        <div class="legend">
            <h3>Interpretation Guide</h3>
            <div class="legend-item"><span class="badge badge-green">Green highlight</span> = Best log(Z) (highest evidence)</div>
            <div class="legend-item"><span class="badge badge-orange">Orange highlight</span> = Best BIC (lowest value)</div>
            <div class="legend-item"><strong>Δlog(Z):</strong> Difference in log-evidence from best model (0 = best)</div>
            <div class="legend-item"><strong>Bayes Factor:</strong> exp(Δlog(Z)) - ratio of evidences</div>
            <div class="legend-item">
                <strong>Evidence strength:</strong><br>
                • |Δlog(Z)| > 5.0: Decisive<br>
                • |Δlog(Z)| > 2.5: Strong<br>
                • |Δlog(Z)| > 1.0: Moderate<br>
                • |Δlog(Z)| < 1.0: Weak
            </div>
        </div>
"""
        # Process each dataset
        for dataset in datasets:
            html_content += f'        <div class="dataset-section">\n'
            html_content += f'            <div class="dataset-header">{dataset}</div>\n'

            dataset_df = df[df['Dataset'] == dataset]
            grouped = dataset_df.groupby('Configuration')

            best_models_info = []

            for config_name, group in grouped:
                html_content += f'            <div class="config-section">\n'
                html_content += f'                <h2>Configuration: {config_name}</h2>\n'

                group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
                group = group.sort_values('Planets')

                max_logz_idx = group['log(Z)'].idxmax()
                min_bic_idx = group['Median BIC'].idxmin()
                max_logz_value = group.loc[max_logz_idx, 'log(Z)']
                min_bic_value = group.loc[min_bic_idx, 'Median BIC']

                # Build table
                html_content += '                <table>\n'
                html_content += '                    <tr><th>Planets</th><th>log(Z)</th><th>Δlog(Z)</th><th>Median BIC</th><th>ΔBIC</th><th>Efficiency %</th><th>N calls</th></tr>\n'

                for idx, row in group.iterrows():
                    logz_class = 'highlight-logz' if idx == max_logz_idx else ''
                    bic_class = 'highlight-bic' if idx == min_bic_idx else ''

                    delta_logz = row['log(Z)'] - max_logz_value
                    delta_bic = row['Median BIC'] - min_bic_value

                    html_content += f'                    <tr>\n'
                    html_content += f'                        <td>{row["Planets"]}</td>\n'
                    html_content += f'                        <td class="{logz_class}">{row["log(Z)"]:.2f} ± {row["log(Z) error"]:.2f}</td>\n'
                    html_content += f'                        <td class="{logz_class}">{delta_logz:.2f}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{row["Median BIC"]:.2f}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{delta_bic:.2f}</td>\n'
                    html_content += f'                        <td>{row["Efficiency %"]:.2f}%</td>\n'
                    html_content += f'                        <td>{row["N calls"]:,}</td>\n'
                    html_content += f'                    </tr>\n'

                html_content += '                </table>\n'

                # Add summary
                html_content += f'                <div class="summary">\n'
                html_content += f'                    <strong>Best by log(Z):</strong> {group.loc[max_logz_idx, "Planets"]} (log(Z) = {max_logz_value:.2f})<br>\n'
                html_content += f'                    <strong>Best by BIC:</strong> {group.loc[min_bic_idx, "Planets"]} (BIC = {min_bic_value:.2f})\n'
                html_content += f'                </div>\n'
                html_content += f'            </div>\n'

                # Store best model info for summary table
                best_model_data = group.loc[max_logz_idx]
                best_models_info.append({
                    'Configuration': config_name,
                    'Best Model': best_model_data['Planets'],
                    'log(Z)': best_model_data['log(Z)'],
                    'BIC': best_model_data['Median BIC'],
                    'Orbital Parameters': best_model_data['Orbital Parameters'],
                    'Activity Parameters': best_model_data['Activity Parameters']
                })

            # Add orbital & activity parameters summary section
            if best_models_info:
                html_content += '            <div class="params-summary-section">\n'
                html_content += '                <h2>Parameters Summary (Best Models by log(Z))</h2>\n'

                # Table 1: Orbital Parameters
                html_content += '                <h3>Table 1: Orbital Parameters</h3>\n'
                html_content += '                <table class="params-table">\n'
                html_content += '                    <tr><th>Config</th><th>Model</th><th>Planet</th>'
                html_content += '<th>P (days)</th><th>K (m/s)</th><th>mean_long (°)</th><th>e</th><th>ω (°)</th></tr>\n'

                for model_info in best_models_info:
                    config_name = model_info['Configuration']
                    best_model = model_info['Best Model']
                    orbital_params = model_info['Orbital Parameters']

                    planets = sorted(orbital_params.keys()) if orbital_params else []

                    if not planets:
                        html_content += f'                    <tr>\n'
                        html_content += f'                        <td>{config_name}</td><td>{best_model}</td><td>-</td>\n'
                        html_content += f'                        <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>\n'
                        html_content += f'                    </tr>\n'
                    else:
                        for i, planet in enumerate(planets):
                            planet_params = orbital_params[planet]

                            # Use original string values to preserve decimal places from log file
                            p_val = planet_params.get('P', {}).get('value_str', '-') if 'P' in planet_params else '-'
                            k_val = planet_params.get('K', {}).get('value_str', '-') if 'K' in planet_params else '-'
                            ml_val = planet_params.get('mean_long', {}).get('value_str', '-') if 'mean_long' in planet_params else '-'
                            e_val_str = planet_params.get('e', {}).get('value_str') if 'e' in planet_params else None
                            omega_val_str = planet_params.get('omega', {}).get('value_str') if 'omega' in planet_params else None

                            e_str = e_val_str if e_val_str is not None else '-'
                            omega_str = omega_val_str if omega_val_str is not None else '-'

                            html_content += f'                    <tr>\n'
                            if i == 0:
                                html_content += f'                        <td rowspan="{len(planets)}">{config_name}</td>\n'
                                html_content += f'                        <td rowspan="{len(planets)}">{best_model}</td>\n'
                            html_content += f'                        <td>{planet}</td>\n'
                            html_content += f'                        <td>{p_val}</td><td>{k_val}</td><td>{ml_val}</td>\n'
                            html_content += f'                        <td>{e_str}</td><td>{omega_str}</td>\n'
                            html_content += f'                    </tr>\n'

                html_content += '                </table>\n'

                # Table 2: Activity Parameters
                html_content += '                <h3>Table 2: Activity Parameters</h3>\n'
                html_content += '                <table class="params-table">\n'
                html_content += '                    <tr><th>Config</th><th>Model</th>'
                html_content += '<th>Prot (days)</th><th>Pdec (days)</th><th>Oamp</th></tr>\n'

                for model_info in best_models_info:
                    config_name = model_info['Configuration']
                    best_model = model_info['Best Model']
                    activity_params = model_info['Activity Parameters']

                    # Use original string values to preserve decimal places from log file
                    prot_val = activity_params.get('Prot', {}).get('value_str', '-') if 'Prot' in activity_params else '-'
                    pdec_val = activity_params.get('Pdec', {}).get('value_str', '-') if 'Pdec' in activity_params else '-'
                    oamp_val = activity_params.get('Oamp', {}).get('value_str', '-') if 'Oamp' in activity_params else '-'

                    html_content += f'                    <tr>\n'
                    html_content += f'                        <td>{config_name}</td><td>{best_model}</td>\n'
                    html_content += f'                        <td>{prot_val}</td><td>{pdec_val}</td><td>{oamp_val}</td>\n'
                    html_content += f'                    </tr>\n'

                html_content += '                </table>\n'
                html_content += '            </div>\n'

            html_content += '        </div>\n'

        html_content += """
    </div>
</body>
</html>
"""

        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Export Best-Fit Planet Parameters CSV files for ESSP submission
        print(f"\n{'='*100}")
        print("Exporting Best-Fit Planet Parameters CSV files...")
        planet_fit_files = export_planet_fit_csv(datasets, df, group_name="DTU-Padova-PSU_dynesty", reference_epoch=59334.700184, output_dir=output_dir)
        
        # Export directory names for best models
        print(f"\nExporting best model directory names...")
        # Extract folder name from search_directory for output filename
        if search_directory:
            folder_name = os.path.basename(os.path.normpath(search_directory))
            output_filename = f"{folder_name}.csv"
        else:
            output_filename = None
        dir_file = export_best_model_directories(datasets, df, output_filename=output_filename, output_dir=output_dir)

        print(f"\n{'='*100}")
        print(f"Results exported to:")
        print(f"  • HTML: {html_filepath} (for visual display)")
        if planet_fit_files:
            print(f"  • Planet Fit CSV files: {len(planet_fit_files)} file(s) created")
            print("    Format: <<Dataset>>_DTU-Padova-PSU_dynesty_<<Method Name>>_planetFit.csv")
            print("    Note: t0 is calculated using reference_epoch=59334.700184. Adjust if needed based on your PyORBIT configuration.")
        if dir_file:
            print(f"  • Directory list: {dir_file} (directory names only)")
        print(f"\nKey Metrics:")
        print(f"  • log(Z): Bayesian evidence (higher is better)")
        print(f"  • Δlog(Z): Difference from best model (0 = best)")
        print(f"  • BIC: Bayesian Information Criterion (lower is better)")
        print(f"  • Efficiency: Sampling efficiency (higher is better, typically 1-5%)")
        print(f"  • Parameters shown for best models (by log(Z)) in each configuration")
        print(f"  • Planet Fit CSV files contain best-fit parameters from best log(Z) model")
        print(f"    with columns: K [m/s], P [d], t0 [eMJD], e, w [deg]")
        print(f"{'='*100}\n")

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    import glob

    # Allow specifying a directory as command line argument
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
        if not os.path.exists(search_directory):
            print(f"Error: Directory '{search_directory}' does not exist.")
            sys.exit(1)
        if not os.path.isdir(search_directory):
            print(f"Error: '{search_directory}' is not a directory.")
            sys.exit(1)
        print(f"Searching for .log files in: {search_directory}")
    else:
        search_directory = "."
        print("Searching for .log files in current directory and subdirectories...")

    # Find all .log files
    all_log_files = glob.glob(os.path.join(search_directory, '**/*.log'), recursive=True)

    # Deduplicate by basename, preferring less nested files
    file_groups = {}
    for log_file in all_log_files:
        basename = os.path.basename(log_file)
        if basename not in file_groups:
            file_groups[basename] = []
        file_groups[basename].append(log_file)

    log_files = []
    for basename, files in file_groups.items():
        if len(files) == 1:
            log_files.append(files[0])
        else:
            best_file = min(files, key=lambda f: f.count(os.sep))
            log_files.append(best_file)

    if not log_files:
        print("No .log files found in the current directory or subdirectories.")
    else:
        print(f"Found {len(log_files)} unique log files to analyze (filtered from {len(all_log_files)} total).\n")
        analyze_and_display_dynesty(log_files, search_directory=search_directory)
