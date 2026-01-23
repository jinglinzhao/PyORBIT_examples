#!/usr/bin/env python3
"""
PyORBIT Log File Parser and Analyzer
Extracts BIC, convergence statistics, orbital parameters, and activity parameters
from PyORBIT MCMC log files.
"""

import re
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

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

def export_planet_fit_csv(datasets, df, group_name="DTU-Padova-PSU", reference_epoch=0.0, fit_type="multiple", output_dir="."):
    """
    Export Best-Fit Planet Parameters CSV files for ESSP submission.
    
    Creates CSV files named: <<Dataset>>_<<Group Name>>_<<Fit Type>>_<<Method Name>>_planetFit.csv
    with columns: K [m/s], P [d], t0 [eMJD], e, w [deg]
    
    Args:
        datasets (list): List of dataset names
        df (DataFrame): DataFrame containing parsed log file data
        group_name (str): Group name for file naming (default: "DTU-Padova-PSU")
        reference_epoch (float): Reference epoch in eMJD for t0 calculation (default: 0.0)
        fit_type (str): Fit type - "multiple" or "single" (default: "multiple")
        output_dir (str): Output directory path (default: ".")
    """
    exported_files = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')
        
        for config_name, group in grouped:
            # Find the best model (lowest BIC)
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            min_bic_idx = group['Median BIC'].idxmin()
            best_model_data = group.loc[min_bic_idx]
            
            orbital_params = best_model_data['Orbital Parameters']
            
            # Prepare data for CSV (even for 0-planet models)
            planet_rows = []
            planets = sorted(orbital_params.keys()) if orbital_params else []
            
            # Process planets if they exist
            if planets:
                for planet in planets:
                    planet_params = orbital_params[planet]
                    
                    # Extract parameters
                    K = planet_params.get('K', {}).get('value')
                    P = planet_params.get('P', {}).get('value')
                    e = planet_params.get('e', {}).get('value')
                    omega = planet_params.get('omega', {}).get('value')
                    mean_long = planet_params.get('mean_long', {}).get('value')
                    
                    # Skip if essential parameters are missing
                    if K is None or P is None:
                        continue
                    
                    # Calculate t0 from mean longitude
                    if mean_long is not None and omega is not None:
                        t0 = calculate_t0_from_mean_long(mean_long, omega, P, reference_epoch)
                    elif mean_long is not None:
                        # If omega is missing but mean_long exists, assume circular orbit (omega = 0)
                        t0 = calculate_t0_from_mean_long(mean_long, 0.0, P, reference_epoch)
                    else:
                        # If mean_long is missing, we can't calculate t0
                        # Set to None (will be written as empty or NaN)
                        t0 = None
                    
                    # Set defaults for optional parameters
                    if e is None:
                        e = 0.0  # Circular orbit
                    if omega is None:
                        omega = 0.0
                    
                    # Create row with numeric values
                    row = {
                        'K [m/s]': K,
                        'P [d]': P,
                        't0 [eMJD]': t0 if t0 is not None else np.nan,
                        'e': e,
                        'w [deg]': omega
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
            
            # Clean config_name: remove '_single' suffix if fit_type is 'single' to avoid duplication
            clean_config_name = config_name
            if fit_type == 'single' and config_name.endswith('_single'):
                clean_config_name = config_name[:-7]  # Remove '_single' suffix (7 characters)
            
            # Create filename: DS1_DTU-Padova-PSU_emcee_<fit_type>_<config_name>_planetFit.csv
            filename = f"{dataset}_{group_name}_{fit_type}_{clean_config_name}_planetFit.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Export to CSV with formatting (6 decimal places)
            # For empty DataFrames, this will create a file with just headers
            planet_df.to_csv(filepath, index=False, encoding='utf-8', float_format='%.6f')
            exported_files.append(filepath)
            
            num_planets = len(planet_rows)
            if num_planets == 0:
                print(f"  Exported: {filepath} (0-planet model, headers only)")
            else:
                print(f"  Exported: {filepath} ({num_planets} planet(s))")
    
    return exported_files

def export_best_model_directories(datasets, df, output_filename=None, output_dir="."):
    """
    Export a simple list of directory names where best models (lowest BIC) were found.
    
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
            # Find the best model (lowest BIC)
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            min_bic_idx = group['Median BIC'].idxmin()
            best_model_data = group.loc[min_bic_idx]
            
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

def parse_log_file(filepath):
    """
    Parses a PyORBIT log file to extract Median BIC, convergence status, and orbital parameters.

    Args:
        filepath (str): The full path to the log file.

    Returns:
        dict: A dictionary containing the model name, BIC, convergence status, and orbital parameters.
              Returns None if the required lines are not found.
    """
    median_bic = None
    gelman_rubin_values = []
    orbital_parameters = {}
    activity_parameters = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract Median BIC
            bic_match = re.search(r'Median BIC\s+\(using likelihood\)\s*=\s*(-?[\d\.]+)', content)
            if bic_match:
                median_bic = float(bic_match.group(1))
            
            # Extract Gelman-Rubin values - MORE COMPREHENSIVE PATTERN
            # This will capture patterns like "b_sre_coso", "b_sre_sino", "b_P", "activity_Prot", etc.
            gr_pattern = r'Gelman-Rubin:\s+(\d+)\s+([\d\.]+)\s+([a-zA-Z_][a-zA-Z_0-9]*(?:_[a-zA-Z_][a-zA-Z_0-9]*)*)\s*$'
            gr_matches = re.findall(gr_pattern, content, re.MULTILINE)
            
            # Store GR values in a dictionary keyed by parameter name
            gr_dict = {}
            for match in gr_matches:
                param_name = match[2]
                gr_value = float(match[1])
                gr_dict[param_name] = gr_value
                gelman_rubin_values.append(gr_value)
            
            # Extract orbital and activity parameters from the LAST "Statistics on the model parameters" section
            lines = content.split('\n')
            
            # Find the LAST occurrence of "Statistics on the model parameters obtained from the posteriors samples"
            last_stats_idx = -1
            for i, line in enumerate(lines):
                if "Statistics on the model parameters obtained from the posteriors samples" in line:
                    last_stats_idx = i
            
            if last_stats_idx != -1:
                # Now parse from this section onwards
                current_planet = None
                in_activity_section = False
                
                for i in range(last_stats_idx, len(lines)):
                    line = lines[i]
                    
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
                    
                    # Parse parameter lines (format: "P                     2.914225")
                    param_match = re.match(r'^([A-Za-z_]+)\s+([-\d\.]+)\s*$', line.strip())
                    if param_match:
                        param_name = param_match.group(1)
                        param_value = float(param_match.group(2))
                        
                        if in_activity_section:
                            # Store activity parameters with GR values
                            full_param_name = f'activity_{param_name}'
                            gr_value = gr_dict.get(full_param_name, None)
                            activity_parameters[param_name] = {
                                'value': param_value,
                                'gelman_rubin': gr_value
                            }
                        elif current_planet:
                            # Store orbital parameters for the current planet with GR values
                            full_param_name = f'{current_planet}_{param_name}'
                            gr_value = gr_dict.get(full_param_name, None)
                            orbital_parameters[current_planet][param_name] = {
                                'value': param_value,
                                'gelman_rubin': gr_value
                            }
                
                # Now add sre_coso and sre_sino GR values to orbital parameters
                # These won't have values in the statistics section, only GR values
                for planet in orbital_parameters.keys():
                    # Check for sre_coso
                    coso_key = f'{planet}_sre_coso'
                    if coso_key in gr_dict:
                        if 'sre_coso' not in orbital_parameters[planet]:
                            orbital_parameters[planet]['sre_coso'] = {}
                        orbital_parameters[planet]['sre_coso']['gelman_rubin'] = gr_dict[coso_key]
                    
                    # Check for sre_sino
                    sino_key = f'{planet}_sre_sino'
                    if sino_key in gr_dict:
                        if 'sre_sino' not in orbital_parameters[planet]:
                            orbital_parameters[planet]['sre_sino'] = {}
                        orbital_parameters[planet]['sre_sino']['gelman_rubin'] = gr_dict[sino_key]

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None

    if median_bic is not None:
        # Calculate convergence statistics
        if gelman_rubin_values:
            converged_count = sum(1 for gr in gelman_rubin_values if gr < 1.1)
            total_count = len(gelman_rubin_values)
            convergence_pct = (converged_count / total_count * 100) if total_count > 0 else 0
            max_gr = max(gelman_rubin_values)
        else:
            convergence_pct = 0
            max_gr = None
        
        # Extract model name details from the filename
        basename = os.path.basename(filepath)
        # Remove the common prefix and .log extension
        cleaned_name = basename.replace('configuration_file_emcee_run_', '').replace('.log', '')
        
        # Extract dataset (DS1, DS2, etc.) and the rest
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
            'Median BIC': median_bic,
            'Convergence %': convergence_pct,
            'Max GR': max_gr,
            'Orbital Parameters': orbital_parameters,
            'Activity Parameters': activity_parameters,
            'File': basename,
            'Directory': directory_name
        }
    return None

def analyze_and_display(log_files, search_directory=None, fit_type="multiple"):
    """
    Analyzes a list of log files and prints a formatted comparison table.
    Also exports results to CSV and HTML.

    Args:
        log_files (list): A list of paths to the log files.
        search_directory (str, optional): The directory path where log files were searched.
                                          Used to name output files.
        fit_type (str): Fit type - "multiple" or "single" (default: "multiple")
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
    
    print(f"Processing {len(log_files)} log files...")
    for log_file in log_files:
        data = parse_log_file(log_file)
        if data:
            all_data.append(data)
        else:
            failed_files.append(log_file)
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files could not be parsed (missing Median BIC data):")
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
    print("--- Model Comparison by Dataset and Configuration ---\n")

    # Prepare data for CSV export
    export_data = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        grouped = dataset_df.groupby('Configuration')
        
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset}")
        print(f"{'='*80}\n")
        
        # Store best models info for orbital parameters summary
        best_models_info = []
        
        for config_name, group in grouped:
            print(f"--- Configuration: {config_name} ---\n")
            
            # Sort by number of planets
            group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
            group = group.sort_values('Planets')
            
            # Find the index of the minimum BIC for highlighting
            min_bic_idx = group['Median BIC'].idxmin()
            min_bic_value = group.loc[min_bic_idx, 'Median BIC']
            
            # Create display dataframe
            display_group = group[['Planets', 'Median BIC', 'Convergence %', 'Max GR']].copy()
            # Add ΔBIC column to display
            display_group['ΔBIC'] = group['Median BIC'] - min_bic_value
            
            # Add preferred model indicators and ΔBIC calculation
            group_copy = group.copy()
            group_copy['Preferred_BIC'] = group_copy.index == min_bic_idx
            group_copy['ΔBIC'] = group_copy['Median BIC'] - min_bic_value
            group_copy['Dataset'] = dataset
            group_copy['Configuration'] = config_name
            
            # Reorder columns for export
            export_group = group_copy[['Dataset', 'Configuration', 'Planets', 'Median BIC', 'ΔBIC',
                                      'Convergence %', 'Max GR', 'Preferred_BIC', 'File']]
            export_data.append(export_group)
            
            # Print simple table
            print(display_group.to_string(index=False))
            print(f"\nBest BIC: {group.loc[min_bic_idx, 'Planets']} model (BIC = {group.loc[min_bic_idx, 'Median BIC']:.2f})")
            print("\n" + "-"*80 + "\n")
            
            # Store best model info for summary table
            best_model_data = group.loc[min_bic_idx]
            best_models_info.append({
                'Configuration': config_name,
                'Best Model': best_model_data['Planets'],
                'BIC': best_model_data['Median BIC'],
                'Convergence %': best_model_data['Convergence %'],
                'Max GR': best_model_data['Max GR'],
                'Orbital Parameters': best_model_data['Orbital Parameters'],
                'Activity Parameters': best_model_data['Activity Parameters']
            })
        
        # Display orbital & activity parameters summary table for this dataset
        if best_models_info:
            print(f"\n{'='*160}")
            print(f"ORBITAL & ACTIVITY PARAMETERS SUMMARY FOR {dataset}")
            print(f"{'='*160}\n")
            
            # Print in two separate tables for better readability
            
            # Table 1: Orbital Parameters (Combined)
            print("TABLE 1: ORBITAL PARAMETERS\n")
            print(f"{'Config':<20} {'Model':<8} {'Planet':<8} {'P (days)':<12} {'P_GR':<8} {'K (m/s)':<10} {'K_GR':<8} {'mean_long':<12} {'ml_GR':<8} {'e':<10} {'ω (deg)':<10} {'coso_GR':<10} {'sino_GR':<10}")
            print("-" * 160)
            
            for model_info in best_models_info:
                config_name = model_info['Configuration']
                best_model = model_info['Best Model']
                orbital_params = model_info['Orbital Parameters']
                
                planets = sorted(orbital_params.keys()) if orbital_params else []
                
                if not planets:
                    # 0-planet model
                    print(f"{config_name:<20} {best_model:<8} {'-':<8} {'-':<12} {'-':<8} {'-':<10} {'-':<8} {'-':<12} {'-':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
                else:
                    for planet in planets:
                        planet_params = orbital_params[planet]
                        
                        p_val = f"{planet_params.get('P', {}).get('value', 0):.4f}" if 'P' in planet_params else '-'
                        p_gr = f"{planet_params.get('P', {}).get('gelman_rubin', 0):.3f}" if 'P' in planet_params and planet_params['P'].get('gelman_rubin') is not None else '-'
                        
                        k_val = f"{planet_params.get('K', {}).get('value', 0):.4f}" if 'K' in planet_params else '-'
                        k_gr = f"{planet_params.get('K', {}).get('gelman_rubin', 0):.3f}" if 'K' in planet_params and planet_params['K'].get('gelman_rubin') is not None else '-'
                        
                        ml_val = f"{planet_params.get('mean_long', {}).get('value', 0):.2f}" if 'mean_long' in planet_params else '-'
                        ml_gr = f"{planet_params.get('mean_long', {}).get('gelman_rubin', 0):.3f}" if 'mean_long' in planet_params and planet_params['mean_long'].get('gelman_rubin') is not None else '-'
                        
                        # Get e and omega values (no GR columns)
                        e_val = planet_params.get('e', {}).get('value') if 'e' in planet_params else None
                        omega_val = planet_params.get('omega', {}).get('value') if 'omega' in planet_params else None
                        
                        # Get sre_coso and sre_sino GR values
                        coso_gr = planet_params.get('sre_coso', {}).get('gelman_rubin') if 'sre_coso' in planet_params else None
                        sino_gr = planet_params.get('sre_sino', {}).get('gelman_rubin') if 'sre_sino' in planet_params else None
                        
                        e_str = f"{e_val:.4f}" if e_val is not None else '-'
                        omega_str = f"{omega_val:.2f}" if omega_val is not None else '-'
                        coso_gr_str = f"{coso_gr:.3f}" if coso_gr is not None else '-'
                        sino_gr_str = f"{sino_gr:.3f}" if sino_gr is not None else '-'
                        
                        print(f"{config_name:<20} {best_model:<8} {planet:<8} {p_val:<12} {p_gr:<8} {k_val:<10} {k_gr:<8} {ml_val:<12} {ml_gr:<8} {e_str:<10} {omega_str:<10} {coso_gr_str:<10} {sino_gr_str:<10}")
            
            print()
            
            # Table 2: Activity Parameters
            print("TABLE 2: ACTIVITY PARAMETERS\n")
            print(f"{'Config':<20} {'Model':<8} {'Prot (days)':<15} {'Prot_GR':<10} {'Pdec (days)':<15} {'Pdec_GR':<10} {'Oamp':<10} {'Oamp_GR':<10}")
            print("-" * 120)
            
            for model_info in best_models_info:
                config_name = model_info['Configuration']
                best_model = model_info['Best Model']
                activity_params = model_info['Activity Parameters']
                
                prot_val = f"{activity_params.get('Prot', {}).get('value', 0):.4f}" if 'Prot' in activity_params else '-'
                prot_gr = f"{activity_params.get('Prot', {}).get('gelman_rubin', 0):.3f}" if 'Prot' in activity_params and activity_params['Prot'].get('gelman_rubin') is not None else '-'
                
                pdec_val = f"{activity_params.get('Pdec', {}).get('value', 0):.4f}" if 'Pdec' in activity_params else '-'
                pdec_gr = f"{activity_params.get('Pdec', {}).get('gelman_rubin', 0):.3f}" if 'Pdec' in activity_params and activity_params['Pdec'].get('gelman_rubin') is not None else '-'
                
                oamp_val = f"{activity_params.get('Oamp', {}).get('value', 0):.4f}" if 'Oamp' in activity_params else '-'
                oamp_gr = f"{activity_params.get('Oamp', {}).get('gelman_rubin', 0):.3f}" if 'Oamp' in activity_params and activity_params['Oamp'].get('gelman_rubin') is not None else '-'
                
                print(f"{config_name:<20} {best_model:<8} {prot_val:<15} {prot_gr:<10} {pdec_val:<15} {pdec_gr:<10} {oamp_val:<10} {oamp_gr:<10}")
            
            print("\n" + "="*160 + "\n")

    # Export to CSV and HTML
    if export_data:
        combined_df = pd.concat(export_data, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export CSV for data analysis with UTF-8 encoding
        csv_filename = f"model_comparison_{timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        combined_df.to_csv(csv_filepath, index=False, encoding='utf-8')
        
        # Export HTML with formatting for visual display
        html_filename = f"model_comparison_{timestamp}.html"
        html_filepath = os.path.join(output_dir, html_filename)
        
        # Create HTML content with improved layout
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Comparison Results</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
            color: #2c3e50;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 30px;
            font-size: 32px;
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
            letter-spacing: 0.5px;
        }
        .config-section {
            margin: 0;
            padding: 25px 30px;
            border-bottom: 1px solid #ecf0f1;
        }
        h2 { 
            color: #34495e;
            margin: 20px 0 15px 0;
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
            font-size: 13px;
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
        .gr-converged {
            color: #2e7d32;
            font-weight: 600;
        }
        .gr-not-converged {
            color: #c62828;
            font-weight: 600;
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
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #34495e;
            font-size: 15px;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover td {
            background-color: #f8f9fa;
        }
        .highlight-bic {
            background-color: #e8f5e9 !important;
            font-weight: 600;
            color: #2e7d32;
        }
        .convergence-good {
            color: #2e7d32;
            font-weight: 600;
        }
        .convergence-warning {
            color: #f57c00;
            font-weight: 600;
        }
        .convergence-bad {
            color: #c62828;
            font-weight: 600;
        }
        .summary {
            margin: 15px 0 0 0;
            padding: 15px 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            font-size: 15px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>PyORBIT Model Comparison Results</h1>
        
        <div class="legend">
            <h3>Legend</h3>
            <div class="legend-item"><span class="badge badge-green">Green highlight</span> = Best BIC in configuration</div>
            <div class="legend-item"><strong>Convergence %:</strong> Percentage of parameters with Gelman-Rubin < 1.1</div>
            <div class="legend-item"><strong>Max GR:</strong> Maximum Gelman-Rubin value across all parameters</div>
            <div class="legend-item">
                <span class="convergence-good">Green</span> = 100% converged | 
                <span class="convergence-warning">Orange</span> = 90-99% converged | 
                <span class="convergence-bad">Red</span> = <90% converged
            </div>
            <div class="legend-item"><strong>GR values:</strong> <span class="gr-converged">Green</span> = GR < 1.1 (converged) | <span class="gr-not-converged">Red</span> = GR ≥ 1.1 (not converged)</div>
            <div class="legend-item"><strong>Parameters split into 2 tables:</strong> (1) Orbital: P, K, mean_long, e, ω, coso_GR, sino_GR | (2) Activity: Prot, Pdec, Oamp</div>
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
                
                # Sort by number of planets
                group['Planets'] = pd.Categorical(group['Planets'], categories=['0p', '1p', '2p', '3p'], ordered=True)
                group = group.sort_values('Planets')
                
                min_bic_idx = group['Median BIC'].idxmin()
                
                # Build table
                html_content += '                <table>\n'
                html_content += '                    <tr><th>Planets</th><th>Median BIC</th><th>ΔBIC</th><th>Convergence %</th><th>Max GR</th></tr>\n'
                
                for idx, row in group.iterrows():
                    bic_class = 'highlight-bic' if idx == min_bic_idx else ''
                    
                    # Determine convergence color
                    conv_pct = row['Convergence %']
                    if conv_pct == 100:
                        conv_class = 'convergence-good'
                    elif conv_pct >= 90:
                        conv_class = 'convergence-warning'
                    else:
                        conv_class = 'convergence-bad'
                    
                    max_gr_str = f"{row['Max GR']:.4f}" if row['Max GR'] is not None else "-"
                    
                    # Calculate ΔBIC for this row
                    delta_bic = row['Median BIC'] - group.loc[min_bic_idx, 'Median BIC']
                    
                    html_content += f'                    <tr>\n'
                    html_content += f'                        <td>{row["Planets"]}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{row["Median BIC"]:.2f}</td>\n'
                    html_content += f'                        <td class="{bic_class}">{delta_bic:.2f}</td>\n'
                    html_content += f'                        <td class="{conv_class}">{conv_pct:.1f}%</td>\n'
                    html_content += f'                        <td>{max_gr_str}</td>\n'
                    html_content += f'                    </tr>\n'
                
                html_content += '                </table>\n'
                
                # Add summary
                html_content += f'                <div class="summary">\n'
                html_content += f'                    <strong>Best BIC:</strong> {group.loc[min_bic_idx, "Planets"]} model (BIC = {group.loc[min_bic_idx, "Median BIC"]:.2f})\n'
                html_content += f'                </div>\n'
                html_content += f'            </div>\n'
                
                # Store best model info for summary table
                best_model_data = group.loc[min_bic_idx]
                best_models_info.append({
                    'Configuration': config_name,
                    'Best Model': best_model_data['Planets'],
                    'BIC': best_model_data['Median BIC'],
                    'Orbital Parameters': best_model_data['Orbital Parameters'],
                    'Activity Parameters': best_model_data['Activity Parameters']
                })
            
            # Add orbital & activity parameters summary section with 2 separate tables
            if best_models_info:
                html_content += '            <div class="params-summary-section">\n'
                html_content += '                <h2>Parameters Summary (Best Models)</h2>\n'
                
                def format_gr(gr_value):
                    if gr_value is None:
                        return '-', ''
                    try:
                        gr_float = float(gr_value)
                        gr_class = 'gr-converged' if gr_float < 1.1 else 'gr-not-converged'
                        return f'{gr_float:.3f}', gr_class
                    except:
                        return '-', ''
                
                # Table 1: Orbital Parameters (Combined)
                html_content += '                <h3>Table 1: Orbital Parameters</h3>\n'
                html_content += '                <table class="params-table">\n'
                html_content += '                    <tr><th>Config</th><th>Model</th><th>Planet</th>'
                html_content += '<th>P (days)</th><th>P_GR</th><th>K (m/s)</th><th>K_GR</th>'
                html_content += '<th>mean_long (°)</th><th>ml_GR</th>'
                html_content += '<th>e</th><th>ω (°)</th>'
                html_content += '<th>coso_GR</th><th>sino_GR</th></tr>\n'
                
                for model_info in best_models_info:
                    config_name = model_info['Configuration']
                    best_model = model_info['Best Model']
                    orbital_params = model_info['Orbital Parameters']
                    
                    planets = sorted(orbital_params.keys()) if orbital_params else []
                    
                    if not planets:
                        html_content += f'                    <tr>\n'
                        html_content += f'                        <td>{config_name}</td><td>{best_model}</td><td>-</td>\n'
                        html_content += f'                        <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>\n'
                        html_content += f'                        <td>-</td><td>-</td><td>-</td><td>-</td>\n'
                        html_content += f'                    </tr>\n'
                    else:
                        for i, planet in enumerate(planets):
                            planet_params = orbital_params[planet]
                            
                            p_val = f"{planet_params.get('P', {}).get('value', 0):.4f}" if 'P' in planet_params else '-'
                            p_gr, p_gr_class = format_gr(planet_params.get('P', {}).get('gelman_rubin'))
                            
                            k_val = f"{planet_params.get('K', {}).get('value', 0):.4f}" if 'K' in planet_params else '-'
                            k_gr, k_gr_class = format_gr(planet_params.get('K', {}).get('gelman_rubin'))
                            
                            ml_val = f"{planet_params.get('mean_long', {}).get('value', 0):.2f}" if 'mean_long' in planet_params else '-'
                            ml_gr, ml_gr_class = format_gr(planet_params.get('mean_long', {}).get('gelman_rubin'))
                            
                            # Get e and omega values (NO GR columns)
                            e_val = planet_params.get('e', {}).get('value') if 'e' in planet_params else None
                            omega_val = planet_params.get('omega', {}).get('value') if 'omega' in planet_params else None
                            
                            # Get sre_coso and sre_sino GR values
                            coso_gr, coso_gr_class = format_gr(planet_params.get('sre_coso', {}).get('gelman_rubin'))
                            sino_gr, sino_gr_class = format_gr(planet_params.get('sre_sino', {}).get('gelman_rubin'))
                            
                            e_str = f"{e_val:.4f}" if e_val is not None else '-'
                            omega_str = f"{omega_val:.2f}" if omega_val is not None else '-'
                            
                            html_content += f'                    <tr>\n'
                            if i == 0:
                                html_content += f'                        <td rowspan="{len(planets)}">{config_name}</td>\n'
                                html_content += f'                        <td rowspan="{len(planets)}">{best_model}</td>\n'
                            html_content += f'                        <td>{planet}</td>\n'
                            html_content += f'                        <td>{p_val}</td><td class="{p_gr_class}">{p_gr}</td>\n'
                            html_content += f'                        <td>{k_val}</td><td class="{k_gr_class}">{k_gr}</td>\n'
                            html_content += f'                        <td>{ml_val}</td><td class="{ml_gr_class}">{ml_gr}</td>\n'
                            html_content += f'                        <td>{e_str}</td><td>{omega_str}</td>\n'
                            html_content += f'                        <td class="{coso_gr_class}">{coso_gr}</td><td class="{sino_gr_class}">{sino_gr}</td>\n'
                            html_content += f'                    </tr>\n'
                
                html_content += '                </table>\n'
                
                # Table 2: Activity Parameters
                html_content += '                <h3>Table 2: Activity Parameters</h3>\n'
                html_content += '                <table class="params-table">\n'
                html_content += '                    <tr><th>Config</th><th>Model</th>'
                html_content += '<th>Prot (days)</th><th>Prot_GR</th><th>Pdec (days)</th><th>Pdec_GR</th>'
                html_content += '<th>Oamp</th><th>Oamp_GR</th></tr>\n'
                
                for model_info in best_models_info:
                    config_name = model_info['Configuration']
                    best_model = model_info['Best Model']
                    activity_params = model_info['Activity Parameters']
                    
                    prot_val = f"{activity_params.get('Prot', {}).get('value', 0):.4f}" if 'Prot' in activity_params else '-'
                    prot_gr, prot_gr_class = format_gr(activity_params.get('Prot', {}).get('gelman_rubin'))
                    
                    pdec_val = f"{activity_params.get('Pdec', {}).get('value', 0):.4f}" if 'Pdec' in activity_params else '-'
                    pdec_gr, pdec_gr_class = format_gr(activity_params.get('Pdec', {}).get('gelman_rubin'))
                    
                    oamp_val = f"{activity_params.get('Oamp', {}).get('value', 0):.4f}" if 'Oamp' in activity_params else '-'
                    oamp_gr, oamp_gr_class = format_gr(activity_params.get('Oamp', {}).get('gelman_rubin'))
                    
                    html_content += f'                    <tr>\n'
                    html_content += f'                        <td>{config_name}</td><td>{best_model}</td>\n'
                    html_content += f'                        <td>{prot_val}</td><td class="{prot_gr_class}">{prot_gr}</td>\n'
                    html_content += f'                        <td>{pdec_val}</td><td class="{pdec_gr_class}">{pdec_gr}</td>\n'
                    html_content += f'                        <td>{oamp_val}</td><td class="{oamp_gr_class}">{oamp_gr}</td>\n'
                    html_content += f'                    </tr>\n'
                
                html_content += '                </table>\n'
                html_content += '            </div>\n'
            
            html_content += '        </div>\n'
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML file with UTF-8 encoding
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Export Best-Fit Planet Parameters CSV files for ESSP submission
        print(f"\n{'='*80}")
        print("Exporting Best-Fit Planet Parameters CSV files...")
        planet_fit_files = export_planet_fit_csv(datasets, df, group_name="DTU-Padova-PSU_emcee", reference_epoch=59334.700184, fit_type=fit_type, output_dir=output_dir)
        
        # Export directory names for best models
        print(f"\nExporting best model directory names...")
        # Extract folder name from search_directory for output filename
        if search_directory:
            folder_name = os.path.basename(os.path.normpath(search_directory))
            output_filename = f"{folder_name}.csv"
        else:
            output_filename = None
        dir_file = export_best_model_directories(datasets, df, output_filename=output_filename, output_dir=output_dir)
        
        print(f"\n{'='*80}")
        print(f"Results exported to:")
        print(f"  - CSV: {csv_filepath} (for data analysis)")
        print(f"  - HTML: {html_filepath} (for visual display with formatting)")
        if planet_fit_files:
            print(f"  - Planet Fit CSV files: {len(planet_fit_files)} file(s) created")
            print(f"    Format: <<Dataset>>_DTU-Padova-PSU_emcee_{fit_type}_<<Method Name>>_planetFit.csv")
            print("    Note: t0 is calculated using reference_epoch=59334.700184. Adjust if needed based on your PyORBIT configuration.")
        if dir_file:
            print(f"  - Directory list: {dir_file} (directory names only)")
        print("\nNotes:")
        print("  - Preferred_BIC=True indicates the model with lowest BIC in that configuration")
        print("  - Convergence % shows percentage of parameters with Gelman-Rubin < 1.1")
        print("  - Parameters split into 2 tables for better readability:")
        print("    1. Orbital: P, K, mean_long, e, ω, coso_GR, sino_GR")
        print("    2. Activity: Prot, Pdec, Oamp")
        print("  - GR values: Green = converged (< 1.1), Red = not converged (≥ 1.1)")
        print("  - '-' indicates parameter not available")
        print("  - Planet Fit CSV files contain best-fit parameters from best BIC model")
        print("    with columns: K [m/s], P [d], t0 [eMJD], e, w [deg]")
        print(f"{'='*80}\n")

def main():
    """
    Main function to run the analyzer.
    Usage: python pyorbit_analyzer.py [directory] [fit_type]
    If no directory is specified, searches the current directory.
    fit_type can be 'multiple' (default) or 'single'
    """
    if len(sys.argv) > 1:
        search_directory = sys.argv[1]
    else:
        search_directory = "."
    
    # Parse fit_type argument (optional, defaults to 'multiple')
    fit_type = "multiple"
    if len(sys.argv) > 2:
        fit_type_arg = sys.argv[2].lower()
        if fit_type_arg in ['multiple', 'single']:
            fit_type = fit_type_arg
        else:
            print(f"Warning: Invalid fit_type '{sys.argv[2]}'. Using default 'multiple'.")
            print("Valid options: 'multiple' or 'single'")
    
    if not os.path.isdir(search_directory):
        print(f"Error: Directory '{search_directory}' does not exist.")
        sys.exit(1)
    
    print(f"Searching for log files in: {os.path.abspath(search_directory)}\n")
    
    # Find all .log files in the specified directory and subdirectories
    all_log_files = glob.glob(os.path.join(search_directory, '**/*.log'), recursive=True)
    
    # Filter to only configuration_file_emcee_run_*.log files
    all_log_files = [f for f in all_log_files if os.path.basename(f).startswith('configuration_file_emcee_run_') and f.endswith('.log')]
    
    # More robust duplicate detection: prefer files that are not in deeply nested subdirectories
    # Group files by basename and select the best one from each group
    file_groups = {}
    
    for log_file in all_log_files:
        basename = os.path.basename(log_file)
        if basename not in file_groups:
            file_groups[basename] = []
        file_groups[basename].append(log_file)
    
    # Select the best file from each group (prefer less nested files)
    log_files = []
    for basename, files in file_groups.items():
        if len(files) == 1:
            # No duplicates, use the single file
            log_files.append(files[0])
        else:
            # Multiple files with same basename - choose the one with least nesting
            # Calculate depth for each file and choose the shallowest
            best_file = min(files, key=lambda f: f.count(os.sep))
            log_files.append(best_file)
            print(f"Note: Found {len(files)} copies of '{basename}', using: {best_file}")
    
    if not log_files:
        print("No log files found matching the pattern 'configuration_file_emcee_run_*.log'")
        sys.exit(1)
    
    print(f"\nFound {len(log_files)} unique log files to analyze (filtered from {len(all_log_files)} total).\n")
    print(f"Using fit_type: {fit_type}\n")
    
    # Analyze and display results
    analyze_and_display(log_files, search_directory=search_directory, fit_type=fit_type)

if __name__ == "__main__":
    main()