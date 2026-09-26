#!/bin/bash

# HD88986 dynesty PyORBIT Setup Generator
# First-step models: NO GP, all instruments, 1p / 2p / 3p.
# (0p omitted: no GP and no planets means no fit is needed.)
#
# Planet initial-guess guidance (Heidari+2024):
#   b  — P~146.05 d, K~1.85 m/s, e~0.24, Tc~2458891.69
#   c  — long-period companion: P~116 yr (~4.24e4 d), e~0.46,
#        K~O(100–1000) m/s, Tc~2465000 (BJD-2400000 = 65000)
#   d  — free third Keplerian (wide search; only in 3p)
#
# Stellar priors from paper: R=1.543±0.065 Rsun, ρ=472±~36 kg/m³
#   → ρ/ρsun≈0.335, M≈(ρ/ρsun)*(R/Rsun)^3 ≈ 1.23 Msun

base_dir="/work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal"
data_dir="${base_dir}/data/processed_data"
results_dir="${base_dir}/results_HD88986_dynesty"
out_dir="${base_dir}/out_HD88986_dynesty"
scripts_dir="${base_dir}/scripts_HD88986_dynesty"

mkdir -p "${results_dir}" "${out_dir}" "${scripts_dir}"
cd "${scripts_dir}" || exit 1

# LSF configuration
queue="hpc"
email="jzhao@space.dtu.dk"

# Resources for no-GP dynesty jobs (32 cores; walltime longer than emcee)
cores_dynesty=32
threads_dynesty=$((cores_dynesty - 1))
mem_per_core_dynesty="2GB"
mem_limit_dynesty="2GB"
walltime_dynesty="48:00"

# First-step configuration axes (no GP)
data_configs=("all_instr")
gp_flag="no_gp"
planets=("1p" "2p" "3p")

# All RV instruments (paper Tables D.1–D.3 + APF)
rv_inputs=(
    "HD88986_APF_RV"
    "HD88986_ELODIE_RV"
    "HD88986_HIRES_RV"
    "HD88986_HIRES-PLUS_RV"
    "HD88986_SOPHIE_RV"
    "HD88986_SOPHIE-PLUS_RV"
)

# Reference time = earliest RV epoch in the combined set
TREF="2450420.109460"

yaml_count=0
script_count=0

echo "HD88986 dynesty PyORBIT Setup (no GP, all instruments)"
echo "======================================================"
echo "Base dir:      ${base_dir}"
echo "Data dir:      ${data_dir}"
echo "Results dir:   ${results_dir}"
echo "Output dir:    ${out_dir}"
echo "Scripts dir:   ${scripts_dir}"
echo "Cores:         ${cores_dynesty} (cpu_threads=${threads_dynesty})"
echo ""

# ------------------ YAML GENERATOR ------------------ #
generate_yaml() {
    local yaml_file="$1"
    local data_config="$2"   # all_instr
    local planet_conf="$3"   # 1p / 2p / 3p

    local num_planets
    case "${planet_conf}" in
        "1p") num_planets=1 ;;
        "2p") num_planets=2 ;;
        "3p") num_planets=3 ;;
        *) echo "Unknown planet_conf ${planet_conf}"; exit 1 ;;
    esac

    if [ "${data_config}" != "all_instr" ]; then
        echo "Unknown data_config ${data_config} (this setup is all_instr only)"
        exit 1
    fi

    # --- inputs (RV only; no GP / no activity indicators) ---
    cat > "${yaml_file}" << EOF
inputs:
EOF

    for rv_name in "${rv_inputs[@]}"; do
        cat >> "${yaml_file}" << EOF
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
      - radial_velocities
EOF
    done

    # --- common ---
    cat >> "${yaml_file}" << EOF

common:
EOF

    if [ "${num_planets}" -gt 0 ]; then
        cat >> "${yaml_file}" << EOF
  planets:
EOF
        # Planet b: published sub-Neptune (~146 d)
        if [ "${num_planets}" -ge 1 ]; then
            cat >> "${yaml_file}" << EOF
    b:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [100.0, 200.0]
        K: [0.1, 10.0]
        e: [0.00, 0.60]
EOF
        fi
        # Planet c: long-period massive companion (~116 yr)
        if [ "${num_planets}" -ge 2 ]; then
            cat >> "${yaml_file}" << EOF
    c:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [5000.0, 150000.0]
        K: [10.0, 2000.0]
        e: [0.00, 0.90]
EOF
        fi
        # Planet d: free additional Keplerian
        if [ "${num_planets}" -ge 3 ]; then
            cat >> "${yaml_file}" << EOF
    d:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [1.1, 5000.0]
        K: [0.1, 50.0]
        e: [0.00, 0.80]
EOF
        fi
    fi

    cat >> "${yaml_file}" << EOF
  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 1.23, 0.10]
        radius: ['Gaussian', 1.543, 0.065]
        density: ['Gaussian', 0.335, 0.027]

models:
  radial_velocities:
    planets:
EOF

    local planet_letters=("b" "c" "d")
    local p
    for ((p = 0; p < num_planets; p++)); do
        echo "      - ${planet_letters[$p]}" >> "${yaml_file}"
    done

    cat >> "${yaml_file}" << EOF

parameters:
  Tref: ${TREF}
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: ${threads_dynesty}

solver:
  pyde:
    ngen: 50000
    npop_mult: 6
  nested_sampling:
    nlive: 1000
    dlogz: 0.01
    nthreads: ${threads_dynesty}
    sample: 'auto'
    bound: 'multi'
    use_threading_pool: True
  recenter_bounds: True
EOF
}

# ------------------ JOB SCRIPT GENERATOR ------------------ #
generate_job_script() {
    local script_file="$1"
    local job_name="$2"
    local config_dir="$3"
    local yaml_name="$4"

    cat > "${script_file}" << EOF
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q ${queue}
### -- set the job Name --
#BSUB -J ${job_name}
### -- ask for number of cores (default: 1) --
#BSUB -n ${cores_dynesty}
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need ${mem_per_core_dynesty} of memory per core/slot --
#BSUB -R "rusage[mem=${mem_per_core_dynesty}]"
### -- specify that we want the job to get killed if it exceeds ${mem_limit_dynesty} per core/slot --
#BSUB -M ${mem_limit_dynesty}
### -- set walltime limit: hh:mm --
#BSUB -W ${walltime_dynesty}
### -- set the email address --
#BSUB -u ${email}
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o ${out_dir}/Output_${job_name}.out

# Change to configuration directory
cd ${config_dir}

# Clean up previous runs
rm -f configuration_file_dynesty_run_${job_name}.log

# Activate PyORBIT environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis with dynesty
pyorbit_run dynesty ${yaml_name} > configuration_file_dynesty_run_${job_name}.log
pyorbit_results dynesty ${yaml_name} -all >> configuration_file_dynesty_run_${job_name}.log

# Deactivate environment
conda deactivate

echo "Job ${job_name} completed at: \$(date)"
EOF

    chmod +x "${script_file}"
}

# ------------------ MAIN LOOP ------------------ #
for data_config in "${data_configs[@]}"; do
    for planet_conf in "${planets[@]}"; do
        # Include no_gp in the name so GP variants can be added later without clashing
        job_name="HD88986_${data_config}_${gp_flag}_${planet_conf}_dynesty"
        config_dir="${results_dir}/${data_config}/${gp_flag}/${planet_conf}/${job_name}"
        mkdir -p "${config_dir}"

        yaml_name="${job_name}.yaml"
        yaml_path="${config_dir}/${yaml_name}"

        generate_yaml "${yaml_path}" "${data_config}" "${planet_conf}"
        ((yaml_count++))

        script_name="run_${job_name}.sh"
        script_path="${scripts_dir}/${script_name}"
        generate_job_script "${script_path}" "${job_name}" "${config_dir}" "${yaml_name}"
        ((script_count++))

        echo "Created: ${yaml_path}"
        echo "Created: ${script_path}"
    done
done

# ------------------ MANAGEMENT SCRIPTS ------------------ #
cat > "${scripts_dir}/submit_all_dynesty.sh" << 'EOF'
#!/bin/bash
# Always run from this script's directory so globs find run_*.sh
# even if invoked as ../scripts_HD88986_dynesty/submit_all_dynesty.sh
cd "$(dirname "$0")" || exit 1

echo "Submitting ALL HD88986 dynesty jobs (no GP)..."
echo "=============================================="
echo "Working directory: $(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_*_dynesty.sh; do
  echo "Submitting: $script"
  bsub < "$script"
  job_count=$((job_count + 1))
  sleep 0.5
done

if [ "$job_count" -eq 0 ]; then
  echo "ERROR: no run_HD88986_*_dynesty.sh scripts found in $(pwd)" >&2
  exit 1
fi
echo "Submitted $job_count jobs."
EOF
chmod +x "${scripts_dir}/submit_all_dynesty.sh"

cat > "${scripts_dir}/submit_all_instr_dynesty.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")" || exit 1

echo "Submitting HD88986 all_instr dynesty jobs..."
echo "============================================"
echo "Working directory: $(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_all_instr_*_dynesty.sh; do
  echo "Submitting: $script"
  bsub < "$script"
  job_count=$((job_count + 1))
  sleep 0.5
done

if [ "$job_count" -eq 0 ]; then
  echo "ERROR: no matching run scripts found in $(pwd)" >&2
  exit 1
fi
echo "Submitted $job_count all_instr jobs."
EOF
chmod +x "${scripts_dir}/submit_all_instr_dynesty.sh"

for planet_conf in "${planets[@]}"; do
    cat > "${scripts_dir}/submit_${planet_conf}_dynesty.sh" << EOF
#!/bin/bash
cd "\$(dirname "\$0")" || exit 1

echo "Submitting HD88986 ${planet_conf} dynesty jobs..."
echo "================================================="
echo "Working directory: \$(pwd)"

job_count=0
shopt -s nullglob
for script in run_HD88986_*_${planet_conf}_dynesty.sh; do
  echo "Submitting: \$script"
  bsub < "\$script"
  job_count=\$((job_count + 1))
  sleep 0.5
done

if [ "\$job_count" -eq 0 ]; then
  echo "ERROR: no matching run scripts found in \$(pwd)" >&2
  exit 1
fi
echo "Submitted \$job_count ${planet_conf} jobs."
EOF
    chmod +x "${scripts_dir}/submit_${planet_conf}_dynesty.sh"
done

cat > "${scripts_dir}/monitor_dynesty.sh" << 'EOF'
#!/bin/bash

# Use -w so LSF does not truncate Job Name (default bjobs
# shortens HD88986_all_instr_no_gp_1p_dynesty -> *_1p_dynesty).

echo "HD88986 dynesty Job Monitor"
echo "==========================="
jobs=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_dynesty" || true)
if [ -z "$jobs" ]; then
  echo "No dynesty jobs found."
else
  echo "$jobs"
fi
echo ""
echo "Job counts by planet configuration:"
for planets in 1p 2p 3p; do
  count=$(bjobs -w 2>/dev/null | grep -cE "HD88986_.*_${planets}_dynesty" || true)
  echo "  ${planets}: ${count}"
done
EOF
chmod +x "${scripts_dir}/monitor_dynesty.sh"

cat > "${scripts_dir}/cancel_dynesty.sh" << 'EOF'
#!/bin/bash

echo "Canceling all HD88986 dynesty jobs..."
echo "====================================="

job_ids=$(bjobs -w 2>/dev/null | grep -E "HD88986_.*_dynesty" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No dynesty jobs to cancel."
  exit 0
fi

echo "Jobs to cancel:"
echo "$job_ids"
echo ""
read -p "Proceed? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for id in $job_ids; do
    echo "Canceling job $id"
    bkill "$id"
  done
  echo "Done."
else
  echo "Aborted."
fi
EOF
chmod +x "${scripts_dir}/cancel_dynesty.sh"

cat > "${scripts_dir}/status_dynesty.sh" << EOF
#!/bin/bash

echo "HD88986 dynesty Job Status Summary"
echo "=================================="
echo ""

results_dir="${results_dir}"

for config in all_instr; do
  for gp in no_gp; do
    echo "Configuration: \${config} / \${gp}"
    for planets in 1p 2p 3p; do
      job_name="HD88986_\${config}_\${gp}_\${planets}_dynesty"
      job_dir="\${results_dir}/\${config}/\${gp}/\${planets}/\${job_name}"
      if [ -d "\${job_dir}" ]; then
        if [ -f "\${job_dir}/dynesty_results.pkl" ] || ls "\${job_dir}"/*dynesty*results* >/dev/null 2>&1; then
          echo "  \${planets}: COMPLETED (or results present)"
        elif [ -f "\${job_dir}/configuration_file_dynesty_run_\${job_name}.log" ]; then
          echo "  \${planets}: RUNNING / LOG PRESENT"
        else
          echo "  \${planets}: NOT STARTED"
        fi
      else
        echo "  \${planets}: NOT CREATED"
      fi
    done
    echo ""
  done
done
EOF
chmod +x "${scripts_dir}/status_dynesty.sh"

echo ""
echo "Setup complete for dynesty sampling (no GP)."
echo "YAML files created:   ${yaml_count}"
echo "Job scripts created:  ${script_count}"
echo ""
echo "Directory structure:"
echo "  ${results_dir}/"
echo "    └── all_instr/"
echo "          └── no_gp/"
echo "                ├── 1p/   (planet b ~146 d)"
echo "                ├── 2p/   (b + outer companion c ~116 yr)"
echo "                └── 3p/   (b + c + free d)"
echo ""
echo "Available commands (cwd-independent; safe from proposal root or scripts dir):"
echo "  ${scripts_dir}/submit_all_dynesty.sh"
echo "  ${scripts_dir}/submit_1p_dynesty.sh   # (also 2p / 3p)"
echo "  ${scripts_dir}/monitor_dynesty.sh"
echo "  ${scripts_dir}/status_dynesty.sh"
echo "  ${scripts_dir}/cancel_dynesty.sh"
echo ""
echo "Or: cd ${scripts_dir} && bsub < run_HD88986_all_instr_no_gp_1p_dynesty.sh"
