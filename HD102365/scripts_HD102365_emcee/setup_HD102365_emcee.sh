#!/bin/bash

# HD102365 emcee PyORBIT Setup Generator
# Creates directory structure, YAML configs, and LSF job scripts for:
#   - all_instr (all instruments with GP on all)
#   - all_instr_espresso_gp (GP only on ESPRESSO, others Keplerian-only)
#   - espresso_only (ESPRESSO data only with GP)
#   - no_espresso (all classical instruments with GP)
#   - ucles_only (UCLES data only with GP)
#   - no_ucles (all data except UCLES, GP on all instruments)
#   - 0p, 1p, 2p, 3p planet configurations
#   - emcee sampler

base_dir="/work2/lbuc/jzhao/PyORBIT_ESSP/HD102365"
data_dir="/work2/lbuc/jzhao/PyORBIT_ESSP/HD102365/data/processed_data"
results_dir="${base_dir}/results_HD102365_emcee_no_derivative"
out_dir="${base_dir}/out_HD102365_emcee"
scripts_dir="${base_dir}/scripts_HD102365_emcee"

mkdir -p "${results_dir}" "${out_dir}" "${scripts_dir}"
cd "${scripts_dir}" || exit 1

# LSF configuration
queue="hpc"
email="jzhao@space.dtu.dk"

# Resources for emcee (GP jobs)
cores_emcee=16
threads_emcee=$((${cores_emcee}-1))
mem_per_core_emcee="8GB"
mem_limit_emcee="9GB"
walltime_emcee="72:00"

# Configuration axes
data_configs=("all_instr" "all_instr_espresso_gp" "espresso_only" "no_espresso" "ucles_only" "no_ucles")
planets=("0p" "1p" "2p" "3p")

# Classical RV inputs
rv_inputs_classical=(
    "HD102365_HARPS-Post_RV"
    "HD102365_HARPS-Pre_RV"
    "HD102365_HIRES-Post_RV"
    "HD102365_PFS-Post_RV"
    "HD102365_PFS-Pre_RV"
    "HD102365_UCLES_RV"
)

yaml_count=0
script_count=0

echo "HD102365 emcee PyORBIT Setup"
echo "============================"
echo "Base dir:      ${base_dir}"
echo "Results dir:   ${results_dir}"
echo "Output dir:    ${out_dir}"
echo "Scripts dir:   ${scripts_dir}"
echo "Cores:         ${cores_emcee}"
echo ""

# ------------------ YAML GENERATOR ------------------ #
generate_yaml() {
    local yaml_file="$1"
    local data_config="$2"   # all_instr / all_instr_espresso_gp / espresso_only / no_espresso / ucles_only / no_ucles
    local planet_conf="$3"   # 0p / 1p / 2p / 3p

    # Determine number of planets
    local num_planets
    case "${planet_conf}" in
        "0p") num_planets=0 ;;
        "1p") num_planets=1 ;;
        "2p") num_planets=2 ;;
        "3p") num_planets=3 ;;
        *) echo "Unknown planet_conf ${planet_conf}" ; exit 1 ;;
    esac

    # Determine which instruments to include and GP strategy
    local include_espresso=false
    local include_classical=false
    local gp_on_espresso=false
    local gp_on_classical=false
    local is_ucles_only=false
    local exclude_ucles=false

    case "${data_config}" in
        "all_instr")
            include_espresso=true
            include_classical=true
            gp_on_espresso=true
            gp_on_classical=true
            ;;
        "all_instr_espresso_gp")
            include_espresso=true
            include_classical=true
            gp_on_espresso=true
            gp_on_classical=false
            ;;
        "espresso_only")
            include_espresso=true
            include_classical=false
            gp_on_espresso=true
            gp_on_classical=false
            ;;
        "no_espresso")
            include_espresso=false
            include_classical=true
            gp_on_espresso=false
            gp_on_classical=true
            ;;
        "ucles_only")
            include_espresso=false
            include_classical=false
            is_ucles_only=true
            gp_on_classical=true
            ;;
        "no_ucles")
            include_espresso=true
            include_classical=true
            gp_on_espresso=true
            gp_on_classical=true
            exclude_ucles=true
            ;;
        *)
            echo "Unknown data_config ${data_config}"
            exit 1
            ;;
    esac

    # Start YAML
    cat > "${yaml_file}" << EOF
inputs:
EOF

    # ESPRESSO RV
    if [ "${include_espresso}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_RV:
    file: ${data_dir}/HD102365_ESPRESSO_RV.dat
    kind: RV
    models:
      - radial_velocities
EOF
        if [ "${gp_on_espresso}" == true ]; then
            echo "      - gp_multidimensional" >> "${yaml_file}"
        fi
    fi

    # Classical RV inputs
    if [ "${include_classical}" == true ]; then
        for rv_name in "${rv_inputs_classical[@]}"; do
            if [ "${exclude_ucles}" == true ] && [ "${rv_name}" == "HD102365_UCLES_RV" ]; then
                continue
            fi
            cat >> "${yaml_file}" << EOF
  ${rv_name}:
    file: ${data_dir}/${rv_name}.dat
    kind: RV
    models:
      - radial_velocities
EOF
            if [ "${gp_on_classical}" == true ]; then
                echo "      - gp_multidimensional" >> "${yaml_file}"
            fi
        done
    fi

    # UCLES only
    if [ "${is_ucles_only}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_UCLES_RV:
    file: ${data_dir}/HD102365_UCLES_RV.dat
    kind: RV
    models:
      - radial_velocities
      - gp_multidimensional
EOF
    fi

    # Activity indicators (only if GP is used)
    if [ "${gp_on_espresso}" == true ] && [ "${include_espresso}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_ESPRESSO_BIS:
    file: ${data_dir}/HD102365_ESPRESSO_BIS.dat
    kind: BIS
    models:
      - gp_multidimensional
EOF
    fi

    if [ "${gp_on_classical}" == true ] && [ "${include_classical}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_HARPS-Post_SHK:
    file: ${data_dir}/HD102365_HARPS-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_HARPS-Pre_SHK:
    file: ${data_dir}/HD102365_HARPS-Pre_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_HIRES-Post_SHK:
    file: ${data_dir}/HD102365_HIRES-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_PFS-Post_SHK:
    file: ${data_dir}/HD102365_PFS-Post_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
  HD102365_PFS-Pre_SHK:
    file: ${data_dir}/HD102365_PFS-Pre_SHK.dat
    kind: SHK
    models:
      - gp_multidimensional
EOF
        # UCLES activity indicator only when UCLES data is included
        if [ "${exclude_ucles}" != true ]; then
            cat >> "${yaml_file}" << EOF
  HD102365_UCLES_EWHa:
    file: ${data_dir}/HD102365_UCLES_EWHa.dat
    kind: EWHa
    models:
      - gp_multidimensional
EOF
        fi
    fi

    if [ "${is_ucles_only}" == true ]; then
        cat >> "${yaml_file}" << EOF
  HD102365_UCLES_EWHa:
    file: ${data_dir}/HD102365_UCLES_EWHa.dat
    kind: EWHa
    models:
      - gp_multidimensional
EOF
    fi

    # ---------- COMMON SECTION ---------- #
    cat >> "${yaml_file}" << EOF

common:
EOF

    # Planets section (only if num_planets > 0)
    if [ ${num_planets} -gt 0 ]; then
        cat >> "${yaml_file}" << EOF
  planets:
EOF
        planet_letters=("b" "c" "d")
        for ((p=0; p<num_planets; p++)); do
            pl=${planet_letters[$p]}
            cat >> "${yaml_file}" << EOF
    ${pl}:
      orbit: keplerian
      parametrization: Eastman2013
      boundaries:
        P: [1.1, 2000.0]
        K: [0.1, 5.0]
        e: [0.00, 0.95]
EOF
        done
    fi

    # Activity hyperparameters (GP is always used in this setup)
    if [ "${is_ucles_only}" == true ]; then
        # UCLES-only uses different Prot bounds
        cat >> "${yaml_file}" << EOF
  activity:
    boundaries:
      Prot: [3000.0, 4500.0]
      Pdec: [500.0, 10000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 3500.00, 500.0]
EOF
    else
        cat >> "${yaml_file}" << EOF
  activity:
    boundaries:
      Prot: [20.0, 60.0]
      Pdec: [10.0, 1000.0]
      Oamp: [0.01, 1.0]
    priors:
      Prot: ['Gaussian', 36.00, 10.0]
EOF
    fi

    # Star parameters
    cat >> "${yaml_file}" << EOF
  star:
    star_parameters:
      priors:
        mass: ['Gaussian', 0.85, 0.03]
        radius: ['Gaussian', 0.99, 0.02]
        density: ['Gaussian', 0.8760186293091098, 0.04]

models:
  radial_velocities:
    planets:
EOF

    # Add planet list (empty for 0p)
    if [ ${num_planets} -gt 0 ]; then
        for ((p=0; p<num_planets; p++)); do
            pl=${planet_letters[$p]}
            echo "      - ${pl}" >> "${yaml_file}"
        done
    else
        echo "      []" >> "${yaml_file}"
    fi

    # ---------- GP MODEL BLOCK ---------- #
    cat >> "${yaml_file}" << EOF
  gp_multidimensional:
    model: spleaf_multidimensional_esp
    common: activity
    n_harmonics: 4
    hyperparameters_condition: True
    rotation_decay_condition: True
EOF

    # ESPRESSO RV GP parameters
    if [ "${gp_on_espresso}" == true ] && [ "${include_espresso}" == true ]; then
        cat >> "${yaml_file}" << EOF
    HD102365_ESPRESSO_RV:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [-20.0, 20.0]
      derivative: False
EOF
    fi

    # Classical RV GP parameters
    if [ "${gp_on_classical}" == true ] && [ "${include_classical}" == true ]; then
        for rv_name in "${rv_inputs_classical[@]}"; do
            local con_amp
            if [[ "${rv_name}" == "HD102365_UCLES_RV" ]]; then
                con_amp="-30.0, 20.0"
            else
                con_amp="-20.0, 20.0"
            fi
            cat >> "${yaml_file}" << EOF
    ${rv_name}:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [${con_amp}]
      derivative: False
EOF
        done
    fi

    # UCLES only
    if [ "${is_ucles_only}" == true ]; then
        cat >> "${yaml_file}" << EOF
    HD102365_UCLES_RV:
      boundaries:
        rot_amp: [0.0, 10.0]
        con_amp: [-30.0, 20.0]
      derivative: False
EOF
    fi

    # ESPRESSO BIS
    if [ "${gp_on_espresso}" == true ] && [ "${include_espresso}" == true ]; then
        cat >> "${yaml_file}" << EOF
    HD102365_ESPRESSO_BIS:
      boundaries:
        rot_amp: [-80.0, 80.0]
        con_amp: [-10.0, 10.0]
      derivative: False
EOF
    fi

    # Classical activity indicators
    if [ "${gp_on_classical}" == true ] && [ "${include_classical}" == true ]; then
        cat >> "${yaml_file}" << EOF
    HD102365_HARPS-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
    HD102365_HARPS-Pre_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
    HD102365_HIRES-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
    HD102365_PFS-Post_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
    HD102365_PFS-Pre_SHK:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
    HD102365_UCLES_EWHa:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
EOF
    fi

    # UCLES only
    if [ "${is_ucles_only}" == true ]; then
        cat >> "${yaml_file}" << EOF
    HD102365_UCLES_EWHa:
      boundaries:
        rot_amp: [-1.0, 1.0]
        con_amp: [-10.0, 10.0]
      derivative: False
EOF
    fi

    # Tref: espresso_only and no_ucles use different reference times
    local tref="2450830.213590"
    if [ "${data_config}" == "espresso_only" ]; then
        tref="2458488.835880"
    elif [ "${data_config}" == "no_ucles" ]; then
        tref="2452990.877720"
    fi

    # Parameters & solver
    cat >> "${yaml_file}" << EOF

parameters:
  Tref: ${tref}
  low_ram_plot: True
  plot_split_threshold: 1000
  cpu_threads: ${threads_emcee}

solver:
  pyde:
    ngen: 50000
    npop_mult: 6
  emcee:
    npop_mult: 6
    nsteps: 50000
    nburn: 15000
    nsave: 35000
    thin: 15
  nested_sampling:
    nlive: 1000
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
#BSUB -n ${cores_emcee}
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need ${mem_per_core_emcee} of memory per core/slot -- 
#BSUB -R "rusage[mem=${mem_per_core_emcee}]"
### -- specify that we want the job to get killed if it exceeds ${mem_limit_emcee} per core/slot -- 
#BSUB -M ${mem_limit_emcee}
### -- set walltime limit: hh:mm -- 
#BSUB -W ${walltime_emcee}
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
rm -f configuration_file_emcee_run_${job_name}.log

# Activate PyORBIT environment
# source ~/anaconda3/etc/profile.d/conda.sh
source /work2/lbuc/iara/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis with emcee
pyorbit_run emcee ${yaml_name} > configuration_file_emcee_run_${job_name}.log
pyorbit_results emcee ${yaml_name} -all >> configuration_file_emcee_run_${job_name}.log

# Deactivate environment
conda deactivate

echo "Job ${job_name} completed at: \$(date)"
EOF

    chmod +x "${script_file}"
}

# ------------------ MAIN LOOP ------------------ #
for data_config in "${data_configs[@]}"; do
    for planet_conf in "${planets[@]}"; do
        job_name="HD102365_${data_config}_${planet_conf}_emcee"
        config_dir="${results_dir}/${data_config}/${planet_conf}/${job_name}"
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
cat > "${scripts_dir}/submit_all_emcee.sh" << 'EOF'
#!/bin/bash

echo "Submitting ALL HD102365 emcee jobs..."
echo "======================================"

job_count=0
for script in run_HD102365_*_emcee.sh; do
  if [ -f "$script" ]; then
    echo "Submitting: $script"
    bsub < "$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted $job_count jobs."
EOF
chmod +x "${scripts_dir}/submit_all_emcee.sh"

# Submit scripts for each data configuration
for data_config in "${data_configs[@]}"; do
    cat > "${scripts_dir}/submit_${data_config}_emcee.sh" << EOF
#!/bin/bash

echo "Submitting HD102365 ${data_config} emcee jobs..."
echo "=================================================="

job_count=0
for script in run_HD102365_${data_config}_*_emcee.sh; do
  if [ -f "\$script" ]; then
    echo "Submitting: \$script"
    bsub < "\$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted \$job_count ${data_config} jobs."
EOF
    chmod +x "${scripts_dir}/submit_${data_config}_emcee.sh"
done

# Submit scripts for each planet configuration
for planet_conf in "${planets[@]}"; do
    cat > "${scripts_dir}/submit_${planet_conf}_emcee.sh" << EOF
#!/bin/bash

echo "Submitting HD102365 ${planet_conf} emcee jobs..."
echo "=================================================="

job_count=0
for script in run_HD102365_*_${planet_conf}_emcee.sh; do
  if [ -f "\$script" ]; then
    echo "Submitting: \$script"
    bsub < "\$script"
    ((job_count++))
    sleep 0.5
  fi
done

echo "Submitted \$job_count ${planet_conf} jobs."
EOF
    chmod +x "${scripts_dir}/submit_${planet_conf}_emcee.sh"
done

cat > "${scripts_dir}/monitor_emcee.sh" << 'EOF'
#!/bin/bash

echo "HD102365 emcee Job Monitor"
echo "=========================="
bjobs | grep "HD102365_.*_emcee" || echo "No emcee jobs found."
echo ""
echo "Job counts by configuration:"
for config in all_instr all_instr_espresso_gp espresso_only no_espresso ucles_only no_ucles; do
  count=$(bjobs | grep "HD102365_${config}_.*_emcee" | wc -l)
  echo "  ${config}: ${count}"
done
echo ""
echo "Job counts by planet configuration:"
for planets in 0p 1p 2p 3p; do
  count=$(bjobs | grep "HD102365_.*_${planets}_emcee" | wc -l)
  echo "  ${planets}: ${count}"
done
EOF
chmod +x "${scripts_dir}/monitor_emcee.sh"

cat > "${scripts_dir}/cancel_emcee.sh" << 'EOF'
#!/bin/bash

echo "Canceling all HD102365 emcee jobs..."
echo "===================================="

job_ids=$(bjobs | grep "HD102365_.*_emcee" | awk '{print $1}')
if [ -z "$job_ids" ]; then
  echo "No emcee jobs to cancel."
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
chmod +x "${scripts_dir}/cancel_emcee.sh"

cat > "${scripts_dir}/status_emcee.sh" << 'EOF'
#!/bin/bash

echo "HD102365 emcee Job Status Summary"
echo "=================================="
echo ""

results_dir="/work2/lbuc/iara/GitHub/PyORBIT_examples/HD102365/results_HD102365_emcee"

for config in all_instr all_instr_espresso_gp espresso_only no_espresso ucles_only no_ucles; do
  echo "Configuration: ${config}"
  for planets in 0p 1p 2p 3p; do
    job_dir="${results_dir}/${config}/${planets}/HD102365_${config}_${planets}_emcee"
    if [ -d "${job_dir}" ]; then
      if [ -f "${job_dir}/emcee_results.pkl" ]; then
        echo "  ${planets}: COMPLETED"
      elif [ -f "${job_dir}/configuration_file_emcee_run_HD102365_${config}_${planets}_emcee.log" ]; then
        echo "  ${planets}: RUNNING"
      else
        echo "  ${planets}: NOT STARTED"
      fi
    else
      echo "  ${planets}: NOT CREATED"
    fi
  done
  echo ""
done
EOF
chmod +x "${scripts_dir}/status_emcee.sh"

echo ""
echo "Setup complete for emcee sampling."
echo "YAML files created:   ${yaml_count}"
echo "Job scripts created:  ${script_count}"
echo ""
echo "Directory structure:"
echo "  ${results_dir}/"
echo "    ├── all_instr/           (all instruments, GP on all)"
echo "    ├── all_instr_espresso_gp/ (GP only on ESPRESSO)"
echo "    ├── espresso_only/       (ESPRESSO data only)"
echo "    ├── no_espresso/         (classical instruments only)"
echo "    ├── ucles_only/          (UCLES data only)"
echo "    └── no_ucles/            (all data except UCLES)"
echo ""
echo "Available commands:"
echo "  cd ${scripts_dir}"
echo ""
echo "Submit all jobs:"
echo "  ./submit_all_emcee.sh"
echo ""
echo "Submit by data configuration:"
echo "  ./submit_all_instr_emcee.sh"
echo "  ./submit_all_instr_espresso_gp_emcee.sh"
echo "  ./submit_espresso_only_emcee.sh"
echo "  ./submit_no_espresso_emcee.sh"
echo "  ./submit_ucles_only_emcee.sh"
echo "  ./submit_no_ucles_emcee.sh"
echo ""
echo "Submit by planet configuration:"
echo "  ./submit_0p_emcee.sh"
echo "  ./submit_1p_emcee.sh"
echo "  ./submit_2p_emcee.sh"
echo "  ./submit_3p_emcee.sh"
echo ""
echo "Monitor and manage:"
echo "  ./monitor_emcee.sh    (check running jobs)"
echo "  ./status_emcee.sh     (check completion status)"
echo "  ./cancel_emcee.sh     (cancel all jobs)"
