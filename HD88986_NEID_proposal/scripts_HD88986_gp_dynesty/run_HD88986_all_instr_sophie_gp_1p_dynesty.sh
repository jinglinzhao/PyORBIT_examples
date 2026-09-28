#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J HD88986_all_instr_sophie_gp_1p_dynesty
### -- ask for number of cores (default: 1) --
#BSUB -n 32
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5GB per core/slot --
#BSUB -M 5GB
### -- set walltime limit: hh:mm --
#BSUB -W 72:00
### -- set the email address --
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal/out_HD88986_gp_dynesty/Output_HD88986_all_instr_sophie_gp_1p_dynesty.out

# Change to configuration directory
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal/results_HD88986_gp_dynesty/all_instr/sophie_gp/1p/HD88986_all_instr_sophie_gp_1p_dynesty

# Clean up previous runs
rm -f configuration_file_dynesty_run_HD88986_all_instr_sophie_gp_1p_dynesty.log

# Activate PyORBIT environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis with dynesty
pyorbit_run dynesty HD88986_all_instr_sophie_gp_1p_dynesty.yaml > configuration_file_dynesty_run_HD88986_all_instr_sophie_gp_1p_dynesty.log
pyorbit_results dynesty HD88986_all_instr_sophie_gp_1p_dynesty.yaml -all >> configuration_file_dynesty_run_HD88986_all_instr_sophie_gp_1p_dynesty.log

# Deactivate environment
conda deactivate

echo "Job HD88986_all_instr_sophie_gp_1p_dynesty completed at: $(date)"
