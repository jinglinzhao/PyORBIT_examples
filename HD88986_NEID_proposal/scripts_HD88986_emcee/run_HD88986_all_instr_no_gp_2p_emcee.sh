#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J HD88986_all_instr_no_gp_2p_emcee
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 2GB per core/slot --
#BSUB -M 2GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
#BSUB -u jzhao@space.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal/out_HD88986_emcee/Output_HD88986_all_instr_no_gp_2p_emcee.out

# Change to configuration directory
cd /work2/lbuc/jzhao/PyORBIT_ESSP/HD88986_NEID_proposal/results_HD88986_emcee/all_instr/no_gp/2p/HD88986_all_instr_no_gp_2p_emcee

# Clean up previous runs
rm -f configuration_file_emcee_run_HD88986_all_instr_no_gp_2p_emcee.log

# Activate PyORBIT environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyorbit

# Run PyORBIT analysis with emcee
pyorbit_run emcee HD88986_all_instr_no_gp_2p_emcee.yaml > configuration_file_emcee_run_HD88986_all_instr_no_gp_2p_emcee.log
pyorbit_results emcee HD88986_all_instr_no_gp_2p_emcee.yaml -all >> configuration_file_emcee_run_HD88986_all_instr_no_gp_2p_emcee.log

# Deactivate environment
conda deactivate

echo "Job HD88986_all_instr_no_gp_2p_emcee completed at: $(date)"
