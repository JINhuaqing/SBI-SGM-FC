#!/bin/bash
#### sbatch xxx.sh to submit the job

#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=long
#### Allocate 1 GPU resource for this job. 
#####SBATCH --gres=gpu:teslav100:1   
#SBATCH --output=logs/job_%x_%j.out
#SBATCH -J delta-12
#SBATCH --chdir=/home/hujin/jin/MyResearch/SBI-SGM-FC/bash_scripts/


#### You job
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

singularity exec ~/jin/singularity_containers/sgm_mne_lib_fc.sif python ../python_scripts/revision1/RUN_BW.py --noise_sd 1.2 --band delta

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"


