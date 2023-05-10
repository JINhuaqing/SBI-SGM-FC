#!/bin/bash
#### sbatch xxx.sh to submit the job

#### Job memory request
#SBATCH --mem=200gb                  
#SBATCH --nodes=1
#SBATCH --ntasks=25
#### Run on partition "dgx" (e.g. not the default partition called "long")
### long for CPU, gpu/dgx for CPU, dgx is slow
#SBATCH --partition=long
#### Allocate 1 GPU resource for this job. 
#####SBATCH --gres=gpu:teslav100:1   
#SBATCH --output=logs/job-%x.out
#SBATCH -J new_sgm_bandwise_delta_w0


#### You job
echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

bash $1


