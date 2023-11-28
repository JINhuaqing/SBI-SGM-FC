#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/SBI-SGM-FC/bash_scripts/
#### Specify job name
#$ -N ANNnv2_dlt_100 #!!!
#### Output file
#$ -o wynton/logs/$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=4G
#### number of cores 
#$ -pe smp 40
#### Maximum run time 
#$ -l h_rt=48:00:00
#### job requires up to 2 GB local space
#$ -l scratch=16G
#### Specify queue
###  gpu.q for using gpu
###  if not gpu.q, do not need to specify it
###$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M

echo "Start running"


singularity exec ~/MyResearch/sgm_mne_lib_fc.sif python -u ../python_scripts/M1_RUN_ANN_newv2.py --band delta --nepoch 100 #!!!  
#### End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
