singularity exec ~/MyResearch/sgm_latest.sif python ../python_scripts/newbds-SBI-FC-Reparam-Allband.py --noise_sd 0.8

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
