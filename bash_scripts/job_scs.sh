#singularity exec ~/jin/singularity_containers/sgm_latest.sif python ../python_scripts/RUN_allband.py --noise_sd 0.8
singularity exec ~/jin/singularity_containers/sgm_latest.sif python ../python_scripts/RUN_bandwise.py --noise_sd 0.8 --band delta

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
