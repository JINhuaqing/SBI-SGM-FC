#!/bin/bash
qsub -cwd -pe smp 30 -l mem_free=2G -l h_rt=18:00:00 $1
