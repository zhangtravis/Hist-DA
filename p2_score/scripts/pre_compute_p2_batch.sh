#!/bin/bash
set -e
for ((part=0; part < 31; part++)) do
    s_idx=$(( 200*part ))
    e_idx=$(( 200*part + 200 ))
    echo $s_idx
    echo $e_idx
    sbatch scripts/run.sh python pre_compute_p2_score.py dataset="lyft" data_paths="lyft_gt_seg.yaml" start_idx=$s_idx end_idx=$e_idx
done