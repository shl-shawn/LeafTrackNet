#!/usr/bin/env bash

python3 TrackEval/scripts/run_mot_challenge.py \
    --SPLIT_TO_EVAL val  \
    --METRICS HOTA CLEAR Identity  \
    --GT_FOLDER data/val    \
    --SEQMAP_FILE seqmap \
    --SKIP_SPLIT_FOL True \
    --TRACKERS_FOLDER /mnt/beegfs/home/liu21/LeafTrackNet/outputs/leaf_reid  \
    --TRACKERS_TO_EVAL tracks \
    --USE_PARALLEL True \
    --NUM_PARALLEL_CORES 8 \
    --PLOT_CURVES False \