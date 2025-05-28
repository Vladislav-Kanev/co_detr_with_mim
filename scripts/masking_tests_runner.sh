#!/bin/bash

CONFIG_SWIN_DIR="projects/configs/co_deformable_detr_swin/masking_test"
CONFIG_MIM_DIR="projects/configs/co_deformable_detr_mim/masking_test"

CHECKPOINT_BASELINE="work_dirs/swin_baseline/epoch_6.pth"
CHECKPOINT_BASELINE_MASKED="work_dirs/swin_baseline_masked/epoch_6.pth"
CHECKPOINT_MIM_05="work_dirs/mim_with_swin_improved_mim_weight05/epoch_6.pth"
CHECKPOINT_MIM_1="work_dirs/mim_with_swin_improved_mim_masking/epoch_6.pth"

for i in $(seq 0 2 8); do
    CONFIG="masking_0${i}.py"
    CONFIG_PATH="${CONFIG_SWIN_DIR}/${CONFIG}"

    CONFIG_PATH_MIM="${CONFIG_MIM_DIR}/${CONFIG}"


    WORKDIR_BASELINE="work_dirs/swin_baseline/masking_tests/${CONFIG%.py}"
    WORKDIR_BASELINE_MASKED="work_dirs/swin_baseline_masked/masking_tests/${CONFIG%.py}"
    WORKDIR_MASKED_05="work_dirs/mim_with_swin_improved_mim_weight05/masking_tests/${CONFIG%.py}"
    WORKDIR_MASKED_1="work_dirs/mim_with_swin_improved_mim_masking/masking_tests/${CONFIG%.py}"

    echo "Running test on ${CONFIG}..."

    echo "Baseline"
    PYTHONPATH=$(pwd) python tools/test.py \
        "$CONFIG_PATH" \
        "$CHECKPOINT_BASELINE" \
        --work-dir "$WORKDIR_BASELINE" \
        --eval bbox \
        --show-dir "$WORKDIR_BASELINE/result"

    echo "Baseline masked"    
    PYTHONPATH=$(pwd) python tools/test.py \
    "$CONFIG_PATH" \
    "$CHECKPOINT_BASELINE_MASKED" \
    --work-dir "$WORKDIR_BASELINE_MASKED" \
    --eval bbox \
    --show-dir "$WORKDIR_BASELINE_MASKED/result"


    echo "Masked 0.5"    
    PYTHONPATH=$(pwd) python tools/test.py \
    "$CONFIG_PATH_MIM" \
    "$CHECKPOINT_MIM_05" \
    --work-dir "$WORKDIR_MASKED_05" \
    --eval bbox \
    --show-dir "$WORKDIR_MASKED_05/result"

    
    echo "Masked 1"    
    PYTHONPATH=$(pwd) python tools/test.py \
    "$CONFIG_PATH_MIM" \
    "$CHECKPOINT_MIM_1" \
    --work-dir "$WORKDIR_MASKED_1" \
    --eval bbox \
    --show-dir "$WORKDIR_MASKED_1/result"

done

# PYTHONPATH={pwd} python tools/test.py projects/configs/co_deformable_detr_swin/co_deformable_detr_swin.py work_dirs/swin_baseline_masked/epoch_6.pth --work-dir work_dirs/swin_baseline_masked/ --eval bbox --show-dir work_dirs/swin_baseline_masked/result

   echo "Masked 1"    
    PYTHONPATH=$(pwd) python tools/test.py \
    projects/configs/co_deformable_detr_mim/masking_test/masking_04.py \
    work_dirs/mim_with_swin_improved_mim_masking/epoch_6.pth \
    --work-dir work_dirs/mim_with_swin_improved_mim_masking/restoration_test \
    --eval bbox \
    --show-dir work_dirs/mim_with_swin_improved_mim_masking/restoration_test/result"