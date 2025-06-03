#!/bin/bash

CONFIG_SWIN_DIR="projects/configs/co_deformable_detr_swin/masking_test"
CONFIG_MIM_DIR="projects/configs/co_deformable_detr_mim/masking_test"
CONFIG_MIM_LINEAR_DIR="projects/configs/co_deformable_detr_mim/masking_test_linear"

CHECKPOINT_BASELINE="work_dirs/swin_baseline/epoch_6.pth"
CHECKPOINT_BASELINE_MASKED="work_dirs/swin_baseline_masked/epoch_6.pth"
CHECKPOINT_MIM_05="work_dirs/mim_with_swin_improved_mim_weight05/epoch_6.pth"
CHECKPOINT_MIM_1="work_dirs/mim_with_swin_improved_mim_masking/epoch_6.pth"
CHECKPOINT_MIM_LINEAR="work_dirs/mim_linear_05/epoch_6.pth"

for i in $(seq 0 2 8); do
    CONFIG="masking_0${i}.py"
    # CONFIG_PATH="${CONFIG_SWIN_DIR}/${CONFIG}"

    # CONFIG_PATH_MIM="${CONFIG_MIM_DIR}/${CONFIG}"
    CONFIG_PATH_MIM_LINEAR="${CONFIG_MIM_LINEAR_DIR}/${CONFIG}"


    # WORKDIR_BASELINE="work_dirs/swin_baseline/masking_tests/${CONFIG%.py}"
    # WORKDIR_BASELINE_MASKED="work_dirs/swin_baseline_masked/masking_tests/${CONFIG%.py}"
    # WORKDIR_MASKED_05="work_dirs/mim_with_swin_improved_mim_weight05/masking_tests/${CONFIG%.py}"
    # WORKDIR_MASKED_1="work_dirs/mim_with_swin_improved_mim_masking/masking_tests/${CONFIG%.py}"
    WORKDIR_MASKED_LINEAR="work_dirs/mim_linear_05/masking_tests/${CONFIG%.py}"

    echo "Running test on ${CONFIG}..."

    # echo "Baseline"
    # PYTHONPATH=$(pwd) python tools/test.py \
    #     "$CONFIG_PATH" \
    #     "$CHECKPOINT_BASELINE" \
    #     --work-dir "$WORKDIR_BASELINE" \
    #     --eval bbox \
    #     --show-dir "$WORKDIR_BASELINE/result"

    # echo "Baseline masked"    
    # PYTHONPATH=$(pwd) python tools/test.py \
    # "$CONFIG_PATH" \
    # "$CHECKPOINT_BASELINE_MASKED" \
    # --work-dir "$WORKDIR_BASELINE_MASKED" \
    # --eval bbox \
    # --show-dir "$WORKDIR_BASELINE_MASKED/result"


    # echo "Masked 0.5"    
    # PYTHONPATH=$(pwd) python tools/test.py \
    # "$CONFIG_PATH_MIM" \
    # "$CHECKPOINT_MIM_05" \
    # --work-dir "$WORKDIR_MASKED_05" \
    # --eval bbox \
    # --show-dir "$WORKDIR_MASKED_05/result"

    
    # echo "Masked 1"    
    # PYTHONPATH=$(pwd) python tools/test.py \
    # "$CONFIG_PATH_MIM" \
    # "$CHECKPOINT_MIM_1" \
    # --work-dir "$WORKDIR_MASKED_1" \
    # --eval bbox \
    # --show-dir "$WORKDIR_MASKED_1/result"

    echo "Baseline"
    PYTHONPATH=$(pwd) python tools/test.py \
        "$CONFIG_PATH_MIM_LINEAR" \
        "$CHECKPOINT_MIM_LINEAR" \
        --work-dir "$WORKDIR_MASKED_LINEAR" \
        --eval bbox \
        --show-dir "$WORKDIR_MASKED_LINEAR/result"
done

# PYTHONPATH={pwd} python tools/test.py projects/configs/co_deformable_detr_swin/co_deformable_detr_swin.py work_dirs/swin_baseline_masked/epoch_6.pth --work-dir work_dirs/swin_baseline_masked/ --eval bbox --show-dir work_dirs/swin_baseline_masked/result

# PYTHONPATH=$(pwd) python tools/test.py \
# projects/configs/co_deformable_detr_mim/tests/brightness.py \
# work_dirs/mim_with_swin_improved_mim_weight05/epoch_6.pth \
# --work-dir work_dirs/mim_with_swin_improved_mim_weight05/brightness \
# --eval bbox \
# --show-dir work_dirs/mim_with_swin_improved_mim_weight05/brightness/result


# # OrderedDict([('bbox_mAP', 0.093), ('bbox_mAP_50', 0.146), ('bbox_mAP_75', 0.099), ('bbox_mAP_s', 0.038), ('bbox_mAP_m', 0.1), ('bbox_mAP_l', 0.154), ('bbox_mAP_copypaste', '0.093 0.146 0.099 0.038 0.100 0.154')])
# # OrderedDict([('bbox_mAP', 0.093), ('bbox_mAP_50', 0.147), ('bbox_mAP_75', 0.098), ('bbox_mAP_s', 0.033), ('bbox_mAP_m', 0.097), ('bbox_mAP_l', 0.152), ('bbox_mAP_copypaste', '0.093 0.147 0.098 0.033 0.097 0.152')])

# OrderedDict([('bbox_mAP', 0.084), ('bbox_mAP_50', 0.156), ('bbox_mAP_75', 0.077), ('bbox_mAP_s', 0.029), ('bbox_mAP_m', 0.075), ('bbox_mAP_l', 0.151), ('bbox_mAP_copypaste', '0.084 0.156 0.077 0.029 0.075 0.151')])
# OrderedDict([('bbox_mAP', 0.08), ('bbox_mAP_50', 0.15), ('bbox_mAP_75', 0.074), ('bbox_mAP_s', 0.029), ('bbox_mAP_m', 0.071), ('bbox_mAP_l', 0.143), ('bbox_mAP_copypaste', '0.080 0.150 0.074 0.029 0.071 0.143')])

# PYTHONPATH=$(pwd) python tools/test.py projects/configs/co_deformable_detr_mim/tests/brightness.py work_dirs/mim_with_swin_improved_mim_weight05/epoch_6.pth --work-dir work_dirs/mim_with_swin_improved_mim_weight05/albu --eval bbox --show-dir work_dirs/mim_with_swin_improved_mim_weight05/albu/result 