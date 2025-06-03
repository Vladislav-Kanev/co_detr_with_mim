#!/bin/bash

CONFIG_NAMES=(
    # grid_03_cut_8
    # grid_03_cut_16
    # grid_03_cut_32
    # grid_05_cut_8
    # grid_05_cut_16
    # grid_05_cut_32
    # grid_07_cut_8
    # grid_07_cut_16
    # grid_07_cut_32
    # grid_06_cut_0
    # grid_06_num_8 
    # grid_06_num_16 
    grid_06_num_20 
    # grid_06_num_32
)

BASELINE_CKPT="work_dirs/swin_baseline/epoch_6.pth"
MIM_CKPT="work_dirs/mim_with_swin_improved_mim_weight05/epoch_6.pth"

for CONFIG in "${CONFIG_NAMES[@]}"; do
    echo "▶️ Запуск baseline на конфиге $CONFIG"
    PYTHONPATH=$(pwd) python tools/test.py \
        projects/configs/co_deformable_detr_swin/tests/grid/${CONFIG}.py \
        $BASELINE_CKPT \
        --work-dir work_dirs/swin_baseline/grid/${CONFIG} \
        --eval bbox \
        --show-dir work_dirs/swin_baseline/grid/${CONFIG}/result

    echo "▶️ Запуск MIM на конфиге $CONFIG"
    PYTHONPATH=$(pwd) python tools/test.py \
        projects/configs/co_deformable_detr_mim/tests/grid/${CONFIG}.py \
        $MIM_CKPT \
        --work-dir work_dirs/mim_with_swin_improved_mim_weight05/grid/${CONFIG} \
        --eval bbox \
        --show-dir work_dirs/mim_with_swin_improved_mim_weight05/grid/${CONFIG}/result

    echo "✅ Завершено: $CONFIG"
    echo "------------------------------"
done
