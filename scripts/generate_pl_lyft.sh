#!/bin/bash

set -x
proj_root_dir=$(pwd)
source_domain="ithaca365"
target_domain="lyft"
target_model="pointrcnn_eval_hindsight_car_pedestrian"
source_model="pointrcnn_hindsight_p2_squashlevel"
pl_output="lyft_pl_hindsight_p2_squashlevel"

function generate_pl () {
    local result_path=${1}
    local target_path=${2}
    python ${proj_root_dir}/p2_score/p2_score_filtering_lidar_consistency.py result_path=${result_path} save_path=${target_path}
}

cd ${proj_root_dir}/downstream/OpenPCDet/tools
# generate the predictions on the training set
bash scripts/dist_test.sh 4 --cfg_file cfgs/${target_domain}_models/${target_model}.yaml \
    --extra_tag pl --eval_tag trainset \
    --ckpt ../output/${source_domain}_models/${source_model}/default/ckpt/last_checkpoint.pth \
    --set DATA_CONFIG.DATA_PATH ../data/lyft \
    DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test kitti_infos_train.pkl \
    DATA_CONFIG.load_p2_test_score /home/yy785/datasets/lyft_release_test/training/pp_score_fw70_2m_r0.3

cd ${proj_root_dir}
generate_pl ${proj_root_dir}/downstream/OpenPCDet/output/${target_domain}_models/${target_model}/pl/eval/epoch_no_number/train/trainset/result.pkl ${proj_root_dir}/p2_score/intermediate_results/${pl_output}.pkl

# create the dataset
echo "=> Generating pseudolabel dataset"
cd ${proj_root_dir}/downstream/OpenPCDet/data
mkdir ${pl_output}
cp -r ./lyft/training ./${pl_output}
ln -s ${proj_root_dir}/downstream/OpenPCDet/data/lyft/ImageSets ./${pl_output}/
ln -s ${proj_root_dir}/downstream/OpenPCDet/data/lyft/kitti_infos_val.pkl ./${pl_output}/
cd ./${pl_output}/training
if [ -L "label_2" ]; then
    rm label_2
fi

# run data pre-processing
echo "=> pre-processing pl dataset"
cd ${proj_root_dir}/downstream/OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset update_groundtruth_database tools/cfgs/dataset_configs/lyft_dataset_hindsight.yaml ../data/${pl_output} ${proj_root_dir}/downstream/OpenPCDet/data/lyft/kitti_infos_train.pkl ${proj_root_dir}/p2_score/intermediate_results/${pl_output}.pkl