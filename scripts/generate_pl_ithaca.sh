#!/bin/bash

set -x
proj_root_dir=$(pwd)
source_domain="lyft"
target_domain="ithaca365"
target_model="pointrcnn_eval_hindsight_car_pedestrian"
source_model="pointrcnn_hindsight_p2_squashlevel"
pl_output="ithaca365_pl_pointrcnn"

function generate_pl () {
    local result_path=${1}
    local target_path=${2}
    python ${proj_root_dir}/p2_score/p2_score_filtering_lidar_consistency.py result_path=${result_path} save_path=${target_path} dataset="ithaca365" data_paths="ithaca365.yaml"
}

cd ${proj_root_dir}/downstream/OpenPCDet/tools
# generate the preditions on the training set
bash scripts/dist_test.sh 4 --cfg_file cfgs/${target_domain}_models/${target_model}.yaml \
    --extra_tag ${pl_output} --eval_tag trainset \
    --ckpt ../output/${source_domain}_models/${source_model}/default/ckpt/last_checkpoint.pth \
    --set DATA_CONFIG.DATA_PATH ../data/ithaca365 DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test ithaca365_infos_1sweeps_train.pkl

cd ${proj_root_dir}
mkdir p2_score/intermediate_results
generate_pl ${proj_root_dir}/downstream/OpenPCDet/output/${target_domain}_models/${target_model}/${pl_output}/eval/epoch_no_number/train/trainset/result.pkl ${proj_root_dir}/p2_score/intermediate_results/${pl_output}.pkl

# create the dataset
echo "=> Generating pseudolabel dataset"
cd ${proj_root_dir}/downstream/OpenPCDet/data
mkdir ${pl_output}
mkdir ${pl_output}/v1.1
ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${target_domain}/v1.1/data ./${pl_output}/v1.1/
ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${target_domain}/v1.1/v1.1 ./${pl_output}/v1.1/
ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${target_domain}/v1.1/ithaca365_infos_1sweeps_val.pkl ./${pl_output}/v1.1/
cd ./${pl_output}/v1.1

# run data pre-processing
echo "=> pre-processing pl dataset"
cd ${proj_root_dir}/downstream/OpenPCDet
python -m pcdet.datasets.ithaca365.ithaca365_dataset --func update_groundtruth_database \
    --cfg_file tools/cfgs/dataset_configs/ithaca365_dataset_hindsight.yaml \
    --data_path data/${pl_output} \
    --pseudo_labels ${proj_root_dir}/p2_score/intermediate_results/${pl_output}.pkl \
    --info_path ${proj_root_dir}/downstream/OpenPCDet/data/${target_domain}/v1.1/ithaca365_infos_1sweeps_train.pkl