# 3D Domain Adaptation

## Environment
```bash
conda create --name adaptation python=3.8
conda activate adaptation
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python matplotlib wandb scipy tqdm easydict scikit-learn pillow==8.3.2

# ME
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git checkout c854f0c # 0.5.4
# NOTE: need to run this on a node with GPUs
python setup.py install

# install customized spconv
cd third_party/spconv
python setup.py bdist_wheel
cd ./dist
pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl

# install openpcdet
cd ../../../downstream/OpenPCDet
pip install -r requirements.txt
python setup.py develop

# install ithaca365 devkit
cd ../../..
git clone git@github.com:cxy1997/ithaca365-devkit.git
cd ithaca365-devkit
python setup.py develop

# for managing experiments
pip install hydra-core --upgrade
pip install hydra_colorlog --upgrade
pip install rich
```

## Data path
#### Ithaca365
* data folder: `/home/yy785/projects/adaptation/downstream/OpenPCDet/data/ithaca365/v1.1`

#### Lyft
* data folder: `/home/tz98/continual-DA/downstream/OpenPCDet/data/lyft`
* train idx: `/home/tz98/continual-DA/downstream/OpenPCDet/data/lyft/ImageSets/train.txt`
* val idx: `/home/tz98/continual-DA/downstream/OpenPCDet/data/lyft/ImageSets/val.txt`

# Train on source data
First, go to this directory: `cd downstream/OpenPCDet/tools`.
Then, you can run the following command to train on source data
```
scripts/dist_train.sh <NGPUS> --cfg_file cfgs/<SOURCE_DOMAIN>_models/pointrcnn_hindsight_p2_<TYPE>level.yaml
```

# Generate pseudolabels
Go to the outer scripts directory by calling: `cd ../../scripts/` and then running `generate_pl_<target_domain>.sh`
# Finetune on target data
Once the pseudolabels are generated, you can run the following command to finetune:
```
scripts/dist_train.sh <NGPUS> --cfg_file cfgs/<TARGET_DOMAIN>_models/pointrcnn_hindsight_p2_<TYPE>level.yaml --ckpt <SAVED_SOURCE_DOMAIN_WEIGHTS> --ckpt_save_interval 1 --set DATA_CONFIG.DATA_PATH <PSEUDOLABEL DIRECTORY>
```
This will save the weights at each epoch when finetuning.

# Evaluation on target data
You can run the following command for evaluation:
```
scripts/dist_test.sh <NGPUS> --cfg_file cfgs/<TARGET_DOMAIN>_models/pointrcnn_eval_hindsight_car_pedestrian.yaml --ckpt <SAVED_TARGET_DOMAIN_WEIGHTS>
```