#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# -le 1 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [PRETRAINED_PATH] [TRAIN_DATA] (option)"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
echo $PATH1
echo $PATH2

if [ ! -f $PATH1 ]
then
    echo "error: PRETRAINED_PATH=$PATH1 is not a file"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: TRAIN_DATA=$PATH2 is not a dir"
exit 1
fi


ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

if [ -d "obj_det_training" ];
then
    rm -rf ./obj_det_training
fi
mkdir ./obj_det_training
cp ../*.py ./obj_det_training
cp ../*.yaml ./obj_det_training
cp *.sh ./obj_det_training
cp -r ../src ./obj_det_training
cd ./obj_det_training || exit
echo "start training for device $DEVICE_ID"
env > env.log
python obj_det_training.py --train_data=$PATH2 --device_id=$DEVICE_ID --pre_trained=$PATH1 \
 --device_target="GPU" &> log &
cd ..
