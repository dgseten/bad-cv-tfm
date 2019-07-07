#!/usr/bin/env bash
cd /home/seten/tensorflow-object-detect/models/research
python object_detection/eval.py --logtostderr  --pipeline_config_path="/home/seten/TFM/models/model_shuttlecock_subtract_0/faster_rcnn_resnet101_shuttle.config"     --checkpoint_dir="/home/seten/TFM/models/model_shuttlecock_subtract_0/train" --eval_dir="/home/seten/TFM/models/model_shuttlecock_subtract_0/eval"