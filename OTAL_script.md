# Envirnmental setting (CUDA 10.1)
# Recommand using Docker Image of 'hub.ciplab.ml/yhjo/pytorch:1.2.0'
$ pip install av
$ pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Training script
# 64GB needed for batch-size 64
# ANet
(v1) 66.647
python main.py ANet RGB \
     --gpus 0 1 \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 15 20 --epochs 25 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb
(v2)
python main.py ANet RGB \
     --gpus 0 1 \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 15 20 --epochs 25 \
     --batch-size 48 -j 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres
(v3)
python main.py ANet RGB \
     --gpus 0 1 \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
     --suffix tuneKinetics
(v4)
python main.py ANet RGB \
     --gpus 0 1 \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 48 -j 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
     --suffix tuneKinetics
