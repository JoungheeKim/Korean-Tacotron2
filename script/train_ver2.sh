export CUDA_VISIBLE_DEVICES=1

## 2. KSS Dataset   
## set your data generator path and save path(examples are below:)
GENERATOR_PATH=/home/x1113228/personal/gitRepo/data/tacotron2/checkpoints/kss_29de09d_4500.pt
SAVE_PATH=results/kss_ver2

## train kss non-attentive tacotron
## check config options in [configs/train_kss.yaml]
python train.py \
    base.generator_path=${GENERATOR_PATH} \
    base.save_path=${SAVE_PATH} \
    base.train_batch_size=64 \
    base.gradient_accumulation_steps=1 \
    base.loss_masking=True \
    --config-name train_kss