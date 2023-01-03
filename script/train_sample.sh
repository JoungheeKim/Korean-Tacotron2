## 2. KSS Dataset   
## set your data generator path and save path(examples are below:)
GENERATOR_PATH=/code/gitRepo/Korean-Tacotron2/checkpoints/kss_29de09d_4500.pt
SAVE_PATH=results/kss_sample

## train kss non-attentive tacotron
## check config options in [configs/train_kss.yaml]
python train.py \
    base.generator_path=${GENERATOR_PATH} \
    base.save_path=${SAVE_PATH} \
    --config-name train_kss_sample