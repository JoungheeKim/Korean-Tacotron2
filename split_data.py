import hydra
import os
from tacotron2.utils import reset_logging, set_seed, get_abspath
from tacotron2.configs import SplitDataConfig
from hydra.core.config_store import ConfigStore
import logging
from sklearn.model_selection import train_test_split
import tgt
from pathlib import Path
import unicodedata
from dataclasses import _MISSING_TYPE, dataclass
import string
import re
from g2pk import G2p

def init():
    cs = ConfigStore.instance()

    ## base
    cs.store(group="base", name='split_data', node=SplitDataConfig)


@hydra.main(config_path=os.path.join(".", "configs"), config_name="preprocess")
def main(cfg):

    print(cfg)

    ## initialize remove token
    special_regex = re.compile('[%s]' % re.escape(string.punctuation))
    remove_regex = re.compile('[%s]' % re.escape('"#$%&()*+-/:;<=>@[\\]^_{}”“'))

    ## this is not supported in Wav2vec2 vocabs
    unexpected_letters = re.compile('[%s]' % re.escape("àâäèéêëîïôöœùûüÿçÀÂÄÈÉÊËÎÏÔÖŒÙÛÜŸÇß"))

    selected_token = {
        '’' : "'",
        "," : ","
    }

    ## Resent Logging
    reset_logging()

    args = cfg.base

    ############# INIT #################
    ## set meta and audio data path of LJSpeech-1.1
    audio_path = get_abspath(args.audio_path)
    script_path = get_abspath(args.script_path)

    assert os.path.exists(audio_path), 'There is no file in this audio_path [{}]'.format(audio_path)
    assert os.path.exists(script_path), 'There is no file in this script_path [{}]'.format(script_path)

    ## the path for splited metadata
    save_script_path = get_abspath(args.save_script_path)
    os.makedirs(save_script_path, exist_ok=True)

    with open(script_path, 'r') as f:
        scripts = f.readlines()

    logging.info("[Start] make gridtext information")

    ## 1. building grid
    percent = 100/len(scripts) if len(scripts) > 0 else 1
    new_scripts = list()
    removed_count = 0
    for script_idx, script in enumerate(scripts):
        if script_idx % 50 == 0:
            logging.info("{:.4}% progressed   [{}] passed, [{}] removed".format(percent*script_idx, len(new_scripts), removed_count))

        try:
            script = script.strip()
            items = script.split('|')
            temp_path = os.path.join(audio_path, items[0])
            if Path(temp_path).suffix == '':
                temp_path = '{}.wav'.format(temp_path)

            ## should fix it for your own dataset
            transcript = items[2]
            transcript = remove_regex.sub('', transcript)

            for before, after in selected_token.items():
                transcript.replace(before, after)

            new_scripts.append("|".join([temp_path, transcript]))
        except Exception as e:
            removed_count+=1
            logging.error(str(e))
        
    logging.info("[End] make gridtext information, [{}] passed, [{}] removed".format(len(new_scripts), removed_count))

    logging.info("[Start] make splited script data")
    ## 2. split data
    train_scripts, val_scripts = train_test_split(new_scripts, test_size=args.test_size)

    with open(os.path.join(save_script_path, "train.txt"), 'w') as write_f:
        for train_item in train_scripts:
            print(train_item, file=write_f)

    with open(os.path.join(save_script_path, "dev.txt"), 'w') as write_f:
        for dev_item in val_scripts:
            print(dev_item, file=write_f)

    logging.info("[End] make splited script data")


if __name__ == "__main__":
    init()
    main()

