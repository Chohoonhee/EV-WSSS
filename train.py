"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings_DDD17.yaml"
"""
import argparse

from config.settings import Settings
from training.dsec_trainer import OursSupervisedModel as dual_trainer
from training.ddd17_trainer import OursSupervisedModel as ddd17_trainer

import numpy as np
import torch
import random
import os

# random seed
seed_value = 6
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)


   
    if settings.model_name == 'dsec':
        trainer = dual_trainer(settings)
    elif settings.model_name == 'ddd17':
        trainer = ddd17_trainer(settings)
    else:
        raise ValueError('Model name %s specified in the settings file is not implemented' % settings.model_name)



    trainer.train()


if __name__ == "__main__":
    main()
