#!/usr/bin/env python

import warnings

warnings.filterwarnings('ignore')

import os
import argparse
import yaml
from box import Box 
import torch

from src.tools import set_seed, load_param
from src.loader import MyDataLoader as DataLoader_Seq
from src.model import BertBaseline as Model_Seq_Best
from src.engine import LCTrainer as Trainer_Seq

class Pipeline:
    def __init__(self, args):
        name = args.data_name[:4].lower()
        config = Box(yaml.load(open('src/{}_config.yaml'.format(name), 'r', encoding='utf-8'), Loader=yaml.FullLoader))

        if not os.path.exists(config.target_dir):
            os.makedirs(config.target_dir)

        for k in vars(args):
            config[k] = getattr(args, k)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        config.save_name = ''

        set_seed(config.seed)
        self.config = config

    def main(self):

        train_loader, valid_loader, test_loader, emotion_dict, speaker_dict = DataLoader_Seq(self.config).get_data()

        self.config.emotion_dict, self.config.speaker_dict = emotion_dict, speaker_dict

        model = Model_Seq_Best(self.config).to(self.config.device)

        trainer = Trainer_Seq(model, self.config, train_loader, valid_loader, test_loader)

        self.config = load_param(self.config, model, trainer.train_loader)

        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--cuda_index', default=0)
    # parser.add_argument('-dt', '--data_name', default='MELD', choices=['MELD', 'IEMOCAP'])
    parser.add_argument('-dt', '--data_name', default='IEMOCAP', choices=['MELD', 'IEMOCAP'])
    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.main()
