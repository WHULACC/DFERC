

import os
import random
import pickle as pkl

import yaml
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from box import Box

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler
from collections import Counter


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, config):
        self.cfg= config
        bert_path = config.bert_path.replace('/', '_')
        path = os.path.join(config.data_dir, '{}_{}_{}.pkl'.format(config.dataname, config.model_name, bert_path))

        if not os.path.exists(path):
            self.data = Preprocessor(config).manage()
            pkl.dump(self.data, open(path, 'wb'))
        else:
            self.data = pkl.load(open(path, 'rb'))
    
    def collate(self, batch_data):
        input_ids, input_masks, input_segments, video, audio, input_labels, speaker, input_indices = list(zip(*batch_data))
        patch_nums = list(map(len, input_ids))

        def padding(input_data):
            input_data = [w for line in input_data for w in line]
            max_len = max(map(len, input_data))
            input_data = [w + [0] * (max_len - len(w)) for w in input_data]
            return input_data
        
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])

        dialogue_num = list(map(len, video))
        video = np.vstack(video).astype(np.float32)
        audio = np.vstack(audio).astype(np.float32)

        input_labels = [w for line in input_labels for w in line]

        input_indices = [w for line in input_indices for w in line]
        indices_nums = list(map(len, input_indices))
        max_indices = max(indices_nums)
        input_indices = [w + [[0,1]] * (max_indices - len(w)) for w in input_indices]

        speaker = [w for line in speaker for w in line]

        res = {
            'input_ids': input_ids,
            'input_masks': input_masks,
            'input_segments': input_segments,
            'audio': audio,
            'video': video,
            'input_labels': input_labels,
            'dialogue_num': dialogue_num,
            'patch_nums': patch_nums,
            'input_indices': input_indices,
            'indices_nums': indices_nums,
            'speaker': speaker,
        }
        res = {k : torch.tensor(v).to(self.cfg.device) for k, v in res.items()}

        return res

    def get_data(self):
        modes = 'train valid test'
        res = []
        for i, mode in enumerate(modes.split()):
            cur_data = self.data[i]
            r = DataLoader(MyDataset(cur_data), shuffle=True, batch_size=self.cfg.batch_size if i == 0 else 1, collate_fn=self.collate)
            res.append(r)
        res.append(self.data[-2])
        res.append(self.data[-1])
        return res

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class Preprocessor:
    def __init__(self, config=None):
        if config is None:
            config = Box(yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        print(config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.nouse_words = ['\x92', '\x85']

        self.config = config
    
    def get_train_valid_sampler(self, trainset, valid=0.1):
        size = len(trainset)
        split = int(valid * size)
        idx = list(SubsetRandomSampler(list(trainset)))
        return idx[split:], idx[:split]
    
    def build_spk_dict(self):
        lst = self.train_spk_ids + self.valid_spk_ids + self.test_spk_ids
        lst = Counter(lst).most_common()
        lst = [w[0] for w in lst if w[1] > 1]
        d = {'unk': 0}
        for w in lst:
            if w in d: continue
            d[w] = len(d)
        self.speaker_dict = d
    
    def get_split(self):

        def get_ids(mode='train'):
            csv_file = os.path.join(self.config.source_dir, '{}_sent_emo.csv'.format(mode))
            df = pd.read_csv(csv_file)
            ids = sorted(list(set(df['Dialogue_ID'])))
            speakers = list(df['Speaker'])
            return ids, speakers

        train_ids, train_speakers = get_ids('train')
        valid_ids, valid_speakers = get_ids('valid')
        test_ids, test_speakers = get_ids('test')

        return (train_ids, valid_ids, test_ids), (train_speakers, valid_speakers, test_speakers)
    
    def get_split_ie(self):
        def get_id(mode='train'):
            if mode == 'train':
                speakers = [self.videoSpeakers[w] for w in self.train_ids]
            elif mode == 'valid':
                speakers = [self.videoSpeakers[w] for w in self.valid_ids]
            else:
                speakers = [self.videoSpeakers[w] for w in self.test_ids]
            speakers = [w for line in speakers for w in line]
            return speakers
        self.train_spk_ids = get_id('train')
        self.valid_spk_ids = get_id('valid')
        self.test_spk_ids = get_id('test')
    
    def build_dict(self):
        pass
    
    def stack(self, document):
        document = [w[:self.config.max_doc_length - 2] for w in document]
        cur_indices, indices = [], []
        cur_res, res = [self.config.CLS], []
        for i, w in enumerate(document):
            if len(cur_res) + 1 + len(w) >= self.config.max_doc_length:
                res.append(cur_res)
                indices.append(cur_indices)
                cur_res = [self.config.CLS]
                cur_indices = []
            start = len(cur_res)
            cur_res += w
            end = len(cur_res)
            cur_indices.append([start, end])
        if len(cur_res) > 0:
            res.append(cur_res)
            indices.append(cur_indices)
        res = [w + [self.config.SEP] for w in res]
        return res, indices
    
    def transform2indices(self, mode):
        res = []
        ids = {'train': self.train_ids, 'valid':self.valid_ids, 'test': self.test_ids}[mode]
        spks = {'train': self.train_spk_ids, 'valid':self.valid_spk_ids, 'test': self.test_spk_ids}[mode]

        start = 0
        for idx in ids:
            documents = self.videoSentence[idx]
            end = len(documents) + start
        
            documents = list(map(self.tokenizer.tokenize, documents))
            documents, sentence_indices = self.stack(documents)
            
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, documents))
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]

            video = self.videoVisual[idx]
            audio = self.videoAudio[idx]
            labels = self.videoLabels[idx]
            
            speaker = spks[start:end]
            speaker_id = [self.speaker_dict[w if w in self.speaker_dict else 'unk'] for w in speaker]
            res.append([input_ids, input_masks, input_segments, video, audio, labels, speaker_id, sentence_indices])
            start = end

        return res
    
    def manage(self):
        modes = ['train', 'valid', 'test']
        self.emotion_list = []
        res = []
        
        feature_file = os.path.join(self.config.data_dir, '{}_features_raw.pkl'.format(self.config.dataname))

        if self.config.dataname == 'MELD':
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid, _ = pkl.load(open(feature_file, 'rb'))
            (self.train_ids, self.valid_ids, self.test_ids), (self.train_spk_ids, self.valid_spk_ids, self.test_spk_ids) = self.get_split()
            self.valid_ids = [w for w in self.trainVid if w not in self.train_ids]
            self.test_ids = self.testVid
            self.build_spk_dict()
            # self.emotion_dict = {0: 'neutral', 1: 'fear', 2: 'surprise', 3: 'sadness', 4: 'disgust', 5: 'joy', 6: 'anger'}
            self.emotion_dict = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
            self.emotion_dict = {i: w for i, w in enumerate(self.emotion_dict)}

        else:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid = pkl.load(open(feature_file, 'rb'), encoding='latin1')
            self.train_ids, self.valid_ids = self.get_train_valid_sampler(self.trainVid, valid=0.1)
            self.test_ids = self.testVid
            self.get_split_ie()
            self.build_spk_dict()
            emotion_dict = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
            self.emotion_dict = {i: w for i, w in enumerate(emotion_dict)}
        

        for mode in modes:
            print("Start preprocess {}".format(mode))
            r = self.transform2indices(mode)
            res.append(r)
            print("End preprocess {}".format(mode))

        res.append(self.emotion_dict)
        res.append(self.speaker_dict)
        return res
