import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoConfig, RobertaModel 
from src.contrastive import SupConLoss

from src.layer import MLP, EnhancedLSTM, MlpSigmoid, init_esim_weights

class BertBaseline(nn.Module):
    def __init__(self, config):
        super(BertBaseline, self).__init__()
        bert_config = AutoConfig.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.speaker_embedding = nn.Embedding(len(config.speaker_dict), config.spk_emb_dim)
        self.contrastive = SupConLoss()

        self.contrastive_modality = SupConLoss(temperature=config.cl_temp)
        self.contrastive_utterance = SupConLoss(temperature=config.cl_temp)

        self.text_linear = nn.Sequential(
            nn.Linear(bert_config.hidden_size, config.text_hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(config.audio_in_size, config.audio_hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.video_linear = nn.Sequential(
            nn.Linear(config.video_in_size, config.video_hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        use_dim = 3

        self.classifiers = nn.Linear(
            config.text_hidden_size * (2 + use_dim * 2),
            len(config.emotion_dict))

        self.lstm = EnhancedLSTM(
            "drop_connect",
            config.text_hidden_size * 4,
            config.text_hidden_size * 2,
            num_layers=2,
            ff_dropout=0.3,
            recurrent_dropout=0.3,
            bidirectional=True)

        self.text_modality = MLP(config.text_hidden_size, config.text_hidden_size)
        self.audio_modality = MLP(config.audio_hidden_size, config.audio_hidden_size)
        self.video_modality = MLP(config.video_hidden_size, config.video_hidden_size)

        self.text_utterance = MLP(config.text_hidden_size, config.text_hidden_size)
        self.audio_utterance = MLP(config.audio_hidden_size, config.audio_hidden_size)
        self.video_utterance = MLP(config.video_hidden_size, config.video_hidden_size)

        self.text_align_linear = MLP(config.text_hidden_size * 2, config.align_size)
        self.audio_align_linear = MLP(config.audio_hidden_size * 2, config.align_size)
        self.video_align_linear = MLP(config.video_hidden_size * 2, config.align_size)

        self.text_confident = MlpSigmoid(config.text_hidden_size * 2, 1)
        self.audio_confident = MlpSigmoid(config.audio_hidden_size * 2, 1)
        self.video_confident = MlpSigmoid(config.video_hidden_size * 2, 1)

        self.text_predict = MLP(config.text_hidden_size * 2, len(config.emotion_dict))
        self.audio_predict = MLP(config.audio_hidden_size * 2, len(config.emotion_dict))
        self.video_predict = MLP(config.video_hidden_size * 2, len(config.emotion_dict))

        self.prototype_feature = [0 for _ in range(len(config.emotion_dict))]
        self.prototype_num = [0 for _ in range(len(config.emotion_dict))]

        self.class_weight=torch.tensor([1.0] + [round(config.label_weight, 1)] * (len(config.emotion_dict) - 1)).to(config.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weight)
        self.config = config
        self.apply(init_esim_weights)
        self.bert = RobertaModel.from_pretrained(config.bert_path)
    
    def extract_text(self, input_ids, input_masks, input_segments, input_indices, indices_nums):

        output = self.bert(input_ids, input_masks, input_segments)[0]
        starts = torch.gather(output, 1, input_indices[..., 0].unsqueeze(-1).repeat(1, 1, output.shape[-1]))
        ends = torch.gather(output, 1, (input_indices[..., 1] - 1).unsqueeze(-1).repeat(1, 1, output.shape[-1]))
        output = (starts + ends) / 2

        res = []
        for i in range(len(indices_nums)):
            res.append(output[i, :indices_nums[i]])
        
        return res
    
    def flatten2grid(self, input, dialogue_nums):
        max_num, total_lens = max(dialogue_nums), len(dialogue_nums)
        res = input.new_zeros([total_lens, max_num, input.shape[-1]])
        cur_start = 0
        for i in range(len(dialogue_nums)):
            cur_end = cur_start + dialogue_nums[i]
            res[i, :dialogue_nums[i]] = input[cur_start : cur_end]
            cur_start = cur_end
        return res
    
    def decompose(self, text, audio, video):
        """
        : text: [batch_size, text_hidden_size]
        : audio: [batch_size, audio_hidden_size]
        : video: [batch_size, video_hidden_size]
        """

        batch_size = text.shape[0]

        text_modality = self.text_modality(text)
        audio_modality = self.audio_modality(audio)
        video_modality = self.video_modality(video)
        modality_concat = torch.cat([text_modality, audio_modality, video_modality], dim=0)
        indicator_list = torch.cat((text.new_ones(batch_size), text.new_zeros(batch_size) + 1, text.new_zeros(batch_size) + 2), dim=0)
        modality_matrix = (indicator_list.unsqueeze(0) == indicator_list.unsqueeze(1))
        modality_loss = self.contrastive_modality(modality_concat, mask=modality_matrix)

        text_utterance = self.text_utterance(text_modality)
        audio_utterance = self.audio_utterance(audio_modality)
        video_utterance = self.video_utterance(video_modality)
        utterance_concat = torch.cat([text_utterance, audio_utterance, video_utterance], dim=0)
        utterance_list = torch.cat([torch.arange(batch_size), torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(text.device)
        utterance_matrix = (utterance_list.unsqueeze(0) == utterance_list.unsqueeze(1))
        utterance_loss = self.contrastive_utterance(utterance_concat, mask=utterance_matrix)

        text_rep = text_modality + text_utterance
        audio_rep = audio_modality + audio_utterance
        video_rep = video_modality + video_utterance

        return text_rep, audio_rep, video_rep, modality_loss + utterance_loss

    def confident(self, text, audio, video, input_label):
        text_prediction = self.text_predict(text)
        audio_prediction = self.audio_predict(audio)
        video_prediction = self.video_predict(video)

        criterion = nn.CrossEntropyLoss(weight=self.class_weight)
        text_loss_pri = criterion(text_prediction, input_label)
        audio_loss_pri = criterion(audio_prediction, input_label)
        video_loss_pri = criterion(video_prediction, input_label)

        text_prediction = F.softmax(text_prediction, -1)
        audio_prediction = F.softmax(audio_prediction, -1)
        video_prediction = F.softmax(video_prediction, -1)

        text_label = torch.gather(text_prediction, 1, input_label.unsqueeze(1))
        audio_label = torch.gather(audio_prediction, 1, input_label.unsqueeze(1))
        video_label = torch.gather(video_prediction, 1, input_label.unsqueeze(1))

        text_confident = self.text_confident(text)
        audio_confident = self.audio_confident(audio)
        video_confident = self.video_confident(video)

        criterion = nn.MSELoss()
        text_loss = criterion(text_confident, text_label)
        audio_loss = criterion(audio_confident, audio_label)
        video_loss = criterion(video_confident, video_label)

        loss = text_loss + audio_loss + video_loss 

        # self.config.use_pri = 'use'
        # if self.config.use_pri == 'use':
        loss += text_loss_pri + audio_loss_pri + video_loss_pri 
        
        sum_conf = text_confident + audio_confident + video_confident + 1e-13

        text_confident = text_confident / sum_conf
        audio_confident = audio_confident / sum_conf
        video_confident = video_confident / sum_conf
        
        return loss, text_confident, audio_confident, video_confident
    
    def context_mixed(self, features, input_lengths, similarity):

        new_features = self.lstm(features, None, input_lengths.cpu())

        batch_size, seq_len, hid_size = new_features.shape

        weight = 1 - features.new_tensor(similarity.tolist()).unsqueeze(1)

        context_feature = []
        for line, lens in zip(new_features, input_lengths):
            l0 = torch.cat((line[1:lens], torch.zeros(1, hid_size).to(self.config.device)), 0)
            l1 = torch.cat((torch.zeros(1, hid_size).to(self.config.device), line[:lens-1]), 0)
            context_feature.append(l0 + l1)
        context_feature = torch.cat(context_feature)

        flatten_feature = []
        for line, lens in zip(features, input_lengths):
            flatten_feature.append(line[:lens])
        flatten_feature = torch.cat(flatten_feature)

        final_features = torch.cat((flatten_feature, context_feature* weight * self.config.sim_weight), -1)

        return final_features 
    
    def alignment(self, text, audio, video, input_label):

        proj_text = self.text_align_linear(text)
        proj_audio = self.audio_align_linear(audio)
        proj_video = self.video_align_linear(video)

        losses = []
        criterion = nn.MSELoss(reduction='none')

        similarities = []

        for i in range(len(self.config.emotion_dict)):
            indices = (input_label == i).nonzero().flatten()
            if len(indices) > 0:
                select_text = proj_text[indices]
                select_audio = proj_audio[indices]
                select_video = proj_video[indices]

                last_num = self.prototype_num[i]

                if last_num > 0:
                    self.prototype_feature[i] = (self.prototype_feature[i] * last_num * 3 + select_text.sum(0) + 
                                       select_audio.sum(0) + select_video.sum(0)) / (last_num * 3 + len(indices) * 3)
                else:
                    self.prototype_feature[i] = (self.prototype_feature[i] * last_num * 3 + select_text.sum(0) + 
                                       select_audio.sum(0) + select_video.sum(0)) / (last_num * 3 + len(indices) * 3)
                
                loss0 = criterion(select_text, self.prototype_feature[i].unsqueeze(0).repeat(len(indices), 1))
                loss1 = criterion(select_audio, self.prototype_feature[i].unsqueeze(0).repeat(len(indices), 1))
                loss2 = criterion(select_video, self.prototype_feature[i].unsqueeze(0).repeat(len(indices), 1))

                def get_sub_loss(lossx):
                    lossy = torch.max(lossx.mean(1) - self.config.margin, torch.zeros_like(lossx.mean(1)))
                    lossz = lossy.sum() / (lossy.nonzero().shape[0] + 1e-13)
                    return lossz
                loss0, loss1, loss2 = map(get_sub_loss, [loss0, loss1, loss2])
                losses += [loss0, loss1, loss2]
                self.prototype_feature[i] = self.prototype_feature[i].detach()

        sim_func = nn.CosineSimilarity(1)
        similarities.append(sim_func(proj_text, proj_audio).tolist())
        similarities.append(sim_func(proj_text, proj_video).tolist())
        similarities.append(sim_func(proj_audio, proj_video).tolist())
        similarities = np.mean(similarities, 0)

        return sum(losses) / len(losses), similarities
    
    def forward(self, kwargs):

        cfg = self.config

        input_ids, input_masks, input_segments, audio, video, dialogue_num, input_labels, input_indices, indices_nums = [kwargs[w] for w in ' \
        input_ids, input_masks, input_segments, audio, video, dialogue_num, input_labels, input_indices, indices_nums '.strip().split(', ')]

        input_labels = kwargs['input_labels']

        pooledoutput = []

        loss_grl = 0

        text = self.extract_text(input_ids, input_masks, input_segments, input_indices, indices_nums)
        text = torch.cat(text)
        text = self.text_linear(text)

        audio = self.audio_linear(audio)

        video = self.video_linear(video)
        
        new_text, new_audio, new_video, contrastive_loss = self.decompose(text, audio, video)

        text = torch.cat((text, new_text * cfg.cl_weight), -1)
        audio = torch.cat((audio, new_audio * cfg.cl_weight), -1)
        video = torch.cat((video, new_video * cfg.cl_weight), -1)

        loss_grl += contrastive_loss * cfg.cl_weight

        loss_conf, text_conf, audio_conf, video_conf = self.confident(text, audio, video, input_labels)
        
        real_text = text
        real_audio = audio
        real_video = video

        pooledoutput = torch.cat(((text_conf * real_text + audio_conf * real_audio + video_conf * real_video) * cfg.conf_weight, real_text + real_audio + real_video), -1)

        loss_sim, similarity = self.alignment(text, audio, video, input_labels)
        pooledoutput = self.flatten2grid(pooledoutput, dialogue_num)
        pooledoutput = self.context_mixed(pooledoutput, dialogue_num, similarity)

        logits = self.classifiers(pooledoutput)

        loss = self.criterion(logits, input_labels)

        loss = loss + loss_grl + loss_sim * cfg.sim_loss + loss_conf  * cfg.conf_loss
        
        return logits, loss