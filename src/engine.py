#!/use/bin/env python

import sys
import torch.nn as nn

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tqdm import tqdm
import os

import torch
import numpy as np

from torch.cuda.amp import GradScaler, autocast

class LCTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
    
    def train_iter(self):
        self.model.train()

        train_data = tqdm(self.train_loader, total=self.train_loader.__len__(), file=sys.stdout)
        count, correct_count = 0, 0
        losses = 0
        true_labels, predict_labels = [], []
        scaler = GradScaler()
    
        for idx, data in enumerate(train_data):
            self.config.optimizer.zero_grad()

            with autocast():
                out, loss = self.model(data)
            input_labels = data['input_labels']
            
            scaler.scale(loss).backward()

            scaler.unscale_(self.config.optimizer)

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            scaler.step(self.config.optimizer)
            self.config.scheduler.step()

            scaler.update()

            losses += loss.item()
            true_labels += input_labels.tolist()
            predict_labels += out.argmax(1).tolist()
            correct_count += (out.argmax(1) == input_labels).sum().item()
            count += len(data['input_ids'])

            acc = accuracy_score(true_labels, predict_labels)

            description = "Epoch {}, loss: {:.4f}, acc: {:.4f}".format(self.global_epoch, losses / count, acc)

            train_data.set_description(description)

    def evaluate_iter(self, dataloader=None):
        self.model.eval()
        if dataloader is None:
            dataloader = self.valid_loader
        dataiter = tqdm(dataloader, total=dataloader.__len__(), file=sys.stdout)
        res = []
        correct_count, count = 0, 0
        losses = 0.0
        true_labels, predict_labels = [], []

        for epoch, data in enumerate(dataiter):
            input_labels = data['input_labels']
            input_ids = data['input_ids']
            with torch.no_grad():
                out, loss = self.model(data)
            losses += loss.item()
            res += out.argmax(1).tolist()
            correct_count += (out.argmax(1) == input_labels).sum().item()
            count += len(input_ids)
            true_labels += input_labels.tolist()
            predict_labels += out.argmax(1).tolist()

        acc = accuracy_score(true_labels, predict_labels)
        p, r, f1, _ = precision_recall_fscore_support(true_labels, predict_labels, average='weighted', labels=np.arange(len(self.config.emotion_dict)))
        description = "Valid epoch {}, loss: {:.4f}, acc: {:.4f}, f1:{:.4f}".format(self.global_epoch, losses / count, acc, f1)
        print(description)
        print(np.unique(predict_labels), np.unique(true_labels), self.config.emotion_dict)
        return f1, acc, losses / count

    def evaluate(self, epoch=0):
        modes = 'tav'
        name = [self.config.seed, modes]
        name = '_'.join(list(map(str, name)))
        PATH = os.path.join(self.config.target_dir, "best_{}_{}_{}.pth.tar".format(epoch, name, self.config.save_name))

        self.global_epoch = epoch
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        f1, acc, loss = self.evaluate_iter(self.test_loader)
        print("Test on best epoch {}, loss: {:.4f}, acc: {:.4f}, f1:{:.4f}".format(epoch, loss, acc, f1))
    
    def train(self):
        self.global_epoch = 0
        self.is_distill = False

        best_score, best_iter = 0, 0
        best_name = ''

        for epoch in range(self.config.epoch_size):
            self.global_epoch = epoch

            self.train_iter()
            score, acc, loss = self.evaluate_iter()

            if score > best_score:
                best_score = score
                best_iter = epoch
                modes = 'tav'
                name = [self.config.seed, modes]
                name = '_'.join(list(map(str, name)))
                if best_name != '':
                    os.remove(best_name)
                best_name = os.path.join(self.config.target_dir, "best_{}_{}_{}.pth.tar".format(epoch, name, self.config.save_name))
                torch.save({'epoch': epoch,
                            'model': self.model.cpu().state_dict(),
                            'best_score': best_score},
                            best_name)
                self.model.to(self.config.device)

            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break

        self.evaluate(best_iter)