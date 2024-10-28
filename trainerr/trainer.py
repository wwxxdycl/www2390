import json
import numpy as np
import torch
import torch.nn as nn
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from sentence_transformers import SentenceTransformer
from model.bert_CLF import BertCLF
from trainerr.dataset import Dataset

from sklearn import metrics
from torchmetrics import Precision
from tqdm import tqdm


class TrainerModel(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args

        # 设置设备
        if isinstance(args.device, list):
            self.devices = [torch.device(d) for d in args.device]
        else:
            self.devices = [torch.device(args.device)]

        # 设置sbert的设备
        self.sbert = SentenceTransformer(args.sbert, device=str(self.devices[0]))  # 设置第一个设备

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.train_data_path = args.train_data_path
        self.test_data_path = self.args.test_data_path
        self.dataset = Dataset(self.args.train_data_path,
                               self.args.test_data_path,
                               self.tokenizer,
                               config['batch_size'],
                               config['max_length'],
                               self.sbert)
        self.train_loader, self.test_loader = self.dataset.train_loader, self.dataset.test_loader

        # 模型初始化
        if self.args.mode == "train":
            self.model = BertCLF(self.dataset.label_features.to(self.devices[0]),
                                 config,
                                 args)
            # 使用多个GPU
            if len(self.devices) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=[i for i in range(len(self.devices))])

            self.model.to(self.devices[0])

        elif self.args.mode == "test":
            self.model = torch.load(self.args.checkpoint)
            self.model.to(self.devices[0])

        self.optimizer, self.scheduler = self._get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss()  # 二分类任务的损失函数

    def _get_optimizer(self):
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'layer_norm.bias', 'layer_norm.weight']
        # 列表 no_decay，其中包含不应用权重衰减的参数，
        # 如偏置项和 Layer Normalization 层的偏置和权重
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
             'weight_decay': self.config['weight_decay']},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        # 优化器和优化对象
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['learning_rate'])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_training_steps=self.config['n_epochs'] * len(self.train_loader),
                                                    num_warmup_steps=100)

        return optimizer, scheduler

    # 增加了label值
    def validate(self, dataloader, label_type):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predicted_labels, target_labels = list(), list()

            for i, batch in enumerate(tqdm(dataloader)):
                input_ids, attention_mask, y_true = tuple(t.to(self.args.device[0]) for t in batch)
                output = self.model.forward('test', input_ids, attention_mask,
                                            neg_agree_input_ids=None, neg_agree_attention_mask=None,
                                            neg_consc_input_ids=None, neg_consc_attention_mask=None,
                                            neg_extra_input_ids=None, neg_extra_attention_mask=None,
                                            neg_neuro_input_ids=None, neg_neuro_attention_mask=None,
                                            neg_open_input_ids=None, neg_open_attention_mask=None,
                                            label_type=None
                                            )
                # loss = self.loss_fn(output, y_true.float())
                # y_true = y_true.any(dim=-1)  # 压缩最后两个维度为一维张量
                loss = self.loss_fn(output, y_true.float())
                total_loss += loss.item()

                target_labels.extend(y_true.cpu().detach().numpy())
                predicted_labels.extend(torch.sigmoid(output).cpu().detach().numpy())

            val_loss = total_loss / len(dataloader)

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)

        # 计算多标签分类中单个标签的分类指标
        num_labels = target_labels.shape[1]
        accuracy = []
        micro_f1_scores = []
        macro_f1_scores = []
        precision_scores = []
        recall_scores = []
        auc_scores = []
        rmse_scores = []

        for i in range(num_labels):
            acc = metrics.accuracy_score(target_labels[:, i], predicted_labels.round()[:, i])
            micro_f1 = metrics.f1_score(target_labels[:, i], predicted_labels.round()[:, i], average='micro')
            macro_f1 = metrics.f1_score(target_labels[:, i], predicted_labels.round()[:, i], average='macro')
            precision = metrics.precision_score(target_labels[:, i], predicted_labels.round()[:, i])
            recall = metrics.recall_score(target_labels[:, i], predicted_labels.round()[:, i])
            auc = metrics.roc_auc_score(target_labels[:, i], predicted_labels[:, i])
            rmse = np.sqrt(metrics.mean_squared_error(target_labels[:, i], predicted_labels[:, i]))

            accuracy.append(acc)
            micro_f1_scores.append(micro_f1)
            macro_f1_scores.append(macro_f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            auc_scores.append(auc)
            rmse_scores.append(rmse)

        if label_type == "Agreeableness":
            n = 0
        elif label_type == "Conscientiousness":
            n = 1
        elif label_type == "Extraversion":
            n = 2
        elif label_type == "Neuroticism":
            n = 3
        elif label_type == "Openness":
            n = 4
        else:
            raise ValueError("Invalid label")
        return val_loss, accuracy[n], micro_f1_scores[n], macro_f1_scores[n], precision_scores[n], recall_scores[n], auc_scores[n], rmse_scores[n]


    def step(self, batch, label_type):
        self.model.train()
        input_ids, attention_mask, label, \
        neg_agree_input_ids, neg_agree_attention_mask, \
        neg_consc_input_ids, neg_consc_attention_mask, \
        neg_extra_input_ids, neg_extra_attention_mask, \
        neg_neuro_input_ids, neg_neuro_attention_mask, \
        neg_open_input_ids, neg_open_attention_mask = tuple(t.to(self.args.device[0]) for t in batch)

        # 从批次数据 batch 中获取输入特征 input_ids、注意力掩码 attention_mask 和标签 label
        self.optimizer.zero_grad()
        # 将优化器的梯度置零，以准备进行反向传播
        y_pred, total_contrastive_label = self.model.forward('train', input_ids, attention_mask,
                                    neg_agree_input_ids, neg_agree_attention_mask,
                                    neg_consc_input_ids, neg_consc_attention_mask,
                                    neg_extra_input_ids, neg_extra_attention_mask,
                                    neg_neuro_input_ids, neg_neuro_attention_mask,
                                    neg_open_input_ids, neg_open_attention_mask,
                                                      label_type)  # 前向传播方法,获取预测结果
        # todo:写对比loss
        loss1 = self.loss_fn(y_pred, label.float())
        # loss2 = total_contrastive_label
        loss2 = total_contrastive_label.mean()  # 确保 total_contrastive_label 是标量
        # print('loss1', loss1)
        # print('total_contrastive_label', total_contrastive_label)
        # print('loss2', loss2)
        loss = loss1 + 0.001*loss2
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()


    def train(self, label_type, fold):
        # Essay、Friends、MyPersonality、Pan
        file_name = 'Friends'
        print("数据集：", file_name)
        print("Training...")
        best_score = float("-inf")
        best_epoch = -1
        for epoch in range(self.config['n_epochs']):
            print('epoch:', epoch)
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                loss = self.step(batch, label_type)
                total_loss += loss
                if (i + 1) % 20 == 0 or i == 0 or i == len(self.train_loader) - 1:
                    print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(self.train_loader),
                                                                            total_loss / (i + 1)))
            if i == len(self.train_loader) - 1:
                print("Evaluating...")
                # val_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, p1, p3 = self.validate(self.val_loader)
                test_loss, accuracy, micro_f1, macro_f1, precision_score, recall, auc, rmse = self.validate(
                    self.test_loader, label_type)
                print(
                    "test_loss: {}\nAccuracy: {}\nMicro-F1: {}\nMacro-F1: {}\nPrecision: {}\nRecall: {}\nAuc: {}\nRMSE: {}".format(
                        test_loss,
                        accuracy,
                        micro_f1,
                        macro_f1,
                        precision_score,
                        recall,
                        auc,
                        rmse))

                # 保存每个 epoch 的参数值
                epoch_params = {
                    "Epoch": epoch,
                    "test_loss": test_loss,
                    "Accuracy": accuracy,
                    "Micro-F1": micro_f1,
                    "Macro-F1": macro_f1,
                    "Precision": precision_score,
                    "Recall": recall,
                    "Auc": auc,
                    "RMSE": rmse
                }
                self.save(epoch_params, fold, file_name)

                # 在精确度（Precision）最高的情况下的模型参数，进行记录
                if precision_score > best_score:
                    best_score = precision_score
                    # best_epoch = epoch
                    torch.save(self.model.state_dict(),
                               f"../ckpt/{file_name}/best_model.pt")
                    # best_model_info = {"Best epoch": best_epoch, "test_loss": best_score}
                    # self.save(best_model_info, fold, file_name)

                # return test_loss, accuracy, micro_f1, macro_f1, precision_score, recall, auc, rmse



    def test(self):
        print("Testing...")
        # test_loss, accuracy, micro_f1, macro_f1, ndcg1, ndcg3, p1, p3 = self.validate(self.test_loader)
        val_loss, accuracy, micro_f1, macro_f1, precision_score, recall, auc, rmse = self.validate(self.val_loader)
        print(
            "Val_loss: {}\nAccuracy: {}\nMicro-F1: {}\nMacro-F1: {}\nPrecision: {}\nRecall: {}\nAuc: {}\nRMSE: {}".format(
                val_loss,
                accuracy,
                micro_f1,
                macro_f1,
                precision_score,
                recall,
                auc,
                rmse))


    def save(self, test_metrics, fold, file_name):
        with open(f'../result/{file_name}/test_metrics_fold_{fold}.json', 'a') as f:
            json.dump(test_metrics, f)
            # f.write('\n')

        # with open(f'../result/{file_name}/test_metrics_fold_{fold}.txt', 'a') as f:
        #     f.write(str(test_metrics + '\n'))

