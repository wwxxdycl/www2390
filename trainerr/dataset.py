import json
import math
import torch
from torch.utils.data import TensorDataset, DataLoader, IterableDataset
from sklearn.preprocessing import MultiLabelBinarizer
from utils.utils import *


class Dataset(object):

    def __init__(self,
                 train_data_path,
                 # val_data_path,
                 test_data_path,
                 tokenizer,
                 batch_size,
                 max_length,
                 sbert):

        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sbert = sbert

        self.train_loader, self.test_loader, self.label_features = self.load_dataset(train_data_path, test_data_path)

    # 去掉了验证集的数据加载
    def load_dataset(self, train_data_path, test_data_path):
        train = json.load(open(train_data_path))
        test = json.load(open(test_data_path))

        # 加载句子表征
        train_sents = [clean_string(text) for text in train['content']]
        test_sents = [clean_string(text) for text in test['content']]

        # 加载负样本表征
        train_negs = []
        for entry in train['hardneg']:
            cleaned_entry = {k: [clean_string(v) for v in texts] for k, texts in entry.items()}
            train_negs.append(cleaned_entry)

        # 加载标签表征
        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform(train['labels'])
        test_labels = mlb.transform(test['labels'])
        print("Numbers of labels: ", len(mlb.classes_))

        # 创建标签特征
        label_features = self.create_features(train, mlb)

        # 编码数据
        train_loader = self.encode_train_data(train_sents, train_negs, train_labels, shuffle=True)
        test_loader = self.encode_test_data(test_sents, test_labels, shuffle=False)

        return train_loader, test_loader, label_features

    def create_features(self, train_data, mlb):
        label2id = {v: k for k, v in enumerate(mlb.classes_)}
        features = torch.zeros(len(label2id), 768)
        for label, id in tqdm(label2id.items()):
            features[id] = get_label_embedding(self.sbert, label)[0]
        return features

    # def encode_train_data(self, train_sents, train_negs, train_labels, shuffle=False):
    #     # 编码正样本
    #     X_train = self.tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True,
    #                                                max_length=self.max_length, return_tensors='pt')
    #
    #     # 编码负样本
    #     neg_samples = {label: [] for label in
    #                    ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']}
    #     for neg in train_negs:
    #         for label, texts in neg.items():
    #             neg_encoded = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True,
    #                                                            max_length=self.max_length, return_tensors='pt')
    #             neg_samples[label].append(neg_encoded)
    #
    #     # 转换标签为tensor
    #     y_train = torch.tensor(train_labels, dtype=torch.long)
    #
    #     # 创建TensorDataset
    #     tensors = [X_train['input_ids'], X_train['attention_mask'], y_train]
    #     for label in ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']:
    #         input_ids = torch.stack([item['input_ids'] for item in neg_samples[label]], dim=1)
    #         attention_mask = torch.stack([item['attention_mask'] for item in neg_samples[label]], dim=1)
    #         tensors.extend([input_ids, attention_mask])
    #
    #     train_tensor = TensorDataset(*tensors)
    #
    #     # 创建DataLoader
    #     train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=shuffle)
    #
    #     return train_loader

    def encode_train_data(self, train_sents, train_negs, train_labels, shuffle=False):
        # 编码正样本
        X_train = self.tokenizer.batch_encode_plus(train_sents, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')

        # 编码负样本
        neg_samples = {}
        max_length = 0

        for label in ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']:
            neg_texts = [neg[label] for neg in train_negs]  # 获取负样本文本列表
            neg_texts = [item for sublist in neg_texts for item in sublist]  # 展开嵌套的列表
            max_length = max(max_length, max(len(text.split()) for text in neg_texts))

            neg_encoded = [self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt') for text in neg_texts]

            input_ids = torch.stack([item['input_ids'].squeeze(0) for item in neg_encoded])
            attention_mask = torch.stack([item['attention_mask'].squeeze(0) for item in neg_encoded])

            neg_samples[label] = {
                'input_ids': input_ids.view(-1, 10, input_ids.size(1)),
                'attention_mask': attention_mask.view(-1, 10, attention_mask.size(1))
            }

        # 转换标签为tensor
        y_train = torch.tensor(train_labels, dtype=torch.long)

        # 创建TensorDataset
        train_tensor = TensorDataset(
            X_train['input_ids'],
            X_train['attention_mask'],
            y_train,
            neg_samples['Agreeableness']['input_ids'],
            neg_samples['Agreeableness']['attention_mask'],
            neg_samples['Conscientiousness']['input_ids'],
            neg_samples['Conscientiousness']['attention_mask'],
            neg_samples['Extraversion']['input_ids'],
            neg_samples['Extraversion']['attention_mask'],
            neg_samples['Neuroticism']['input_ids'],
            neg_samples['Neuroticism']['attention_mask'],
            neg_samples['Openness']['input_ids'],
            neg_samples['Openness']['attention_mask']
        )

        # 创建DataLoader
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=shuffle)

        return train_loader

    def encode_test_data(self, test_sents, test_labels, shuffle=False):
        # 编码正样本
        X_test = self.tokenizer.batch_encode_plus(test_sents, padding=True, truncation=True, max_length=self.max_length,
                                                  return_tensors='pt')

        # 转换标签为tensor
        y_test = torch.tensor(test_labels, dtype=torch.long)

        # 创建TensorDataset
        test_tensor = TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test)

        # 创建DataLoader
        test_loader = DataLoader(test_tensor, batch_size=self.batch_size, shuffle=shuffle)

        return test_loader
