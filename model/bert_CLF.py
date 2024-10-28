import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BertCLF(nn.Module):
    def __init__(self, features, config, args):
        super(BertCLF, self).__init__()
        self.config = config
        self.args = args
        self.device = args.device[0] if isinstance(args.device, list) else args.device
        self.label_features = features.to(self.device)  # 确保label_features在正确的设备上
        self.dropout = nn.Dropout(config['dropout_prob'])

        self.bert = AutoModel.from_pretrained(args.pretrained)  # 加载预训练的 BERT 模型
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_features.size(0))

    def forward(self, dtype, input_ids, attention_mask,
                neg_agree_input_ids=None, neg_agree_attention_mask=None,
                neg_consc_input_ids=None, neg_consc_attention_mask=None,
                neg_extra_input_ids=None, neg_extra_attention_mask=None,
                neg_neuro_input_ids=None, neg_neuro_attention_mask=None,
                neg_open_input_ids=None, neg_open_attention_mask=None,
                label_type=None):
        
        device = input_ids.device  # 获取当前输入张量所在的设备

        # 将所有输入张量移到当前设备上
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if neg_agree_input_ids is not None:
            neg_agree_input_ids = neg_agree_input_ids.to(device)
            neg_agree_attention_mask = neg_agree_attention_mask.to(device)
        if neg_consc_input_ids is not None:
            neg_consc_input_ids = neg_consc_input_ids.to(device)
            neg_consc_attention_mask = neg_consc_attention_mask.to(device)
        if neg_extra_input_ids is not None:
            neg_extra_input_ids = neg_extra_input_ids.to(device)
            neg_extra_attention_mask = neg_extra_attention_mask.to(device)
        if neg_neuro_input_ids is not None:
            neg_neuro_input_ids = neg_neuro_input_ids.to(device)
            neg_neuro_attention_mask = neg_neuro_attention_mask.to(device)
        if neg_open_input_ids is not None:
            neg_open_input_ids = neg_open_input_ids.to(device)
            neg_open_attention_mask = neg_open_attention_mask.to(device)

        if dtype == 'train':
            bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]

            def get_neg_output(neg_input_ids, neg_attention_mask):
                batch_size, num_negs, seq_len = neg_input_ids.size()
                neg_input_ids = neg_input_ids.view(-1, seq_len)  # 展开负样本维度
                neg_attention_mask = neg_attention_mask.view(-1, seq_len)
                neg_output = self.bert(neg_input_ids, neg_attention_mask)['last_hidden_state'][:, 0]
                neg_output = neg_output.view(batch_size, num_negs, -1)  # 恢复负样本维度
                return neg_output

            def get_contrastive_label(bert_output, neg_output):
                # 计算正样本和负样本之间的相似性
                cosine_sim = F.cosine_similarity(bert_output.unsqueeze(1), neg_output, dim=-1)
                # 计算注意力权重
                attention_weights = torch.softmax(cosine_sim, dim=1)
                # 计算正样本与负样本相似性的加权和
                weighted_neg_sim = (attention_weights * cosine_sim).sum(dim=1)
                # 计算total_contrastive_label
                total_contrastive_label = 1 / (1 + weighted_neg_sim)
                return total_contrastive_label.mean()

            total_contrastive_label = 0.0
            if label_type == "Agreeableness" and neg_agree_input_ids is not None:
                neg_agree_output = get_neg_output(neg_agree_input_ids, neg_agree_attention_mask)
                total_contrastive_agree = get_contrastive_label(bert_output, neg_agree_output)
                total_contrastive_label += total_contrastive_agree
            elif label_type == "Conscientiousness" and neg_consc_input_ids is not None:
                neg_consc_output = get_neg_output(neg_consc_input_ids, neg_consc_attention_mask)
                total_contrastive_consc = get_contrastive_label(bert_output, neg_consc_output)
                total_contrastive_label += total_contrastive_consc
            elif label_type == "Extraversion" and neg_extra_input_ids is not None:
                neg_extra_output = get_neg_output(neg_extra_input_ids, neg_extra_attention_mask)
                total_contrastive_extra = get_contrastive_label(bert_output, neg_extra_output)
                total_contrastive_label += total_contrastive_extra
            elif label_type == "Neuroticism" and neg_neuro_input_ids is not None:
                neg_neuro_output = get_neg_output(neg_neuro_input_ids, neg_neuro_attention_mask)
                total_contrastive_neuro = get_contrastive_label(bert_output, neg_neuro_output)
                total_contrastive_label += total_contrastive_neuro
            elif label_type == "Openness" and neg_open_input_ids is not None:
                neg_open_output = get_neg_output(neg_open_input_ids, neg_open_attention_mask)
                total_contrastive_open = get_contrastive_label(bert_output, neg_open_output)
                total_contrastive_label += total_contrastive_open
            else:
                raise ValueError("Invalid label")

            label_embed = F.relu(self.label_features).to(device)  # 确保label_features在正确的设备上

            output = torch.zeros((bert_output.size(0), label_embed.size(0)), device=device)

            for i in range(bert_output.size(0)):
                for j in range(label_embed.size(0)):
                    output[i, j] = self.classifier(bert_output[i] + label_embed[j])[j]

            return output, total_contrastive_label

        elif dtype == 'test':
            bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]

            label_embed = F.relu(self.label_features).to(device)  # 确保label_features在正确的设备上

            output = torch.zeros((bert_output.size(0), label_embed.size(0)), device=device)

            for i in range(bert_output.size(0)):
                for j in range(label_embed.size(0)):
                    output[i, j] = self.classifier(bert_output[i] + label_embed[j])[j]

            return output
