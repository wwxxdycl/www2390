import nltk
nltk.data.path.append('../../../../nltk_data')
nltk.data.find('corpora/wordnet')
# 现在可以使用WordNet资源
from sentence_transformers import SentenceTransformer
import re
import math
import random, os
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# # TODO:label表征
# def get_label_embedding(sbert, text):
#     word_embedding = sbert.encode(text, convert_to_tensor=True)
#     return word_embedding

# TODO:策略一label表征用wordnet得到10个近义词，label_emb是原词、近义词的拼接形式
import torch
from nltk.corpus import wordnet
import jieba
def get_synonyms(word, n_synonyms=10):
    synonyms = []
    seg_list = jieba.lcut(word)  # 使用结巴分词将文本分词
    for seg in seg_list:
        synsets = wordnet.synsets(seg)
        for syn in synsets:
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
                if len(synonyms) >= n_synonyms:
                    return synonyms
    return synonyms

def get_text_synonyms(text, n_synonyms=10):
    synonyms = get_synonyms(text, n_synonyms)
    synonyms.append(text)  # 将原始文本也包含在内
    return ' '.join(synonyms)

def get_label_embedding(sbert, text, n_synonyms=10):
    text_with_synonyms = get_text_synonyms(text, n_synonyms)
    print(text_with_synonyms)
    embedding = sbert.encode(text_with_synonyms, convert_to_tensor=True)
    return embedding


def clean_string(string): #清理字符串，去除非字母数字字符，并将字符串转换为小写
    string = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()
    
def normalizeAdjacencyv2(W): #对输入的邻接矩阵进行标准化处理
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1) #计算每行的和
    d = 1/d #取倒数
    D = torch.diag(d) #构造对角矩阵D
    return D @ W 

def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d) #开方取到数
    D = torch.diag(d)
    return D @ W @ D

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True