import csv
import json
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from trainerr.trainer import TrainerModel
sys.path.append("..")
from argparse import ArgumentParser

from config.config import Configuration
import nltk
nltk.data.path.append('/home/Q22301185/nltk_data')
nltk.data.find('tokenizers/punkt')#punkt模块是NLTK中的一个标记器（tokenizer），用于将文本拆分成句子或单词的集合。

from utils.utils import seed_everything
seed_everything(42) #通过设置相同的种子，可以确保在不同的运行环境中生成的随机数序列是一致的

file_name = 'Friends'
label = 'Agreeableness'

if __name__ == '__main__':
    parser = ArgumentParser(description="Model trainer")  # 解析命令行参数
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--pretrained", type=str, default='/home/Q22301185/Bigfive/pre_model/roberta-base',
                        help="Pretrained Transformer Model")
    parser.add_argument("--checkpoint", type=str, default=f'../ckpt/{file_name}/checkpoint_6.pt')
    parser.add_argument("--tokenizer_name", type=str, default='/home/Q22301185/Bigfive/pre_model/roberta-base')
    parser.add_argument("--sbert", type=str, default='/home/Q22301185/Bigfive/pre_model/paraphrase-distilroberta-base-v1')  # 用于生成语言的近义词或近义句，是一个轻量级的RoBERTa模型
    parser.add_argument("--mode", type=str, default='train',
                        help="train/test")
    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在args对象

    config = Configuration.get_config("../config/config.yml")['train']  # 获取配置文件中的参数值


    if args.mode == "train":
        all_test_metrics = []
        for fold in range(1, 6):
            print("fold: ", fold)
            train_data_path = f'/home/Q22301185/Make_Paper/debug/A_hardneg/new_datasets_right/new_datasets_tf/{file_name}/fold_{fold}/train.json'
            test_data_path = f'/home/Q22301185/Make_Paper/debug/A_hardneg/new_datasets_right/new_datasets_tf/{file_name}/fold_{fold}/test.json'
            args.train_data_path = train_data_path
            args.test_data_path = test_data_path
            model_trainer = TrainerModel(config, args)
            model_trainer.train(label, fold)
           

    elif args.mode == "test":
        # 不需要用test函数，先保留不进行修改
        model_trainer.test()
