import os
import argparse

from datamodules.wsi_embedding_datamodule import PatchEmbeddingDataModule
from models import ReportModel
from tokenizers import Tokenizer
from trainer import Trainer

reports_json_path = ''

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_fea_length', type=int, default=10000,
                        help='the maximum sequence length of the patch embeddings.')
    parser.add_argument('--max_seq_length', type=int, default=600, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=1, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch.')

    args = parser.parse_args()
    return args


def train():
    args = parse_agrs()
    split_frac = [0.75, 0.12, 0.13]
    tokenizer = Tokenizer(reports_json_path)
    model = ReportModel(args, tokenizer)
    trainer = Trainer(args, model, tokenizer, split_frac)
    metrics = trainer.train()
    print(metrics)





if __name__ == "__main__":
    pass