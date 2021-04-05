from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import DataPreprocess, CompDataSet
from model import ESIM, ESIMV1


import random
import logging
import time
import json

import torch


def test(model, data):
    model.eval()
    with torch.no_grad():
        probs = []
        for data in tqdm(data, total=len(data)):
            outputs = model(data["premise"], data["premise_mask"], data["hypothese"], data["hypothese_mask"])
            prob = outputs["probs"].tolist()
            prob = [str(i[1]) for i in prob]
            probs.extend(prob)
    return probs 


def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--emb_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--emb_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    logging.info("start preparing data")
    data_preprocessor = DataPreprocess(random_state=args.seed)
    emb, word_idx_map = data_preprocessor.build_emb_vocab(args.emb_file)
    data_preprocessor.load(args.test_file, use_mask=False, is_test=True) 
    test_dataset = data_preprocessor.generate_train_dev_dataset(ratio=[1, 0], is_test=True)
    test_dataset = CompDataSet(test_dataset, word_idx_map, max_len=args.max_length, emb_size=args.emb_size)

    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info("load model")
    if has_cuda:
        model = torch.load(args.model)
    else:
        model = torch.load(args.model, map_location=torch.device('cpu'))

    logging.info("start testing")
    probs = test(model, test_dataset)

    logging.info("file written")
    with open(args.output_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(probs))

if __name__ == "__main__":
    main()
