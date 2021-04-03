from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score

from data import DataPreprocess, CompDataSet
from model import ESIM, ESIMV1


import random
import logging
import time
import json

import torch


def validate(model, data):
    with torch.no_grad():
        probs = []
        labels = []
        for step, data in enumerate(data):
            start_time = time.time()
            outputs, _ = model(data["premise"], data["premise_mask"], data["hypothese"], data["hypothese_mask"])
            prob = outputs.tolist()
            probs.extend(prob)
            label = data["label"].tolist()
            labels.extend(label)

        neg_probs, pos_probs = zip(*probs)
        pos_auc = roc_auc_score(labels, pos_probs)
        neg_auc = roc_auc_score(labels, neg_probs)
    return neg_auc, pos_auc


def main():
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--emb_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=False)
    parser.add_argument("--ratio", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--emb_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=int, required=True)

    args = parser.parse_args()

    split_ratio = [float(val) for val in args.ratio.split(",")]

    has_cuda = torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info("start preparing data")
    data_preprocessor = DataPreprocess()
    emb, word_idx_map = data_preprocessor.build_emb_vocab(args.emb_file)
    data_preprocessor.load(args.train_file, use_mask=False, is_test=False) 
    train_dataset, dev_dataset = data_preprocessor.generate_train_dev_dataset(ratio=split_ratio)
    train_dataset, dev_dataset = CompDataSet(train_dataset, word_idx_map, max_len=args.max_length, emb_size=args.emb_size), CompDataSet(dev_dataset, word_idx_map, max_len=args.max_length, emb_size=args.emb_size)
    
    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    logging.info("init model")
    model = ESIM(args.vocab_size, args.emb_size, emb, max_len=args.max_length)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = SGD(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)
    criterion = CrossEntropyLoss()

    if has_cuda:
        model = model.cuda()

    logging.info("start training")
    neg_auc, pos_auc = validate(model, dev_dataset)
    logging.info(f"pre-train neg_auc {str(neg_auc)} pos_auc {str(pos_auc)}")
    for epoch in range(args.epoch):
        running_loss = 0.0
        epoch_acc_num, epoche_total_num = 0, 0
        for step, data in enumerate(train_dataset):
            start_time = time.time()
            optimizer.zero_grad()

            outputs, labels = model(data["premise"], data["premise_mask"], data["hypothese"], data["hypothese_mask"])
            loss = criterion(outputs, data["label"])
            loss.backward()

            for gold, pred in zip(data["label"], labels):
                if gold == pred:
                    epoch_acc_num += 1
                epoche_total_num += 1

            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            
            end_time = time.time()
            running_loss += loss.item()
            if step % 100 == 99:
                logging.info(f"epoch: {epoch}, step: {step}, time: {end_time - start_time} loss: {running_loss / 100}")
                running_loss = 0
            if step % 500 == 499:
                neg_auc, pos_auc = validate(model, dev_dataset)
                logging.info(f"pre-train neg_auc {str(neg_auc)} pos_auc {str(pos_auc)}")
                torch.save(model, Path(args.checkpoint) / f"{epoch}_{step}.pt")
        epoch_acc = epoch_acc_num / epoche_total_num
        scheduler.step(epoch_acc)


if __name__ == "__main__":
    main()
