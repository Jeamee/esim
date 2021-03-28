from collections import defaultdict
from typing import Tuple, Any
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


import logging
import random
import torch
import torch.nn as nn
import pandas as pd



def add_mask(row):
    text1 = row[1].text1.split()
    text2 = row[1].text2.split()
    word_idx_map = {}
    for idx, word in enumerate(text1):
        if word in text2:
            if word not in word_idx_map:
                word_idx_map[word] = len(word_idx_map)

            text1[idx] = f"[MASK{word_idx_map[word]}]"

    for idx, word in enumerate(text2):
        if word in text1:
            if word not in word_idx_map:
                word_idx_map[word] = len(word_idx_map)

            text2[idx] = f"[MASK{word_idx_map[word]}]"

    row[1].text1 = (" ").join(text1)
    row[1].text2 = (" ").join(text2)


def load_output_corpus(self, tsv: list, outfile: str) -> None:
    corpus = []
    for sub_tsv in tsv:
        tsv_data = pd.read_csv(
                sub_tsv,
                sep="\t",
                names=["text1", "text2", "label"]
                )
        for row in tsv_data.iterrows():
            corpus.append(row[1].text1)
            corpus.append(row[1].text2)

    with open(outfile, "w") as writer:
        writer.write(" <sep> ".join(corpus))


class DataPreprocess():

    def __init__(self, random_state=43) -> None:
        self.ids = []
        self.premises = []
        self.hypersises = []
        self.labels = []
        self.word_idx_map = {}
        self.emb_matrix = None
        self.sent_pair_neg_map = defaultdict(list)
        self.sent_pair_pos_map = defaultdict(list)

        random.seed(random_state)

    def load(self, tsv: str, use_mask: bool, is_test: bool) -> None:
        tsv_data = pd.read_csv(
                tsv,
                sep="\t",
                names=["text1", "text2", "label"]
                )
        neg_count = 0
        pos_count = 0
        for row in tsv_data.iterrows():
            if use_mask:
                add_mask(row)

            if row[1].label == 1:
                pos_count += 1
                self.sent_pair_pos_map[row[1].text1].append(row[1].text2)
                self.sent_pair_pos_map[row[1].text2].append(row[1].text1)
            elif row[1].label == 0:
                neg_count += 1
                self.sent_pair_neg_map[row[1].text1].append(row[1].text2)
                self.sent_pair_neg_map[row[1].text2].append(row[1].text1)

            self.ids.append(len(self.ids))
            self.premises.append(row[1].text1)
            self.hypersises.append(row[1].text2)

            if is_test:
                self.labels.append("1")
            else:
                self.labels.append(str(row[1].label))

        if not is_test:
            logging.info(f"pos:{pos_count}, neg:{neg_count}")

        if not is_test:
            self.data_augment()

    def data_augment(self, methods=["label transition"]):
        for method in methods:
            if method == "label transition":
                self.data_augment_by_label_transition()

    def data_augment_by_label_transition(self):
        pos_count, neg_count = 0, 0
        pos_added, neg_added = set(), set()
        for key, values in self.sent_pair_pos_map.items():
            for value in values:
                if len(self.sent_pair_pos_map[value]) == 1:
                    continue

                for sent in self.sent_pair_pos_map[value]:
                    if sent == key:
                        continue
                    if (sent, key) in pos_added or (key, sent) in pos_added:
                        continue

                    pos_added.add((key, sent))

                    pos_count += 1
                    self.sent_pair_pos_map[key].append(sent)
                    self.ids.append(len(self.ids))
                    self.labels.append("1")
                    self.premises.append(key)
                    self.hypersises.append(sent)

        for key, values in self.sent_pair_neg_map.items():
            for value in values:
                for sent in self.sent_pair_pos_map[value]:
                    if (sent, key) in neg_added or (key, sent) in neg_added:
                        continue
                    else:
                        neg_added.add((key, sent))

                    neg_count += 1
                    self.sent_pair_neg_map[key].append(sent)
                    self.ids.append(len(self.ids))
                    self.labels.append("0")
                    self.premises.append(key)
                    self.hypersises.append(sent)

        logging.info(f"post label transition data augment,\
                pos:{pos_count}, neg:{neg_count}")

    def generate_train_dev_dataset(self, ratio: list, random_state=43, is_test=False) -> Any:
        assert len(ratio) == 2, "invalid ratio input, check again"
        datas = list(
                zip(self.ids, self.premises, self.hypersises, self.labels)
                )

        if is_test:
            return datas

        train_datas, dev_datas = train_test_split(
                datas,
                test_size=ratio[1],
                train_size=ratio[0],
                random_state=random_state
                )

        return train_datas, dev_datas

    def build_emb_vocab(self, emb_file: str) -> Tuple[torch.Tensor, dict]:
        emb_matrix = []
        with open(emb_file, "r", encoding="utf-8") as reader:
            for idx, line in enumerate(reader.readlines()):
                parts = line.split()
                emb = [float(i) for i in parts[1:]]
                emb_matrix.append(emb)
                self.word_idx_map[parts[0]] = idx

        return torch.tensor(emb_matrix), self.word_idx_map


class CompDataSet(Dataset):

    def __init__(self, data: list, word_idx_map: dict, max_len: int, emb_size: int) -> None:
        super().__init__()
        self.data = data
        self.word_idx_map = word_idx_map
        self.max_len = max_len
        self.emb_size = emb_size
        self.has_cuda = torch.cuda.is_available()

    def pad(self, parts: list):
        length = len(parts)
        if length == self.max_len:
            return [[1] * self.emb_size for _ in range(length)]
        if length > self.max_len:
            tail = True
            while len(parts) != self.max_len:
                if tail:
                    parts.pop()
                    tail = False
                else:
                    parts.pop(0)
                    tail = True
            return [[1] * self.emb_size for _ in range(self.max_len)]
        pads = ["<seq>"] * (self.max_len - length)
        pad_masks = [[1] * self.emb_size for _ in range(length)]
        pad_masks.extend([[0] * self.emb_size for _ in range(self.max_len - length)])
        parts.extend(pads)

        return pad_masks

    def reduce(self, premise, hypothese):
        premise_parts = premise.split()
        hypothese_parts = hypothese.split()
        dct = defaultdict(int)
        for part in premise_parts:
            dct[part] += 1
        for part in hypothese_parts:
            dct[part] += 1

        new_premise_parts, new_hypothese_parts  = [], []
        for part in premise_parts:
            if dct[part] > 1:
                new_premise_parts.append(part)
        for part in hypothese_parts:
            if dct[part] > 1:
                new_hypothese_parts.append(part)

        return new_premise_parts, new_hypothese_parts

    def get_line_indices(self, chars: list) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_masks = self.pad(chars)
        indices = [self.word_idx_map[char] if char in self.word_idx_map else self.word_idx_map["<unk>"] for char in chars]

        return torch.LongTensor(indices), torch.FloatTensor(pad_masks)

    def __getitem__(self, index):
        _, premise, hypothese, label = self.data[index]

        # premise, hypothese = self.reduce(premise, hypothese)
        premise, hypothese = premise.split(), hypothese.split()

        premise, premise_mask = self.get_line_indices(premise)
        hypothese, hypothese_mask = self.get_line_indices(hypothese)
        label = torch.tensor(int(label), dtype=torch.long)

        if self.has_cuda:
            premise = premise.cuda()
            hypothese = hypothese.cuda()
            premise_mask = premise_mask.cuda()
            hypothese_mask = hypothese_mask.cuda()
            label = label.cuda()

        return {
                "premise": premise,
                "hypothese": hypothese,
                "premise_mask": premise_mask,
                "hypothese_mask": hypothese_mask,
                "label": label
                }

    def __len__(self):
        return len(self.data)
