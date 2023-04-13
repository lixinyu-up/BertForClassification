import os
import random
import logging

import torch
import numpy as np


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]



def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    results.update(intent_result)
    return results


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }