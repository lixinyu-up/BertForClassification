import os
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer
from model import JointBERT


def get_intent_labels():
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), encoding='utf-8')]


def predict(texts, classifizer_task, threshold = 0.9):
    results = []
    for text in tqdm(texts, desc="predict"):
        inputs = tokenizer(text, return_tensors = "pt")
        """
        inputs:
        {'input_ids': tensor([[19, 20179,  9423,  2296, 17634, 2611, 4, 3]]), 
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 2]]), 
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
        """
        inputs["intent_label_ids"] = None
        outputs = model(**inputs)
        if classifizer_task == "Single_label":
            scores = torch.nn.functional.softmax(outputs[1], dim=1) # 单标签
            for score in scores:
                score = list(score.detach().numpy())
                idx = score.index(max(score))
                results.append(intent_label_lst[idx])
        elif classifizer_task == "Multi_label":
            scores = 1/ (1 + torch.exp(-outputs[1])) # 多标签        
            for item in scores:
                preds = []
                score = []
                for idx, s in enumerate(item):
                    if s > threshold:
                        preds.append(intent_label_lst[idx])
                        score.append(round(float(s.detach().cpu().numpy()), 4))
                results.append({"text": text, "preds": preds, "scores": score})
    return results

def analysis(labels, classifizer_task, results):
    df = pd.DataFrame()
    df["text"] = texts
    df["label"] = labels

    if classifizer_task == "Single_label":       
        df["pred"] = results    
    
    elif classifizer_task == "Multi_label":
        preds = []
        scores = []
        for res in results:
            preds.append(", ".join(res["preds"]))
            scores.append(res["scores"])

        df["pred"] = preds
        df["score"] = scores
        
    df["isTrue"] = (df["label"] == df["pred"])
    print("******** accuary: {} ********".format(df["isTrue"].mean()))
    return df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="./data/dataname/test", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="./result", type=str, help="Output file for prediction")
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model selected in the model")
    parser.add_argument("--model_dir", default="./save", type=str, help="Path to save, load model")
    parser.add_argument("--threshold", default=0.9, type=float, help="Threshold for classification")
    parser.add_argument("--classification_task", "-ct", default="Single_label", type=str, help="task for classification")

    pred_config = parser.parse_args()
    args = torch.load(os.path.join(pred_config.model_dir, pred_config.model_type, "training_args.bin"))
    intent_label_lst = get_intent_labels()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = JointBERT.from_pretrained(os.path.join(args.model_dir, args.model_type), args=args, intent_label_lst=intent_label_lst)

    model.to("cpu")
    model.eval()

    with open(os.path.join(pred_config.input_file, "seq.txt")) as f1, open(os.path.join(pred_config.input_file, "label.txt")) as f2:
        texts = [i.strip() for i in f1]
        labels = [j.strip() for j in f2]
    
    classifizer_task = pred_config.classification_task
    results = predict(texts, classifizer_task = classifizer_task, threshold = 0.92)
    df = analysis(labels, classifizer_task, results)
    
    if not os.path.exists(pred_config.output_file):
        os.makedirs(pred_config.output_file)
    df.to_csv(os.path.join(pred_config.output_file, "{}_epochs_{}_result.csv".format(args.model_type, int(args.num_train_epochs))), index=False)