# BertForClassification
# 使用Hugging Face Transformers预训练模型进行中文文本意图检测

## Training & Evaluation

```bash
$ conda create -n torch python=3.9
$ conda activate torch
```


```bash
$ python main.py --task {TASK_NAME} \
                 --model_type {MODEL_TYPE} \
                 --do_train --do_eval \

# For dataname
$ python main.py --task dataname  --do_train --do_eval \
                 --num_train_epochs 10 --logging_steps 100 --save_steps 100

# bert
$ python main.py --task dataname  --do_train --do_eval \
                 --num_train_epochs 10

# macbert
$ python main.py --task dataname --do_train --do_eval \
                 --num_train_epochs 10 --model_type macbert

# roberta
$ python main.py --task dataname --do_train --do_eval \
                 --num_train_epochs 10 --model_type roberta

# roberta_large
$ python main.py --task dataname --do_train --do_eval \
                 --num_train_epochs 10 --model_type roberta_large
```

## Prediction

```bash
$ python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH} --classification_task {TASK}

# For dataname
$ python dm_test.py --model_type macbert
$ python dm_test.py --model_type macbert -ct Multi_label
```
