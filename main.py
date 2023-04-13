import argparse

from trainer import Trainer
from utils import init_logger, set_seed
from data_loader import load_and_cache_examples
from transformers import BertTokenizer

MODEL_PATH_MAP = {
    "bert": 'bert-base-chinese',
    "macbert": "hfl/chinese-macbert-base", 
    "roberta": "hfl/chinese-roberta-wwm-ext", 
    "roberta_large": "hfl/chinese-roberta-wwm-ext-large",
    "xlnet": "hfl/chinese-xlnet-base",
    "RBT3": "hfl/rbt3",
    "RBTL3": "hfl/rbtl3"
}


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./save", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the model")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
