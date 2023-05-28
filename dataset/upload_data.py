from model import Model
import argparse
import logging
import json
import multiprocessing
import numpy as np
import torch
import pymysql
import os
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

logger = logging.getLogger(__name__)
cpu_cont = 16


# 解析args|命令行参数
def parse_args():
    # 使用 Python 标准库中的 argparse 模块创建了一个 ArgumentParser 对象。
    # 这个对象用于解析命令行参数，可以轻松地定义程序需要接受的参数类型、名称、默认值、说明等信息，
    # 方便地生成帮助信息，并进行参数检查和错误提示等操作。
    parser = argparse.ArgumentParser()
    # soda 添加一个命令行选项或参数。
    parser.add_argument('--data_aug_type', default="replace_type", choices=["replace_type", "random_mask", "other"],
                        help="the ways of soda", required=False)
    parser.add_argument('--aug_type_way', default="random_replace_type",
                        choices=["random_replace_type", "replace_special_type", "replace_special_type_with_mask"],
                        help="the ways of soda", required=False)
    parser.add_argument('--print_align_unif_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_ineer_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--only_save_the_nl_code_vec', action='store_true', help='print_align_unif_loss',
                        required=False)
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--agg_way', default="avg", choices=["avg", "cls_pooler", "avg_cls_pooler"],
                        help="base is codebert/graphcoder/unixcoder", required=False)
    parser.add_argument('--weight_decay', default=0.01, type=float, required=False)
    parser.add_argument('--do_single_lang_continue_pre_train', action='store_true',
                        help='do_single_lang_continue_pre_train', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, help='save_evaluation_reuslt', required=False)
    parser.add_argument('--epoch', type=int, default=50,
                        help="random seed for initialization")
    # new continue pre-training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument("--loaded_codebert_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--do_multi_lang_continue_pre_train', action='store_true',
                        help='do_multi_lang_continue_pre_train', required=False)
    parser.add_argument("--couninue_pre_train_data_files",
                        default=["dataset/ruby/train.jsonl", "dataset/java/train.jsonl", ], type=str, nargs='+',
                        required=False,
                        help="The input training data files (some json files).")
    # parser.add_argument("--couninue_pre_train_data_files", default=["dataset/go/train.jsonl",  "dataset/java/train.jsonl",
    # "dataset/javascript/train.jsonl",  "dataset/php/train.jsonl",  "dataset/python/train.jsonl",  "dataset/ruby/train.jsonl",], type=list, required=False,
    #                     help="The input training data files (some json files).")
    parser.add_argument('--do_continue_pre_trained', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_fine_tune', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_whitening', action='store_true',
                        help='do_whitening https://github.com/Jun-jie-Huang/WhiteningBERT', required=False)
    parser.add_argument("--time_score", default=1, type=int, help="cosine value * time_score")
    parser.add_argument("--max_steps", default=100, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    # new moco
    parser.add_argument('--moco_type', default="encoder_queue",
                        choices=["encoder_queue", "encoder_momentum_encoder_queue"],
                        help="base is codebert/graphcoder/unixcoder", required=False)

    # debug
    parser.add_argument('--use_best_mrr_model', action='store_true', help='cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--max_codeblock_num", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument("--mlm_probability", default=0.1, type=float, required=False)

    # model type
    parser.add_argument('--do_avg', action='store_true', help='avrage hidden status', required=False)
    parser.add_argument('--model_type', default="base",
                        choices=["base", "cocosoda", "multi-loss-cocosoda", "no_aug_cocosoda"],
                        help="base is codebert/graphcoder/unixcoder", required=False)
    # moco
    # moco specific configs:
    parser.add_argument('--moco_dim', default=768, type=int,
                        help='feature dimension (default: 768)')
    parser.add_argument('--moco_k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536), which is divided by 32, etc.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true', help='use mlp head')

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")

    parser.add_argument("--lang", default="java", type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=0, type=int,
                        help="Optional Data Flow input sequence length after tokenization.", required=False)

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    return args


# 定义了 InputFeatures 类，作为将原始数据转换为模型输入的中间结果。
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 code,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.code = code


# 用于将原始数据 js 转化成 PyTorch 可以使用的格式。这个函数主要实现以下几个步骤：
def convert_examples_to_features_unixcoder(js, tokenizer, args):
    """convert examples to token ids"""
    # code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code = ' '.join(js['code_tokens'])
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, code)


# 定义了一个继承自 PyTorch 中 Dataset 类的 TextDataset_unixcoder 类，用于处理文本数据集。
class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pooler=None):
        self.examples = []
        data = []
        # n_debug_samples = args.n_debug_samples
        with open(file_path, encoding='utf-8') as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append(js)
                # if args.debug and len(data) >= n_debug_samples:
                #     break
        for js in data:
            self.examples.append(convert_examples_to_features_unixcoder(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids))


def evaluate(args, model, tokenizer, pool, eval_when_training=False):

    logger.info("***** Running evaluation on %s *****" % args.lang)
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 从新的数据库开始操作（连接新数据库）
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="mysql",
    )
    mycursor = mydb.cursor()
    mycursor.execute("CREATE TABLE cj_code (id INT AUTO_INCREMENT PRIMARY KEY, code_vec TEXT, code TEXT)")
    mydb.commit()

    dataset_class = TextDataset_unixcoder

    code_vecs = []
    codes = []

    # 要读取的文件夹路径及生成文件名列表
    folder_name = "train"
    file_names = os.listdir(folder_name)

    # 进入文件夹
    os.chdir(folder_name)

    for file_name in file_names:
        # print(file_name)
        logger.info("  file_name = %s", file_name)
        code_dataset = dataset_class(tokenizer, args, file_name, pool)
        code_sampler = SequentialSampler(code_dataset)
        code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

        # Eval!
        logger.info("  Num codes = %d", len(code_dataset))

        for example in code_dataset.examples:
            codes.append(example.code)

        # 这段代码的作用是对代码文本进行编码，并将其向量表示添加到一个列表 code_vecs 中。
        # 这里使用了 PyTorch 的 DataLoader 类来批处理输入的代码文本数据，批大小为一次处理的代码数量。
        for batch in code_dataloader:
            logger.info("正在遍历code_dataloader")
            # 可以将某些不需要求导的计算操作排除在计算图之外，以避免计算不必要的梯度。因此，这个语句通常用于测试或者评估模型时用于加速计算速度，减少GPU显存的占用。
            with torch.no_grad():
                code_inputs = batch.to(args.device)
                if args.model_type == "base":
                    code_vec = model(code_inputs=code_inputs)
                code_vecs.append(code_vec.cpu().numpy())

    logger.info("code_dataloader完成遍历")

    # 将code_vecs变成了一个拼接好的一维数组。
    code_vecs = np.concatenate(code_vecs, 0)

    logger.info("  code_vecs.length = %d", len(code_vecs))
    logger.info("  codes.length = %d", len(codes))

    # 数据库提交
    sql = "INSERT INTO cj_code (code_vec, code) VALUES (%s, %s)"
    for vec, code in zip(code_vecs, codes):
        val = (vec, code)
        mycursor.execute(sql, val)
        mydb.commit()

    mydb.close()
    print("完成数据填写")


def create_model(args ,model ,tokenizer, config=None):
    if args.model_type ==  "base" :
        model = Model(model)
    logger.info(model.model_parameters())
    return model


def main():
    args = parse_args()
    # set log  输出日志
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # set device 选择可用的gpu设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # 创建多进程池，将其保存到变量 pool 中，以便后续使用。
    pool = multiprocessing.Pool(cpu_cont)

    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained('DeepSoftwareAnalytics/CoCoSoDa')
    model = RobertaModel.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
    model = create_model(args,model,tokenizer,config)
    #
    # # TODO:这一行output_dir需要修改模型的绝对路径
    # output_dir = 'D:\OneDrive_1\OneDrive\桌面\queue\model.bin'
    #
    # model.load_state_dict(torch.load(output_dir, map_location='cpu'), strict=False)
    # model.to(args.device)

    evaluate(args, model, tokenizer, pool)


if __name__ == "__main__":
    main()