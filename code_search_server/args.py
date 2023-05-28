import argparse


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