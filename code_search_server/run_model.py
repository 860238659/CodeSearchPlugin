from model import Model
from args import parse_args
import logging
import numpy as np
import torch
import pymysql
import pandas as pd
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 nl_tokens,
                 nl_ids,
    ):
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


def convert_examples_to_features_unixcoder(nl, tokenizer, args):
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(nl_tokens, nl_ids)


class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, source_list):
        self.examples = []
        for nl in source_list:
            self.examples.append(convert_examples_to_features_unixcoder(nl, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].nl_ids))


def evaluate(args, model, tokenizer, source_list, eval_when_training=False):
    # 连接数据库
    mydb = pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="mysql",
    )
    dataset_class = TextDataset_unixcoder

    # 查询文本
    query_dataset = dataset_class(tokenizer, args, source_list)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation on %s *****" % args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nl_vecs = []
    for batch in query_dataloader:
        logger.info("正在遍历query_dataloader")
        nl_inputs = batch.to(args.device)
        logger.info(nl_inputs.shape)
        with torch.no_grad():
            if args.model_type == "base":
                nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    logger.info("query_dataloader完成遍历")

    nl_vecs = np.concatenate(nl_vecs, 0)

    df = pd.read_sql_query("SELECT code_vec FROM cj_code", mydb)
    code_vecs = []
    code_vec_list = df['code_vec'].tolist()
    for code_vec in code_vec_list:
        code_vecs_0 = code_vec.replace('\n', '').replace('[', '').replace(']', '').split(' ')
        res_code = []
        for temp in code_vecs_0:
            if len(temp) != 0:
                res_code.append(float(temp))
        code_vecs.append(res_code)
    code_vecs = np.array(code_vecs)

    # 计算相似度分数
    similarity_scores = np.matmul(nl_vecs, code_vecs.T)
    top_12_indices = np.argsort(similarity_scores, axis=-1, kind='quicksort', order=None)[:, :-13:-1]

    result = []
    for i in range(len(nl_vecs)):
        codes = []
        for j in range(12):
            index = top_12_indices[i][j]
            # 查找每个index对应的url
            mycursor = mydb.cursor()
            sql = "SELECT code FROM cj_code WHERE id=%s"
            val = (index,)
            mycursor.execute(sql, val)
            res = mycursor.fetchone()

            if res is not None:
                codes.append(res[0])
        result.append(codes)

    return result


def create_model(args, model, tokenizer, config=None):
    if args.model_type == "base":
        model = Model(model)
    # logger.info(model.model_parameters())
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


    # build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained('DeepSoftwareAnalytics/CoCoSoDa')
    model = RobertaModel.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
    model = create_model(args, model, tokenizer, config)
    output_dir = 'D:\OneDrive_1\OneDrive\桌面\queue\model.bin'
    model.load_state_dict(torch.load(output_dir, map_location='cpu'), strict=False)
    model.to(args.device)

    try_list = ['Increment an Int8 type value.', 'Removes the element at the specified location',
                'Remove last element', 'Iterator next value']
    outputs = evaluate(args, model, tokenizer, try_list)

    print(outputs)


if __name__ == "__main__":
    main()