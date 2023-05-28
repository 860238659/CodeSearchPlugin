import json
import queue
import time
from flask import Flask, request
import threading
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from run_model import create_model, evaluate
from args import parse_args
import logging
from task import Task


logger = logging.getLogger(__name__)
app = Flask(__name__)
request_queue = queue.Queue()

args = parse_args()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.n_gpu = torch.cuda.device_count()
args.device = device

# build model
config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
model = RobertaModel.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
model = create_model(args, model, tokenizer, config)

@app.before_first_request
def poll():
    def process_requests():
        # 该函数使用一个无限循环while True来进行轮询，以检查请求队列req_queue是否为空。
        while True:
            if not request_queue.empty():
                # 获取请求队列的长度，以决定batch size的长度
                request_queue_size = request_queue.qsize()
                # batch size 暂定为4
                batch_size = min(request_queue_size, 4)
                task_list = []  # 用于储存已经封装成task的request
                source_list = []  # 传递给模型的string数据list，即每个request的描述
                for _ in range(batch_size):
                    task = request_queue.get()
                    task_list.append(task)
                    source_list.append(task.code)

                # 调用模型
                outputs = evaluate(args, model, tokenizer, source_list)

                # 对输出结果处理
                for i in range(len(outputs)):
                    result = outputs[i]
                    task_list[i].set_result(result)

            time.sleep(0.01)

    thread = threading.Thread(target=process_requests)
    thread.start()


# 获取后端传递的request，将 request 加入队列末尾
@app.route('/', methods=['POST'])
def handle_post_request():
    # 获取POST请求的请求体，将其解码并存储在变量 data 中
    data = request.args.get('searchKey')
    print(data)
    # 封装请求
    task = Task(data)
    # 请求放入队列
    request_queue.put(task)
    # 返回结果
    result = task.get_result()
    print(result)
    return json.dumps(result)


if __name__ == '__main__':
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    # 启动Flask服务
    app.run()