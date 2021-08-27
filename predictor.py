# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings
import jieba
import datetime
from keras import backend as K
import numpy as np
import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import datetime
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
import flask
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
from keras import backend as K

K.tensorflow_backend._get_available_gpus()


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return self.last_token(decoder).predict([c_encoded, output_ids])

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


# The flask app for serving predictions

app = flask.Flask(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
# 提前load模型，即起服务的同时加载好模型,这里一定注意加载之前起一个sess，加载完成保存graph，inference的时候使用加载模型时候的graph才能保证每次inference都是同一个graph
# 起sess
global sess
global graph
sess = tf.Session()
K.set_session(sess)
cfg_path = './chinese_t5_pegasus_base/config.json'
checkpoint_path = './chinese_t5_pegasus_base/model.ckpt'
dict_path = './chinese_t5_pegasus_base/vocab.txt'
weight_path = './best_model.weights'
max_c_len = 500
max_t_len = 200
t1 = datetime.datetime.now()
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)
t5 = build_transformer_model(
    config_path=cfg_path,
    checkpoint_path=checkpoint_path,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)
encoder = t5.encoder
decoder = t5.decoder
model = t5.model
model.load_weights(weight_path)
# 保存graph
graph = tf.get_default_graph()
autotitle = AutoTitle(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=max_t_len
)
t2 = datetime.datetime.now()

import sys

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # from autogluon import ImageClassification as task

s3_client = boto3.client('s3')


def pegasus_infer(text):
    t1 = datetime.datetime.now()
    result = autotitle.generate(text)
    t2 = datetime.datetime.now()
    print("<<<<done")
    return result, str(t2 - t1)


@app.route('/ping', methods=['GET'])  # 路由请求方式GET，默认GET
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    # parse json in request
    #     print ("<<<< flask.request.content_type", flask.request.content_type)

    #     data = flask.request.data.decode('utf-8')
    #     data = json.loads(data)

    #     bucket = data['bucket']
    #     image_uri = data['image_uri']

    #     download_file_name = image_uri.split('/')[-1]
    #     print ("<<<<download_file_name ", download_file_name)

    #     s3_client.download_file(bucket, image_uri, download_file_name)
    # local test
    data = flask.request.data.decode('utf-8')
    print("<<<<<input data: ", data)
    print("<<<<<input content type: ", flask.request.content_type)

    data = json.loads(data)
    data_input = data['data']

    with graph.as_default():  # 使用保存后的graph做inference
        K.set_session(sess)
        res, infer_time = pegasus_infer(data_input)
    print("Done inference! ")
    inference_result = {
        'result': res,
        'infer_time': infer_time
    }
    _payload = json.dumps(inference_result, ensure_ascii=False)
    return flask.Response(response=_payload, status=200, mimetype='application/json')
    # response类用来包装一下响应内容（为了满足服务器要求的响应标准：header，status，body等），
    # 即手动包装，也可以直接返回，这时
    # flask会自动包装
