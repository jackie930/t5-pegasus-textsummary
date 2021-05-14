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
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import datetime
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
import flask
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
warnings.filterwarnings("ignore",category=FutureWarning)
# 提前load模型，即起服务的同时加载好模型,这里一定注意加载之前起一个sess，加载完成保存graph，inference的时候使用加载模型时候的graph才能保证每次inference都是同一个graph
# 起sess
global sess
global graph
sess = tf.Session()
K.set_session(sess)
cfg_path = './chinese_t5_pegasus_base/config.json'
checkpoint_path = './chinese_t5_pegasus_base/model.ckpt'
dict_path = './chinese_t5_pegasus_base/vocab.txt'
weight_path='./best_model.h5'
max_c_len = 500
max_t_len = 200
t1= datetime.datetime.now()
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
t2= datetime.datetime.now()
sys.path.append('/opt/program/textrank4zh')

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task


s3_client = boto3.client('s3')

def pegasus_infer(text):
    t1= datetime.datetime.now()
    result=autotitle.generate(text)
    t2= datetime.datetime.now()
    print ("<<<<done")
    return result,str(t2-t1)


@app.route('/ping', methods=['GET'])#路由请求方式GET，默认GET
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

    #parse json in request
#     print ("<<<< flask.request.content_type", flask.request.content_type)

#     data = flask.request.data.decode('utf-8')
#     data = json.loads(data)

#     bucket = data['bucket']
#     image_uri = data['image_uri']

#     download_file_name = image_uri.split('/')[-1]
#     print ("<<<<download_file_name ", download_file_name)

#     s3_client.download_file(bucket, image_uri, download_file_name)
    #local test
    text = '原文 来源|零壹财经作者|任俊东12月1日，国家互联网信息办公室发布关于《常见类型移动互联网应用程序（App）必要个人信息范围》公开征求意见的通知。此次《意见稿》规定了支付、借贷、银行等38类常见类型App必要个人信息范围，明确App必要个人信息界限，不得因非必要信息拒绝用户安装使用。零壹财经自今年3月起开展了手机App评测工作，通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。上月，全国App个人信息保护监管会召开。会上阿里、腾讯、字节等互联网巨头遭监管点名批评：在App个人信息保护工作方面，存在思想漠视、侥幸心理、技术对抗三类问题。1.对38类App必要个人信息范围向社会征求意见针对此次《意见稿》，国家网信办表示，近年来App广泛应用在促进经济社会发展、服务民生等方面发挥了重要作用。同时，App超范围收集、强制收集用户个人信息普遍存在，用户拒绝同意就无法安装使用。为落实《中华人民共和国网络安全法》关于个人信息收集合法、正当、必要的原则，规范App个人信息收集行为，因而明确常见App收集必要个人信息范围。意见反馈时间截止到2020年12月16日。2.12类App无须个人信息，即可使用基本功能服务根据《意见稿》，国家网信办拟规定网络直播、在线影音、短视频、新闻资讯、运动健身、浏览器、输入法、安全管理、电子图书、拍摄美化、应用商店、实用工具类共12类App无须个人信息，即可使用基本功能服务。3.零壹App评测：9成以上App存在违规收集信息问题为规范收集APP信息收集和使用、加强个人信息保护，切实维护收集APP消费者合法权益，并依据相关监管政策法规，零壹财经App评测中心于2020年3月2日启动App评测专项工作。中心相关评测工作得到了App消费者、监管部门、相关企业、行业从业者等多方的广泛关注和支持。通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。4.阿里、腾讯、字节等遭监管点名批评，App个人信息保护进入新的发展阶段11月27日，全国App个人信息保护监管会在北京召开，工信部召集国内互联网行业的头部企业，总结过去半年来App个人信息保护专项整治行动的成果，部署下一阶段整治行动。工信部信息通信管理局副局长鲁春从在会上表示，工信部针对大企业的App进行了全覆盖检测，对阿里巴巴的40余款、字节跳动30余款，腾讯30余款、百度20余款、网易10余款、小米10余款用户下载量大、使用率高的App进行了重点检测，发现存在思想漠视、侥幸心理、技术对抗三类问题。互联网个人信息数据野蛮生长时代已成过去，APP个人信息保护正在迎来新的发展阶段。切实维护用户合法权益，严厉惩处互联网企业违法违规行为是今后互联网监管的常态。企业只有从思想上重视、行动上遵守，把用户的利益作为企业的核心关切，才能持续发展。添加作者微信：daodao0312，可获取《常见类型移动互联网应用程序（App）必要个人信息范围（征求意见稿）》，或您有App评测需求请联系作者。'
    print('Download finished!')
    # inference and send result to RDS and SQS
    with graph.as_default():# 使用保存后的graph做inference
        K.set_session(sess)
        res,infer_time = pegasus_infer(text)
    print ("Done inference! ")
    inference_result = {
        'classes':res,
        'infer_time':infer_time
    }
    _payload = json.dumps(inference_result,ensure_ascii=False)
    return flask.Response(response=_payload, status=200, mimetype='application/json')
    # response类用来包装一下响应内容（为了满足服务器要求的响应标准：header，status，body等），
    # 即手动包装，也可以直接返回，这时
    # flask会自动包装
