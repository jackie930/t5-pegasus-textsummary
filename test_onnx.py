#! -*- coding: utf-8 -*-
import jieba
import numpy as np
import datetime
import psutil
import onnxruntime
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer

config_path = './chinese_t5_pegasus_base/config.json'
checkpoint_path = './chinese_t5_pegasus_base/model.ckpt'
dict_path = './chinese_t5_pegasus_base/vocab.txt'
encoder_model = './chinese_t5_pegasus_base/t5_encoder_dynamic.onnx'
decoder_model = './chinese_t5_pegasus_base/t5_decoder_dynamic.onnx'
x1 = '蓝蓝的天上有一朵白白的云'
x2 = '嗯，它的黑胡椒味道非常浓郁的路口，之后有顾问威拉还有骨牛肉的消息，非常好吃，听一家这个范式查看分开中的，但是他这个社会一块钱的包装竟然还能让将只露出来范围这个包装的蜜蜂做得一点都不好去一下企业家你犯比较好吃，颗粒饱满，野蛮圆润的，有点像那种山东大米的口感，还有点侵权的味道，只是他这个包装可以让我究竟很久，还收了我一块钱。'
x3 = '原文 来源|零壹财经作者|任俊东12月1日，国家互联网信息办公室发布关于《常见类型移动互联网应用程序（App）必要个人信息范围》公开征求意见的通知。此次《意见稿》规定了支付、借贷、银行等38类常见类型App必要个人信息范围，明确App必要个人信息界限，不得因非必要信息拒绝用户安装使用。零壹财经自今年3月起开展了手机App评测工作，通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。上月，全国App个人信息保护监管会召开。会上阿里、腾讯、字节等互联网巨头遭监管点名批评：在App个人信息保护工作方面，存在思想漠视、侥幸心理、技术对抗三类问题。1.对38类App必要个人信息范围向社会征求意见针对此次《意见稿》，国家网信办表示，近年来App广泛应用在促进经济社会发展、服务民生等方面发挥了重要作用。同时，App超范围收集、强制收集用户个人信息普遍存在，用户拒绝同意就无法安装使用。为落实《中华人民共和国网络安全法》关于个人信息收集合法、正当、必要的原则，规范App个人信息收集行为，因而明确常见App收集必要个人信息范围。意见反馈时间截止到2020年12月16日。2.12类App无须个人信息，即可使用基本功能服务根据《意见稿》，国家网信办拟规定网络直播、在线影音、短视频、新闻资讯、运动健身、浏览器、输入法、安全管理、电子图书、拍摄美化、应用商店、实用工具类共12类App无须个人信息，即可使用基本功能服务。3.零壹App评测：9成以上App存在违规收集信息问题为规范收集APP信息收集和使用、加强个人信息保护，切实维护收集APP消费者合法权益，并依据相关监管政策法规，零壹财经App评测中心于2020年3月2日启动App评测专项工作。中心相关评测工作得到了App消费者、监管部门、相关企业、行业从业者等多方的广泛关注和支持。通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。4.阿里、腾讯、字节等遭监管点名批评，App个人信息保护进入新的发展阶段11月27日，全国App个人信息保护监管会在北京召开，工信部召集国内互联网行业的头部企业，总结过去半年来App个人信息保护专项整治行动的成果，部署下一阶段整治行动。工信部信息通信管理局副局长鲁春从在会上表示，工信部针对大企业的App进行了全覆盖检测，对阿里巴巴的40余款、字节跳动30余款，腾讯30余款、百度20余款、网易10余款、小米10余款用户下载量大、使用率高的App进行了重点检测，发现存在思想漠视、侥幸心理、技术对抗三类问题。互联网个人信息数据野蛮生长时代已成过去，APP个人信息保护正在迎来新的发展阶段。切实维护用户合法权益，严厉惩处互联网企业违法违规行为是今后互联网监管的常态。企业只有从思想上重视、行动上遵守，把用户的利益作为企业的核心关切，才能持续发展。添加作者微信：daodao0312，可获取《常见类型移动互联网应用程序（App）必要个人信息范围（征求意见稿）》，或您有App评测需求请联系作者。'

if __name__ == '__main__':
    max_c_len = 500
    max_t_len = 200
    tokenizer = Tokenizer(
        dict_path,
        do_lower_case=True,
        pre_tokenize=lambda s: jieba.cut(s, HMM=False)
    )

    # load onnx model，if providers=['CUDAExecutionProvider'],run onnxruntime-gpu,
    # load onnx model，if providers=['CPUExecutionProvider'],run onnxruntime-cpu
    en_session = onnxruntime.InferenceSession(encoder_model, providers=['CUDAExecutionProvider'])
    de_session = onnxruntime.InferenceSession(decoder_model, providers=['CUDAExecutionProvider'])


    class AutoTitle_onnx(AutoRegressiveDecoder):
        """seq2seq解码器
        """

        @AutoRegressiveDecoder.wraps(default_rtype='probas')
        def predict(self, inputs, output_ids, states):
            c_encoded = inputs[0]
            return \
            de_session.run("", {"Decoder-Input-Token": np.float32(output_ids), "Input-Context": np.float32(c_encoded)})[
                0]

        def generate(self, text, topk=1):
            c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
            c_encoded = en_session.run("", {"Encoder-Input-Token": [c_token_ids]})[0][0]
            output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
            return tokenizer.decode(output_ids)


    autotitle_onnx = AutoTitle_onnx(
        start_id=tokenizer._token_start_id,
        end_id=tokenizer._token_end_id,
        maxlen=max_t_len
    )

    # predict
    for i in [x1, x2, x3]:
        t1 = datetime.datetime.now()
        result = autotitle_onnx.generate(i)
        t2 = datetime.datetime.now()
        print("onnx t2 predict time is :", t2 - t1)
        # print("onnx input is :", i)
        print("onnx t2 predict result is :", result)

