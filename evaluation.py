#! -*- coding: utf-8 -*-
import jieba
import numpy as np
import pandas as pd
import tqdm
import re
import json
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from boto3.session import Session

config_path = './chinese_t5_pegasus_base/config.json'
checkpoint_path = './chinese_t5_pegasus_base/model.ckpt'
dict_path = './chinese_t5_pegasus_base/vocab.txt'


def process_summary(summary):
    # test = summary.replace('\n\n','\n').split('\n')
    pattern = re.compile(r'.*?beginbegin([\s\S]*?)endend.*?')
    summary_text = ''.join(re.findall(pattern, summary))
    return summary_text


def load_data_customer(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    df = pd.read_excel(filename, engine='openpyxl')
    D = []

    for i in range(len(df)):
        main_file = df['正文'][i].replace('\n', '').replace(' ', '')
        summary = df['摘要'][i]
        summary_text = process_summary(summary).replace('\n', '').replace(' ', '')
        D.append((summary_text, main_file))
    return D


def get_summary_pegusas():
    # bert4keras版本
    max_c_len = 500
    max_t_len = 200
    tokenizer = Tokenizer(
        dict_path,
        do_lower_case=True,
        pre_tokenize=lambda s: jieba.cut(s, HMM=False)
    )

    t5 = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='t5.1.1',
        return_keras_model=False,
        name='T5',
    )

    encoder = t5.encoder
    decoder = t5.decoder
    model = t5.model

    model.load_weights('./best_model.weights')

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

    autotitle = AutoTitle(
        start_id=tokenizer._token_start_id,
        end_id=tokenizer._token_end_id,
        maxlen=max_t_len
    )

    return autotitle

def invoke_endpoint_textrank(data):
    data = {
        "data": data}
    session = Session()

    runtime = session.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName='textrank',
        ContentType="application/json",
        Body=json.dumps(data),
    )

    result = json.loads(response["Body"].read())
    # print ("<<<< result: ",''.join(result['res']['摘要列表']))
    return ''.join(result['res']['摘要列表'])

def evaluation(data, type):
    smooth = SmoothingFunction().method1
    best_bleu = 0.

    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0

    if type == 'pegusas':
        autotitle = get_summary_pegusas()

    title_ls = []
    pred_title_ls = []
    rouge_1_f_score_ls = []
    rouge_2_f_score_ls = []
    content_ls = []
    rouge_l_f_score_ls = []
    bleu_ls = []

    for (title, content) in data:
        total += 1
        title = ' '.join(title).lower()
        if type == 'pegusas':
            pred_title = ' '.join(autotitle.generate(content,
                                                     topk=1)).lower()
        elif type =='textrank':
            pred_title = invoke_endpoint_textrank(content)

        print("content: ", content)
        print("title: ", title)
        print("pred_title: ", pred_title)

        if pred_title.strip():
            scores = Rouge().get_scores(hyps=pred_title, refs=title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[title.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )

            content_ls.append(content)
            title_ls.append(title)
            pred_title_ls.append(pred_title)
            rouge_1_f_score_ls.append(scores[0]['rouge-1']['f'])
            rouge_2_f_score_ls.append(scores[0]['rouge-2']['f'])
            rouge_l_f_score_ls.append(scores[0]['rouge-2']['f'])
            bleu_ls.append(sentence_bleu(
                references=[title.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )
            )

    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total

    res = pd.DataFrame({"content": content_ls, \
                        "title": title_ls, \
                        "pred_title": pred_title_ls, \
                        "rouge_1_f_score": rouge_1_f_score_ls, \
                        "rouge_2_f_score": rouge_2_f_score_ls,
                        "rouge_l_f_score_ls": rouge_l_f_score_ls,
                        "bleu":bleu_ls})

    print (res.head())
    name = 'result_'+str(type)+'.csv'
    res.to_csv(name, index=False, encoding='utf-8')
    print ("finish process!")

    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }


def main():
    data = load_data_customer('./customer.xlsx')
    res = evaluation(data, 'pegusas')
    print (res)
    res = evaluation(data, 'textrank')
    print (res)


if __name__ == '__main__':
    main()
