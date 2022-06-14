# t5-pegasus-textsummary
使用谷歌2020pegasus模型进行中文文档摘要

谷歌于去年年底发布了一个精简型的机器语义分析项目：飞马(PEGASUS)：预先机器学习及训练后的自动文章摘要项目。近期这个项目迎来的新的版本，这个小型项目可以非常精准的自动提取出文章中的摘要，并且只用一千个训练模型就可以生成媲美人类的摘要内容。
利用提取的间隙句进行摘要概括的预训练模型（Pre-training with Extracted Gap-sentences for Abstractive Summarization）。就是设计一种间隙句生成的自监督预训练目标，来改进生成摘要的微调性能。
本repo参考开源论坛对于中文版本的实现，以mT5为基础架构和初始权重，通过类似PEGASUS的方式进行预训练。

![image](https://github.com/jackie930/t5-pegasus-textsummary/blob/main/1.png)

## 预训练任务

预训练任务模仿了PEGASUS的摘要式预训练。具体来说，假设一个文档有n个句子，我们从中挑出大约n/4个句子（可以不连续），使得这n/4个句子拼起来的文本，跟剩下的3n/4个句子拼起来的文本，最长公共子序列尽可能长，然后我们将3n/4个句子拼起来的文本视为原文，n/4个句子拼起来的文本视为摘要，通过这样的方式构成一个“(原文, 摘要)”的伪摘要数据对。

## 模型下载
备注： 下面提供的下载链接需要登陆作者公司的vpn才可以使用
* [chinese_t5_pegasus_base.zip](https://amazon.awsapps.com/workdocs/index.html#/folder/de75a58625bdc3a6ae7f659c252924917c0f956ee33dc6a5f0a6cc2283a63a72) 目前开源的T5 PEGASUS是base版，总参数量为2.75亿，训练时最大长度为512，batch_size为96，学习率为10-4，使用6张3090训练了100万步，训练时间约13天，数据是30多G的精处理通用语料，训练acc约47%，训练loss约2.97。模型使用bert4keras进行编写、训练和测试。

运行环境：tensorflow 1.15 + keras 2.3.1 + bert4keras 0.10.0
* [best_model.weights](https://amazon.awsapps.com/workdocs/index.html#/folder/de75a58625bdc3a6ae7f659c252924917c0f956ee33dc6a5f0a6cc2283a63a72) 使用100条短文本finetune过后的模型

## quick start

### train

~~~shell script
source activate tensorflow_p36
pip install tensorflow==1.15 keras==2.3.1 bert4keras==0.10.0 jieba tqdm rouge
python finetune.py
~~~
训练结束会产生一个keras结构的模型文件 - best_model.weights

### 预测
下载模型文件，目录结构为

chinese_t5_pegasus_base/ 
best_model.weights

~~~shell script
python test.py
~~~


### 评估

运行 `python evaluatiion.py`

得到结果 `{'rouge-1': 0.885041123153444, 'rouge-2': 0.8795828353099052, 'rouge-l': 0.9046418758557804, 'bleu': 0.8239310846742561}`

### 部署


run locally
~~~
#make sure you got trained models 
sh build_and_push.sh
~~~

run on endpoint

```shell script
endpoint_ecr_image="251885400447.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pegasus"
python create_endpoint.py \
--endpoint_ecr_image_path ${endpoint_ecr_image} \
--endpoint_name 'pegasus' \
--instance_type "ml.g4dn.2xlarge"
```

在部署结束后，看到SageMaker控制台生成了对应的endpoint,可以使用如下客户端代码测试调用
```python
from boto3.session import Session
import json
txt = "来源|零壹财经作者|任俊东12月1日，国家互联网信息办公室发布关于《常见类型移动互联网应用程序（App）必要个人信息范围》公开征求意见的通知。此次《意见稿》规定了支付、借贷、银行等38类常见类型App必要个人信息范围，明确App必要个人信息界限，不得因非必要信息拒绝用户安装使用。零壹财经自今年3月起开展了手机App评测工作，通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。上月，全国App个人信息保护监管会召开。会上阿里、腾讯、字节等互联网巨头遭监管点名批评：在App个人信息保护工作方面，存在思想漠视、侥幸心理、技术对抗三类问题。1.对38类App必要个人信息范围向社会征求意见针对此次《意见稿》，国家网信办表示，近年来App广泛应用在促进经济社会发展、服务民生等方面发挥了重要作用。同时，App超范围收集、强制收集用户个人信息普遍存在，用户拒绝同意就无法安装使用。为落实《中华人民共和国网络安全法》关于个人信息收集合法、正当、必要的原则，规范App个人信息收集行为，因而明确常见App收集必要个人信息范围。意见反馈时间截止到2020年12月16日。2.12类App无须个人信息，即可使用基本功能服务根据《意见稿》，国家网信办拟规定网络直播、在线影音、短视频、新闻资讯、运动健身、浏览器、输入法、安全管理、电子图书、拍摄美化、应用商店、实用工具类共12类App无须个人信息，即可使用基本功能服务。3.零壹App评测：9成以上App存在违规收集信息问题为规范收集APP信息收集和使用、加强个人信息保护，切实维护收集APP消费者合法权益，并依据相关监管政策法规，零壹财经App评测中心于2020年3月2日启动App评测专项工作。中心相关评测工作得到了App消费者、监管部门、相关企业、行业从业者等多方的广泛关注和支持。通过对金融、购物、视频等10大类300多款App评测发现，9成以上APP都存在违规收集信息问题，其中违反必要原则，收集与其业务无关的个人信息、用户拒绝同意就无法安装使用等问题最为严重。4.阿里、腾讯、字节等遭监管点名批评，App个人信息保护进入新的发展阶段11月27日，全国App个人信息保护监管会在北京召开，工信部召集国内互联网行业的头部企业，总结过去半年来App个人信息保护专项整治行动的成果，部署下一阶段整治行动。工信部信息通信管理局副局长鲁春从在会上表示，工信部针对大企业的App进行了全覆盖检测，对阿里巴巴的40余款、字节跳动30余款，腾讯30余款、百度20余款、网易10余款、小米10余款用户下载量大、使用率高的App进行了重点检测，发现存在思想漠视、侥幸心理、技术对抗三类问题。互联网个人信息数据野蛮生长时代已成过去，APP个人信息保护正在迎来新的发展阶段。切实维护用户合法权益，严厉惩处互联网企业违法违规行为是今后互联网监管的常态。企业只有从思想上重视、行动上遵守，把用户的利益作为企业的核心关切，才能持续发展。添加作者微信：daodao0312，可获取《常见类型移动互联网应用程序（App）必要个人信息范围（征求意见稿）》，或您有App评测需求请联系作者。"
data={"data": txt}
session = Session()
    
runtime = session.client("runtime.sagemaker")
response = runtime.invoke_endpoint(
    EndpointName='pegasus',
    ContentType="application/json",
    Body=json.dumps(data),
)

result = json.loads(response["Body"].read())
print (result)
```

## 模型转换和加速
### 环境配置

~~~shell script
# cuda/cudnn/onnxruntime的版本需要对应才能使用gpu进行onnxrntime推理，详见https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
# cuda 10.2
# cudnn 8.0.3
pip install onnxruntime==1.5.1 onnxruntime-gpu==1.5.1 onnxruntime_tools onnxmltools sympy tf2onnx
~~~

### 模型转换

* 支持tensorflow模型转换成onnx模型
* 支持optmization操作和转换成float16模型，这是针对transformer-gpu结构的optmization工具，支持部分模型和部分框架，详见https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/transformers
* 动态量化

~~~shell script
python tf_to_onnx.py
~~~

### 模型测试

在进行测试之前，需要将 bert4keras 库中的 snippets.py 做修改，删除 `prediction = predict(self, inputs, output_ids, states)` 所在行，并再增加三行代码：

~~~shell script
-  prediction = predict(self, inputs, output_ids, states)
+  prediction = predict(self, inputs, output_ids, states)[0]
+  if prediction.shape[0] > 1:
+      prediction = np.expand_dims(prediction[-1], 0)
~~~

或者在终端中直接执行以下代码以安装 [tzq0301](https://github.com/tzq0301) 修改后的 [bert4keras](https://github.com/tzq0301/bert4keras)：

~~~shell script
pip install git+https://github.com/tzq0301/bert4keras.git@tzq
# pip install bert4keras
~~~

可以修改代码中的 providers 以支持 GPU 或 CPU 测试：

* GPU 测试时，`providers=['CUDAExecutionProvider']`
* CPU 测试时，`providers=['CPUExecutionProvider']`

~~~shell script
python test_onnx.py
~~~

## reference

https://github.com/ZhuiyiTechnology/t5-pegasus 

https://github.com/google-research/pegasus

https://github.com/bojone/bert4keras
