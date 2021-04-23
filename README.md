# t5-pegasus-textsummary
使用谷歌2020pegasus模型进行中文文档摘要

谷歌于去年年底发布了一个精简型的机器语义分析项目：飞马(PEGASUS)：预先机器学习及训练后的自动文章摘要项目。近期这个项目迎来的新的版本，这个小型项目可以非常精准的自动提取出文章中的摘要，并且只用一千个训练模型就可以生成媲美人类的摘要内容。
利用提取的间隙句进行摘要概括的预训练模型（Pre-training with Extracted Gap-sentences for Abstractive Summarization）。就是设计一种间隙句生成的自监督预训练目标，来改进生成摘要的微调性能。
本repo参考开源论坛对于中文版本的实现，以mT5为基础架构和初始权重，通过类似PEGASUS的方式进行预训练。

![image](https://github.com/xxx/xxx/blob/master/xxx/xxx.png)

##预训练任务

预训练任务模仿了PEGASUS的摘要式预训练。具体来说，假设一个文档有n个句子，我们从中挑出大约n/4个句子（可以不连续），使得这n/4个句子拼起来的文本，跟剩下的3n/4个句子拼起来的文本，最长公共子序列尽可能长，然后我们将3n/4个句子拼起来的文本视为原文，n/4个句子拼起来的文本视为摘要，通过这样的方式构成一个“(原文, 摘要)”的伪摘要数据对。

## 模型下载
备注： 下面提供的下载链接需要登陆作者公司的vpn才可以使用
* [chinese_t5_pegasus_base.zip](https://amazon.awsapps.com/workdocs/index.html#/folder/de75a58625bdc3a6ae7f659c252924917c0f956ee33dc6a5f0a6cc2283a63a72) 目前开源的T5 PEGASUS是base版，总参数量为2.75亿，训练时最大长度为512，batch_size为96，学习率为10-4，使用6张3090训练了100万步，训练时间约13天，数据是30多G的精处理通用语料，训练acc约47%，训练loss约2.97。模型使用bert4keras进行编写、训练和测试。

运行环境：tensorflow 1.15 + keras 2.3.1 + bert4keras 0.10.0
* [best_model.weights](https://amazon.awsapps.com/workdocs/index.html#/folder/de75a58625bdc3a6ae7f659c252924917c0f956ee33dc6a5f0a6cc2283a63a72) 使用100条短文本finetune过后的模型

## quick start

### train

~~~shell script
pip install tensorflow==1.15 keras==2.3.1 bert4keras==0.10.0
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

### 部署

todo

## reference

https://github.com/ZhuiyiTechnology/t5-pegasus
https://github.com/google-research/pegasus

