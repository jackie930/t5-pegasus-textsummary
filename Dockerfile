# ARG REGISTRY_URI
# FROM ${REGISTRY_URI}/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04
FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN mkdir -p /opt/ml/model

# COPY package/ /opt/ml/code/package/

# COPY serve.py /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install networkx==2.3 flask gevent gunicorn boto3 opencv-python==4.4.0.40 -i https://opentuna.cn/pypi/web/simple
## install dependencies
RUN pip install keras==2.3.1 bert4keras==0.10.0 jieba tqdm rouge

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
# COPY * /opt/program/
# COPY ./pretrained_model/* /opt/program/
COPY ./ /opt/program/
WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]

