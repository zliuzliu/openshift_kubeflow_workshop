#FROM tensorflow/tensorflow:2.0.1-gpu
FROM tensorflow/tensorflow:1.12.0-gpu
MAINTAINER "Zheng Liu <zhengli@redhat.com>"

RUN mkdir /home/tensorflow

# show python logs as they occur
ENV PYTHONUNBUFFERED=0

# run MNIST.py
ADD model.py /home/tensorflow
WORKDIR /home/tensorflow
ENTRYPOINT /usr/bin/python /home/tensorflow/model.py
#ENTRYPOINT /usr/local/bin/python /home/tensorflow/model.py
