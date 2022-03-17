FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

SHELL ["/bin/bash", "-c"]

ENV APP_DIR=/cmc

COPY requirements.txt $APP_DIR/requirements.txt

WORKDIR $APP_DIR

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
