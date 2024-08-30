
# the official Python image. slim is a smaller version of the base Python image
FROM python:3.9-slim  

WORKDIR /deploy

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

RUN apt-get clean
RUN apt-get autoclean -y
RUN apt-get autoremove -y
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY train.py .
COPY cnn_model.pt .
COPY data/fashion_mnist_train.pt data/
COPY data/fashion_mnist_test.pt data/

# executable
ENTRYPOINT ["python", "inference.py"]

# default arguments
CMD ["--test_path", "data/fashion_mnist_test.pt"]