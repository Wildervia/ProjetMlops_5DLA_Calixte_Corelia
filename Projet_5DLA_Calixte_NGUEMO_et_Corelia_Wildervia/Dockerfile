FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir setuptools==65.5.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY config.yaml .
COPY data/creditcard.csv .

CMD ["python", "train.py", "--config", "config.yaml"]