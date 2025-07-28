FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY main.py utils.py requirements.txt ./

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV NO_PROXY="*"

ENTRYPOINT ["python", "main.py"]
