FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Copy code and models
COPY main.py utils.py requirements.txt ./
COPY models/ ./models/
# If you want to copy sample PDFs, use: COPY Collection* ./

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ENV NO_PROXY="*"

ENTRYPOINT ["python", "main.py"]
