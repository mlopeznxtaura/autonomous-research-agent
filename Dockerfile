FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3 python3-pip git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY . .

EXPOSE 7860 8000
ENTRYPOINT ["python3", "main.py"]
CMD ["--mode", "demo", "--topic", "large language model reasoning capabilities"]
