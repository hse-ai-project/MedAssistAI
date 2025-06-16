FROM python:3.8-slim-buster

WORKDIR /app/frontend
COPY . .

RUN mkdir -p logs data

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.enableCORS", "false"]