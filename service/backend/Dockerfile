FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 libomp-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ARCH=$(dpkg-architecture -qDEB_HOST_MULTIARCH) && \
    ln -s /usr/lib/${ARCH}/libgomp.so.1 /usr/lib/libgomp.so.1
ENV LD_PRELOAD=/usr/lib/libgomp.so.1

WORKDIR /app/backend
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
