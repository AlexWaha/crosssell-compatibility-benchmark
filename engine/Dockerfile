FROM python:3.12-slim

WORKDIR /src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY src/requirements.txt .
RUN pip install -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]
