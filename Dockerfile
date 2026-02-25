FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

RUN mkdir -p /app/config
ENV CONFIG_DIR=/app/config

CMD ["python", "bot.py"]
