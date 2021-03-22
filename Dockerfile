FROM amd64/python:3.8-slim

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x /app/main.py
ENTRYPOINT ["/app/main.py"]