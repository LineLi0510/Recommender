# Dockerfile for development
FROM python:3.10

WORKDIR /src

COPY . .
RUN pip install --no-cache-dir -r  requirements.txt

#CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]