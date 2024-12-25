# Dockerfile for development
FROM python:3.10

WORKDIR /src

COPY . .
RUN pip install --no-cache-dir -r  requirements.txt

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8001
#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

#CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]