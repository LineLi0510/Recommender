# Verwende ein Python-Image mit arm64-Unterstützung
FROM --platform=linux/arm64 python:3.10-slim

WORKDIR /src

# Kopiere den Quellcode in das Arbeitsverzeichnis
COPY . .

# Installiere Basis-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Installiere vorab numpy und scikit-surprise
RUN pip install --no-cache-dir numpy==1.21.6 scikit-surprise

# Kopiere die übrigen Abhängigkeiten
COPY requirements.txt .

# Installiere TensorFlow für die ARM64-Architektur
RUN pip install --no-cache-dir tensorflow

# Installiere die restlichen Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Setze die Umgebungsvariable für unbuffered Output
ENV PYTHONUNBUFFERED=1

# Exponiere die Ports für die API
EXPOSE 8000 8001

#CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

#CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]