FROM python:3.10

# Arbeitsverzeichnis setzen
WORKDIR /app

# Installiere nur die benötigten Python-Pakete
RUN pip install --no-cache-dir optuna mysql-connector-python

# Füge das Pre-Boot-Skript hinzu
COPY init_study.py /app/init_study.py

# Standard-Befehl: Skript ausführen
CMD ["python", "/app/init_study.py"]