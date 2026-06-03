FROM python:3.12-slim

WORKDIR /app

# Abhaengigkeiten installieren
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Datensatz, Code und App kopieren
COPY data/ ./data/
COPY train_models.py .
COPY app.py .
COPY pages/ ./pages/

# Modelle beim Build-Schritt trainieren und speichern
RUN python train_models.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
