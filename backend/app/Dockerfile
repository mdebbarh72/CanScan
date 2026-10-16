FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY app/ ./app/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=/models/brain_tumor_cnn.h5 \
    CLASS_NAMES_PATH=/models/class_names.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]