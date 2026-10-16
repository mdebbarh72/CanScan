FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit-app.py .

RUN useradd --create-home appuser
USER appuser

# Inside docker-compose the API is reachable by its service name
ENV API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "streamlit-app.py", "--server.address=0.0.0.0"]