FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-api.txt constraints.txt ./
RUN pip install -r requirements-api.txt -c constraints.txt
COPY . .
RUN useradd --create-home --uid 10001 appuser && mkdir -p data models logs experiments && chown -R appuser /app
USER appuser
EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
