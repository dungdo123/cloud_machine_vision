FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mvas_runtime/ ./mvas_runtime/

EXPOSE 8000
CMD ["uvicorn", "mvas_runtime.server:app", "--host", "0.0.0.0", "--port", "8000"]