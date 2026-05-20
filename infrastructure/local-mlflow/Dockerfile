FROM python:3.10-slim

RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    boto3 \
    psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", "--default-artifact-root", "s3://mlflow/", "--serve-artifacts", "--artifacts-destination", "s3://mlflow/", "--gunicorn-opts", "--workers 2 --timeout 120"]
