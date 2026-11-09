# Hugging Face Space (Docker SDK, free CPU). Built from the staged folder produced by scripts/deploy_hf.py.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 TORCH_THREADS=2 OMP_NUM_THREADS=2
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY --chown=user app ./app
COPY --chown=user src ./src
COPY --chown=user model ./model
USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
