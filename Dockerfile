FROM python:3.10-slim
RUN pip install uv && apt-get update && apt-get install -y git
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV DIFFUSERS_CACHE=/models
ENV APP_MODE=text2image
EXPOSE 8080
CMD ["uv", "run", "python", "-m", "app.main"]
