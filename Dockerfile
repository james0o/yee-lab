FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
COPY . .
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD [".venv/bin/uvicorn", "yeelab.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
