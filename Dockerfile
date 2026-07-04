# GridPulse — production image for Fly.io.
# uv installs from the lockfile into a system env; gunicorn (gthread, 1 worker)
# serves the Fast Dash app + its Flask-SocketIO sidecar in threading mode.
FROM python:3.11-slim

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first (cached layer) from the lockfile, without the project.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra prod

# Then the source, and install the project itself.
COPY gridpulse ./gridpulse
COPY README.md ./
RUN uv sync --frozen --no-dev --extra prod

EXPOSE 8080

# Single worker (in-process chat history + socket.io session affinity);
# gthread handles concurrent Dash callbacks and long-poll connections.
CMD ["gunicorn", "gridpulse.app:server", \
     "--worker-class", "gthread", "--workers", "1", "--threads", "8", \
     "--bind", "0.0.0.0:8080", "--timeout", "120", "--access-logfile", "-"]
