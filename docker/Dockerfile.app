FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv

# Install only the dependencies needed for the client application
# --frozen: Use exact versions from the lock file
# Install full dependency set so shared modules (core, etc.) are available
RUN uv sync --frozen

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/core/ ./core/
COPY src/streamlit_app.py .
COPY src/pages/ ./pages/

CMD ["streamlit", "run", "streamlit_app.py"]
