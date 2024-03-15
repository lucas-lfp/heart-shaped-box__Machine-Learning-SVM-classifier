# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9.5
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the source code
COPY . .

# Use the non-privileged user to run the application
USER appuser

# Specify the Flask app entry point
ENV FLASK_APP=main.py

# Expose the application port
EXPOSE 5000

# Uncomment the following line to use Gunicorn instead of the Flask development server
# CMD ["gunicorn", "myapp:app", "--bind", "0.0.0.0:5000", "--workers", "3"]

# Keep using the Flask development server
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
