# Dockerfile
FROM python:3.12-slim

# Prevent Python from writing pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps:
# - build-essential + pkg-config: building wheels when needed
# - default-libmysqlclient-dev: mysqlclient headers
# - libmariadb-dev-compat sometimes helps on Debian-based images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    libmariadb-dev-compat \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the project
COPY . /app

# (Optional) collect static at build time if you want
# You can also do this at runtime in your entrypoint.
# RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Change "myhealth" below to your Django project module if different.
# Run DB migrations at container start is common; add it if you want.
CMD ["sh", "-c", "python manage.py migrate && gunicorn myhealth.wsgi:application --bind 0.0.0.0:8000"]
