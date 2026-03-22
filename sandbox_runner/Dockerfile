FROM python:3.11-slim

# Disable networking inside container
# (we will also enforce this at runtime)
RUN useradd -m sandbox

WORKDIR /app
USER sandbox

COPY runner.py /app/runner.py

ENTRYPOINT ["python", "runner.py"]
