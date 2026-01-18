FROM python:3.10-slim

WORKDIR /app

# install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# expose port
EXPOSE 8000

# run the api
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
