FROM python:3.10-slim

WORKDIR /app

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
