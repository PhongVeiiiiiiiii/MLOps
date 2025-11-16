FROM python:3.11-slim

WORKDIR /app

# Copy toàn bộ code + model vào container
COPY . /app

# Cài đặt dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Chạy app
EXPOSE 5000
CMD ["python", "src/app.py"]
