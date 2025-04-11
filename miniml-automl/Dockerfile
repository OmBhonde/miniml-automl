FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn scikit-learn pandas
CMD ["python", "main.py"]