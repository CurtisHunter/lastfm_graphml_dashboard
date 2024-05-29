FROM python:3.8.8

WORKDIR /app

COPY LastFMmodel.joblib model_inferencing.py columns.csv /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "model_inferencing.py"]
