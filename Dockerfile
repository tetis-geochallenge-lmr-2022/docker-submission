FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install numpy pandas torch scikit-learn transformers
CMD ["python3", "tetis-geochallenge-submit-1.py"]
