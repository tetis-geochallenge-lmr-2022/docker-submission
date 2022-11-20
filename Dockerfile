FROM python:3.10
COPY . .
ADD ./geoai/ /geoai/
WORKDIR .
RUN pip install numpy pandas torch scikit-learn transformers
CMD ["python3", "tetis-geochallenge-submit-1.py"]
