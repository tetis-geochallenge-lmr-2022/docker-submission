FROM python:3.10
ADD . /tetis/
ADD ./geoai/ /geoai/
WORKDIR /tetis
RUN pip install numpy pandas torch scikit-learn transformers
CMD ["python3", "tetis-geochallenge-submit-1.py"]

