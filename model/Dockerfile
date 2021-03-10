FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY model.py ./

COPY clf_model clf_model
COPY tfidf tfidf

COPY test_pipeline.parquet test_pipeline.parquet

CMD ["python3", "model.py"]