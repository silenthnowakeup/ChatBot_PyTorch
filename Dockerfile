FROM python:3.9

RUN apt-get update && apt-get install -y python3-tk

RUN pip install torch nltk

RUN python -m nltk.downloader punkt

WORKDIR /app
COPY . /app

CMD ["python", "app.py"]
git