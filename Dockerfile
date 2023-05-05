# Parent image
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Run script when the container launches
ENTRYPOINT ["flask", "run"]
