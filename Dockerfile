FROM python:3.8-bullseye

RUN python3 -m pip install -U pip

WORKDIR /app
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt
ADD . .

CMD ["python3", "main.py"]