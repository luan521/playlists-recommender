FROM python:3.10-slim-bullseye

WORKDIR /code

COPY ./requirements.txt ./
COPY ./app.py ./
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py", "0.0.0.0", "8000", "/app/model"]