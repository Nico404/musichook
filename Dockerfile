FROM python:3.10.6-slim-buster

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg
ENV PATH="/usr/bin:${PATH}"

WORKDIR /prod

COPY musichook/flask-app flask-app

RUN pip install -r flask-app/requirements.txt

CMD gunicorn -b 0.0.0.0:$PORT --timeout 180 flask-app.app:app
