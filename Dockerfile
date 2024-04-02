FROM python:3.10.6-slim-buster

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Add ffprobe to PATH
ENV PATH="/usr/bin:${PATH}"

WORKDIR /prod

COPY musichook/flask-app flask-app

RUN pip install -r flask-app/requirements.txt

CMD gunicorn -b 0.0.0.0:$PORT flask-app.app:app
