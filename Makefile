go :
	python musichook/main.py

test_spotify_api :
	python tests/test_spotify_api.py

test_audio_processing :
	python tests/test_audio_processing.py

test_self_chorus_detection :
	python tests/test_self_chorus_detection.py

build_dev:
	docker build -t musichook:dev .

build_prod:
	docker build --platform linux/amd64 -t europe-west1-docker.pkg.dev/spatial-earth-384009/taxifare/musichook:prod .

run_dev:
	docker run -e PORT=8000 -p 8000:8000 musichook:dev

run_dev_it:
	docker run -it -e PORT=8000 -p 8000:8000 musichook:dev bash

push_prod:
	docker push europe-west1-docker.pkg.dev/spatial-earth-384009/taxifare/musichook:prod

deploy_prod:
	gcloud run deploy --image europe-west1-docker.pkg.dev/spatial-earth-384009/taxifare/musichook:prod --memory 2Gi --region europe-west1
