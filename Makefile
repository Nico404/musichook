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
	docker build --platform linux/amd64 -t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(GCP_REPO)/musichook:prod .

run_local:
	gunicorn -w 4 --timeout 60  musichook.flask-app.app:app

run_dev:
	docker run -e PORT=8000 -p 8000:8000 musichook:dev

run_dev_it:
	docker run -it -e PORT=8000 -p 8000:8000 musichook:dev bash

push_prod:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(GCP_REPO)/musichook:prod

deploy_prod:
	gcloud run deploy --image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(GCP_REPO)/musichook:prod --memory 8Gi --region $(GCP_REGION)

ship_fast:
	make build_prod
	make push_prod
	make deploy_prod
