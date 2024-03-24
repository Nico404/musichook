import os

ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

PATHS = {
    'ROOT_PROJECT_PATH': ROOT_PROJECT_PATH,
    'SPOTIFY_FOLDER': os.path.join(ROOT_PROJECT_PATH, 'data/spotify')
}

SECRETS = {
    'CLIENT_ID': os.environ.get("SPOTIFY_CLIENT_ID"),
    'CLIENT_SECRET': os.environ.get("SPOTIFY_CLIENT_SECRET")

}
