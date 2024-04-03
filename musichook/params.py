import os

ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

PATHS = {
    "ROOT_PROJECT_PATH": ROOT_PROJECT_PATH,
    "SPOTIFY_FOLDER": os.path.join(ROOT_PROJECT_PATH, "data/spotify"),
    "MUSIC_FOLDER": os.path.join(ROOT_PROJECT_PATH, "data/music"),
    "STAGING_FOLDER": os.path.join(ROOT_PROJECT_PATH, "data/staging"),
    "APP_ROOT": os.path.join(ROOT_PROJECT_PATH, "musichook/musicflask-app"),
}

SECRETS = {
    "CLIENT_ID": os.environ.get("SPOTIFY_CLIENT_ID"),
    "CLIENT_SECRET": os.environ.get("SPOTIFY_CLIENT_SECRET"),
}
