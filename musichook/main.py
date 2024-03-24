from params import PATHS, SECRETS
from sound_similarity.spotify_api import SpotifyAPI


def test_spotify_api():
    # Get Spotify API credentials
    spotify = SpotifyAPI(SECRETS['CLIENT_ID'], SECRETS['CLIENT_SECRET'])

    # Search for a track ID
    track_id = spotify.search_track_id("Dani California")
    print(f"Dani California track_id is: {track_id}")

    # Search for song preview link
    preview_link = spotify.get_audio_link(track_id)
    print(f"Dani California preview link is: {preview_link}")

    # download the song to spotify folder
    spotify.download_audio(preview_link, PATHS['SPOTIFY_FOLDER'], "Dani California.mp3")



if __name__ == '__main__':
    print('-----------------------')
    print('Testing Spotify API...')
    print('-----------------------')
    test_spotify_api()
    print('-----------------------')
    print('Testing Spotify DONE...')
    print('-----------------------')
