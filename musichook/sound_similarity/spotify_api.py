import requests
import base64
import os


class SpotifyAPI:
    def __init__(self, client_id: str, client_secret: str) -> None:
        """
        Initialize SpotifyAPI object.

        Args:
            client_id (str): Spotify API client ID.
            client_secret (str): Spotify API client secret.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    def _get_access_token(self) -> str:
        """
        Retrieve access token from Spotify API.

        Returns:
            str: Access token.
        """
        if self.access_token:
            return self.access_token

        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        token_url = "https://accounts.spotify.com/api/token"
        headers = {"Authorization": f"Basic {encoded_credentials}"}
        data = {"grant_type": "client_credentials"}
        response = requests.post(token_url, headers=headers, data=data)

        if response.status_code == 200:
            token_info = response.json()
            self.access_token = token_info["access_token"]
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.text}")

    def search_track_id(self, song_name: str) -> str:
        """
        Search for a track ID given a song name.

        Args:
            song_name (str): Name of the song.

        Returns:
            str: Track ID.
        """
        access_token = self._get_access_token()
        search_url = "https://api.spotify.com/v1/search"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"q": song_name, "type": "track", "limit": 1}
        response = requests.get(search_url, headers=headers, params=params)

        if response.status_code == 200:
            track_info = response.json()
            if track_info["tracks"]["items"]:
                return track_info["tracks"]["items"][0]["id"]
            else:
                raise Exception("No track found for the given song name.")
        else:
            raise Exception(f"Failed to search track ID: {response.text}")

    def get_audio_link(self, track_id: str) -> str:
        """
        Get the audio link (preview URL) for a given track ID.

        Args:
            track_id (str): Track ID.

        Returns:
            str: Audio link (preview URL).
        """
        access_token = self._get_access_token()
        track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(track_url, headers=headers)

        if response.status_code == 200:
            track_info = response.json()
            return track_info["preview_url"]
        else:
            raise Exception(f"Failed to get audio link: {response.text}")

    def download_audio(
        self, audio_link: str, destination_path: str, destination_filename: str
    ) -> None:
        """
        Download audio from the given audio link and save it to the destination path.

        Args:
            audio_link (str): Audio link (URL).
            destination_path (str): Destination directory to save the audio file.
            destination_filename (str): Destination filename to save the audio file.
        """
        response = requests.get(audio_link)

        if response.status_code == 200:
            # Save the audio content to the destination file
            with open(
                os.path.join(destination_path, destination_filename), "wb"
            ) as file:
                file.write(response.content)
            print(
                f"Audio file saved to: {os.path.join(destination_path, destination_filename)}"
            )
        else:
            raise Exception(f"Failed to download audio: {response.text}")
