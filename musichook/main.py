import os
import pandas as pd
import matplotlib.pyplot as plt
from params import PATHS, SECRETS
from src.spotify_api import SpotifyAPI
from pydub import AudioSegment
from src.audio_processing import (
    cut_audio_into_sliding_intervals,
    convert_to_chromagram,
    display_chromagram,
    compare_images,
)


if __name__ == "__main__":

    # load spotify preview audio
    preview_path = os.path.join(PATHS["SPOTIFY_FOLDER"], "Dani California.mp3")
    # open spotify preview audio and convert to audio segment
    preview_audio = AudioSegment.from_file(preview_path)
    preview_length = len(preview_audio) / 1000  # seconds
    preview_chromagram = convert_to_chromagram(preview_audio)
    print("preview_chromagram.shape", preview_chromagram.shape)
    print("Spotify preview chromagram created")

    # full song path
    song_path = os.path.join(
        PATHS["MUSIC_FOLDER"],
        "058 - Dani California By Red Hot Chili Peppers 320 - [ambush22].mp3",
    )

    # Cut audio into sliding intervals
    segments = cut_audio_into_sliding_intervals(
        song_path, interval_length=preview_length
    )
    print(len(segments), "segments extracted of duration", preview_length)

    # # Convert first segment into a chromagram
    # chromagram = convert_to_chromagram(segments[1])
    # print(f"chromagram.shape", chromagram.shape)

    # # Display chromagram
    # display_chromagram(chromagram, "Chromagram of first segment")

    # for each segment, get the chromagram and build a new dict with segment number as key and chromagram as value
    chromagrams = {}
    for segment_number, segment in segments.items():
        chromagram = convert_to_chromagram(segment)
        print(f"Segment {segment_number}: chromagram.shape", chromagram.shape)
        chromagrams[segment_number] = chromagram

    print(len(chromagrams), "chromagrams created")

    # Compare chromagrams
    chronogram_comparaison_results = {}
    for segment_number, chromagram in chromagrams.items():
        # Compare chromagram with Spotify preview chromagram
        ssim, mse = compare_images(chromagram, preview_chromagram)
        chronogram_comparaison_results[segment_number] = {"ssim": ssim, "mse": mse}

    # # plot results of comparison
    # df = pd.DataFrame(chronogram_comparaison_results).T
    # df.plot(y=["ssim", "mse"], title="Comparison of chromagrams")
    # plt.show()

    # get segment with maximum ssim
    max_segment_number, max_segment_value = int(), 0
    for segment_number, results in chronogram_comparaison_results.items():
        if results["ssim"] > max_segment_value:
            max_segment_value = results["ssim"]
            max_segment_number = segment_number

    print(f"segment {max_segment_number} has ssim value of {max_segment_value}")
    # save audio segment 175
    segments[max_segment_number].export(
        os.path.join(PATHS["STAGING_FOLDER"], f"segment_{max_segment_number}.mp3"),
        format="mp3",
    )
    print(f"""segment_{max_segment_number}.mp3 saved in {PATHS["STAGING_FOLDER"]}""")
