import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from musichook.params import PATHS, SECRETS
from musichook.src import audio_processing
from musichook.utils import plots

from pydub import AudioSegment
from IPython.display import Audio, display


def test_self_chorus_detection(song_name, window_size):
    print("---------------------------------")
    print("Testing Self Chorus Detection...")
    print("---------------------------------")

    print("cutting song into segments")
    quarter_second_segments = audio_processing.cut_song_into_segments(
        os.path.join(PATHS["MUSIC_FOLDER"], song_name), 250
    )  # 250ms segments
    print("building feature vectors")
    feature_vectors = audio_processing.build_feature_vector(
        quarter_second_segments,
        include_chromagram=True,
        include_mfcc=True,
        include_melspectrogram=True,
    )
    print("compute similarity matrix")
    pearson_similarity_matrix = audio_processing.compute_similarity_matrix(
        feature_vectors, "pearson_correlation"
    )
    print("compute time lag surface")
    time_lag_surface = audio_processing.correlation_filter(
        pearson_similarity_matrix, 20
    )
    print("select thumbnail")
    try:
        start_time, window_size_out = audio_processing.select_thumbnail(
            time_lag_surface
        )
        start_time_formatted, end_time_formatted = audio_processing.format_time_tuple(
            start_time, window_size_out
        )
    except:
        print("No chorus detected")

    print("----------------------------------")
    print("Testing Self Chorus Detection DONE")
    print("----------------------------------")

    return start_time_formatted, end_time_formatted


if __name__ == "__main__":
    print(
        test_self_chorus_detection(
            "065 - Sexual Healing By Marvin Gaye 320 - [kthor].mp3", 20
        )
    )
