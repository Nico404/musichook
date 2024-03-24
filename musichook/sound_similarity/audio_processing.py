from typing import Tuple
import librosa
import numpy as np
from pydub import AudioSegment
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def cut_audio_into_sliding_intervals(file_path: str, interval_length: float) -> dict:
    """
    Extract 30 seconds segments from a sound file and store them as AudioSegment objects in a dictionary.

    Args:
        file_path (str): path to the sound file

    Returns:
        dict: Dictionary with keys as segment names and values as AudioSegment objects
    """
    song = AudioSegment.from_file(file_path)
    duration = len(song) / 1000  # seconds
    extracted_segments = {}

    print(song, duration)

    for i in range(
        int(duration) - int(interval_length)
    ):  # ensure last segment is within file
        start_time = i * 1000  # ms
        end_time = (i + interval_length) * 1000  # ms
        segment = song[start_time:end_time]
        extracted_segments[int(i)] = segment

    return extracted_segments


def convert_to_chromagram(segment: AudioSegment, hop_length: int = 2048) -> np.ndarray:
    """
    Convert an audio segment into a chromagram.

    Args:
        segment (AudioSegment): Audio segment to convert.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Chromagram of the audio segment.
    """
    # Convert AudioSegment to numpy array
    y = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sr = segment.frame_rate

    # Compute chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    return chromagram


def display_chromagram(chromagram: np.ndarray, segment_name: str) -> None:
    """
    Display the chromagram as an image

    Args:
        chromagram (np.array): chromagram of the sound file
    """
    plt.figure(figsize=(20, 8))
    librosa.display.specshow(chromagram, y_axis="chroma", x_axis="time")
    plt.colorbar()
    plt.title(f"Chromagram of {segment_name}")
    plt.tight_layout()
    plt.show()


def compare_images(image1: np.ndarray, image2: np.ndarray) -> Tuple[float, float]:
    """
    Compare two images using SSIM and MSE.

    Parameters:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        float: SSIM score.
        float: MSE score.
    """
    # Compute SSIM score
    ssim_score = ssim(image1, image2, data_range=image1.max() - image1.min())

    # Compute MSE score
    mse_score = mean_squared_error(image1, image2)

    return ssim_score, mse_score
