from typing import Tuple, Dict
import librosa
import numpy as np
from pydub import AudioSegment
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine as cosine_similarity



def cut_chunk_into_audio(file_path: str, start_time_ms: int, end_time_ms: int) -> AudioSegment:
    """
    Cut a chunk of audio from a sound file.

    Args:
        file_path (str): Path to the sound file.
        start_time (float): Start time of the chunk in seconds.
        end_time (float): End time of the chunk in seconds.

    Returns:
        AudioSegment: The extracted chunk of audio.
    """
    song = AudioSegment.from_file(file_path)
    print(start_time_ms, end_time_ms)
    chunk = song[start_time_ms:end_time_ms]
    return chunk


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

    for i in range(int(duration) - int(interval_length)):
        start_time = i * 1000  # ms
        end_time = (i + interval_length) * 1000  # ms
        segment = song[start_time:end_time]
        extracted_segments[int(i)] = segment

    return extracted_segments


def cut_song_into_segments(
    file_path: str, segment_length_ms: int
) -> Dict[int, AudioSegment]:
    """
    Cut a sound file into segments of specified length (in milliseconds) and store them in a dictionary.

    Args:
        file_path (str): Path to the sound file.
        segment_length_ms (int): Length of each segment in milliseconds.

    Returns:
        dict[int, AudioSegment]: A dictionary where keys are segment indices and values are AudioSegments.
    """
    song = AudioSegment.from_file(file_path)
    duration = len(song)
    extracted_segments = {}

    for i in range(0, duration - segment_length_ms + 1, segment_length_ms):
        segment = song[i : i + segment_length_ms]
        extracted_segments[i // segment_length_ms] = segment

    return extracted_segments


def get_segment_time(index: int, interval_length: float) -> tuple:
    """
    Get the start and end time of a segment given its index and interval length.

    Args:
        index (int): Index of the segment.
        interval_length (float): Length of each segment in seconds.

    Returns:
        tuple: A tuple containing the start and end time of the segment formatted as 'mm:ss'.
    """
    start_time = index
    end_time = index + interval_length

    start_minutes, start_seconds = divmod(start_time, 60)
    end_minutes, end_seconds = divmod(int(end_time), 60)

    start_time_formatted = f"{start_minutes:02d}:{start_seconds:02d}"
    end_time_formatted = f"{end_minutes:02d}:{end_seconds:02d}"

    return start_time_formatted, end_time_formatted


# 2048 by default before
def convert_to_chromagram(segment: AudioSegment, hop_length: int = 512) -> np.ndarray:
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


def convert_to_melspectrogram(
    segment: AudioSegment, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 512
) -> np.ndarray:
    """
    Convert an audio segment into a mel spectrogram.

    Args:
        segment (AudioSegment): Audio segment to convert.
        n_mels (int): Number of mel bands to generate.
        n_fft (int): Number of samples to use for FFT.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Mel spectrogram of the audio segment.
    """
    # Convert AudioSegment to numpy array
    y = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sr = segment.frame_rate

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return mel_spectrogram


def convert_to_mfcc(
    segment: AudioSegment, n_mfcc: int = 20, n_fft: int = 1024, hop_length: int = 512
) -> np.ndarray:
    """
    Convert an audio segment into MFCCs.

    Args:
        segment (AudioSegment): Audio segment to convert.
        n_mfcc (int): Number of MFCCs to generate.
        n_fft (int): Number of samples to use for FFT.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: MFCCs of the audio segment.
    """
    # Convert AudioSegment to numpy array
    y = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sr = segment.frame_rate

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc


def build_feature_vector(
    audio_segments: Dict[int, AudioSegment],
    include_chromagram: bool = True,
    include_melspectrogram: bool = True,
    include_mfcc: bool = True,
    chromagram_hop_length: int = 512,
    mel_n_mels: int = 128,
    mel_n_fft: int = 2048,
    mel_hop_length: int = 512,
    mfcc_n_mfcc: int = 20,
    mfcc_n_fft: int = 2048,
    mfcc_hop_length: int = 512,
) -> Dict[int, np.ndarray]:
    """
    Build a feature vector from computed chromagrams, melspectrograms, and MFCCs.

    Args:
        audio_segments (Dict[int, AudioSegment]): Dictionary containing index and AudioSegment pairs.
        include_chromagram (bool): Whether to include chromagrams in the feature vector.
        include_melspectrogram (bool): Whether to include melspectrograms in the feature vector.
        include_mfcc (bool): Whether to include MFCCs in the feature vector.
        chromagram_hop_length (int): Number of samples between successive frames for chromagram computation.
        mel_n_mels (int): Number of mel bands to generate for mel spectrogram computation.
        mel_n_fft (int): Number of samples to use for FFT for mel spectrogram computation.
        mel_hop_length (int): Number of samples between successive frames for mel spectrogram computation.
        mfcc_n_mfcc (int): Number of MFCCs to generate for MFCC computation.
        mfcc_n_fft (int): Number of samples to use for FFT for MFCC computation.
        mfcc_hop_length (int): Number of samples between successive frames for MFCC computation.

    Returns:
        Dict[int, np.ndarray]: Dictionary containing feature vectors for each file.
    """
    feature_vectors: Dict[int, np.ndarray] = {}

    for index, segment in audio_segments.items():
        features = []

        if include_chromagram:
            chromagram = convert_to_chromagram(
                segment, hop_length=chromagram_hop_length
            )
            features.append(chromagram)

        if include_melspectrogram:
            melspectrogram = convert_to_melspectrogram(
                segment, n_mels=mel_n_mels, n_fft=mel_n_fft, hop_length=mel_hop_length
            )
            features.append(melspectrogram)

        if include_mfcc:
            mfcc = convert_to_mfcc(
                segment,
                n_mfcc=mfcc_n_mfcc,
                n_fft=mfcc_n_fft,
                hop_length=mfcc_hop_length,
            )
            features.append(mfcc)

        feature_vectors[index] = np.concatenate(features, axis=0)

    print(
        "shape of chromagram:",
        chromagram.shape if include_chromagram else "N/A",
    )
    print(
        "shape of melspectrogram:",
        melspectrogram.shape if include_melspectrogram else "N/A",
    )
    print("shape of mfcc:", mfcc.shape if include_mfcc else "N/A")

    return feature_vectors


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


def compute_similarity_matrix(
    feature_vectors: dict[int, np.ndarray], metric: str = "pearson_correlation"
) -> np.ndarray:
    """
    Compute the similarity matrix between pairs of numpy arrays.

    Parameters:
        feature_vectors (dict of numpy arrays): Dictionary containing feature vectors.
        metric (str): Metric for computing similarity. Can be 'cosine' or 'pearson_correlation'.

    Returns:
        similarity_matrix (numpy array): Similarity matrix between pairs of arrays.
    """
    num_arrays = len(feature_vectors)
    similarity_matrix = np.zeros((num_arrays, num_arrays))

    for i, vec1 in enumerate(feature_vectors.values()):
        for j, vec2 in enumerate(feature_vectors.values()):
            if j < i:
                similarity_matrix[i, j] = similarity_matrix[j, i]

            if metric == "pearson_correlation":
                correlation_coefficient = np.corrcoef(np.ravel(vec1), np.ravel(vec2))[
                    0, 1
                ]
                similarity_score = correlation_coefficient
            elif metric == "cosine":
                similarity_score = 1 - cosine_similarity(np.ravel(vec1), np.ravel(vec2))
            else:
                raise ValueError(
                    "Invalid metric. Choose either 'cosine' or 'pearson_correlation'."
                )

            similarity_matrix[i, j] = similarity_score

    return similarity_matrix


def moving_average(data, window_size):
    """
    Apply a uniform moving average filter to a one-dimensional array of data.

    Parameters:
        data (numpy.ndarray): Input data array.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: The array after applying the moving average filter.
    """
    # Pad the data array to handle mode 'valid' properly
    padding = window_size // 2  # adjust padding based on the window size
    padded_data = np.pad(data, (padding, padding), mode="edge")

    # Apply convolution with mode 'valid'
    smoothed_data = np.convolve(
        padded_data, np.ones(window_size) / window_size, mode="valid"
    )

    return smoothed_data


def correlation_filter(similarity_matrix, window_size_seconds, chunk_length):
    """
    Filter along the diagonals of the similarity matrix to compute similarity between extended regions of the song.

    Parameters:
        similarity_matrix (numpy.ndarray): The similarity matrix.
        window_size_seconds (float): The size of the moving average window in seconds.
        chunk_length (int): The length of each chunk in milliseconds.

    Returns:
        numpy.ndarray: The restructured time-lag matrix.
    """
    # Calculate the window size in terms of number of chunks
    window_size_chunks = int(window_size_seconds * 1000 / chunk_length)

    n_samples = similarity_matrix.shape[0]
    time_lag_matrix = np.zeros_like(similarity_matrix)

    # Apply uniform moving average filter along diagonals
    for i in range(n_samples):
        diagonal_data = similarity_matrix.diagonal(i)
        smoothed_diagonal = moving_average(diagonal_data, window_size_chunks)
        expected_length = n_samples - i
        if len(smoothed_diagonal) != expected_length:
            if len(smoothed_diagonal) > expected_length:
                smoothed_diagonal = smoothed_diagonal[:-1]  # Remove the last element
        time_lag_matrix[i, i:] = smoothed_diagonal

    return time_lag_matrix


def select_thumbnail(time_lag_surface: np.ndarray, chunk_length: int) -> float | None:
    """
    Selects a thumbnail position from a time-lag surface.

    This selection occurs by locating the maximum element of the time-lag matrix
    subject to two constraints:
    - a lag greater than one-tenth the length of the song
    - occurs less than three-fourths of the way into the song.

    Parameters:
        time_lag_surface (numpy.ndarray): The time-lag surface matrix.
        chunk_length (int): The length of each chunk in milliseconds.

    Returns:
        float | None: The start time of the selected thumbnail if it satisfies the constraints, otherwise returns None.
    """
    n_chunks = time_lag_surface.shape[0]
    # Calculate the length of each chunk in seconds
    chunk_length_seconds = chunk_length / 1000

    max_value = -float("inf")
    max_start_time = None

    # Find the maximum value in the time-lag surface matrix
    for i in range(n_chunks):
        for j in range(n_chunks):
            if i > j and j > (chunk_length_seconds * n_chunks / 10) and j < (3 * chunk_length_seconds * n_chunks / 4):
                if time_lag_surface[i, j] > max_value:
                    max_value = time_lag_surface[i, j]
                    # Calculate the start time from the chunk index and chunk length
                    max_start_time = i * chunk_length_seconds

    # Check if a valid maximum start time is found
    if max_start_time is not None:
        return max_start_time
    else:
        return None




def format_time_tuple(start_time: float, length: float) -> tuple[str, str]:
    """
    Convert a time tuple (start_time, length) to formatted start and end time strings.

    Parameters:
        start_time (float): The start time in seconds.
        length (float): The length of the time interval in seconds.

    Returns:
        tuple[str, str]: A tuple containing the formatted start and end time strings in the format 'mm:ss'.
    """
    # Calculate the end time based on the start time and length
    end_time = start_time + length

    # Convert start time and end time from seconds to minutes and seconds
    start_minutes, start_seconds = divmod(int(start_time), 60)
    end_minutes, end_seconds = divmod(int(end_time), 60)

    # Format start time and end time as 'mm:ss'
    start_time_formatted = f"{start_minutes:02d}:{start_seconds:02d}"
    end_time_formatted = f"{end_minutes:02d}:{end_seconds:02d}"

    return start_time_formatted, end_time_formatted
