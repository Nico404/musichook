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
        metric (str): Metric for computing similarity. Can be 'pearson_correlation'.

    Returns:
        similarity_matrix (numpy array): Similarity matrix between pairs of arrays.
    """
    num_arrays = len(feature_vectors)
    similarity_matrix = np.zeros((num_arrays, num_arrays))

    # Normalize the feature vectors by subtracting the mean of each vector
    normalized_feature_vectors = {key: vec - np.mean(vec) for key, vec in feature_vectors.items()}

    for i, vec1 in enumerate(normalized_feature_vectors.values()):
        for j, vec2 in enumerate(normalized_feature_vectors.values()):
            if j < i:
                similarity_matrix[i, j] = similarity_matrix[j, i]
            else:
                if metric == "pearson_correlation":
                    # Check if either vector is constant (std dev is zero)
                    if np.std(vec1) == 0 or np.std(vec2) == 0:
                        similarity_score = 0  # Assign zero similarity if either vector is constant
                    else:
                        correlation_coefficient = np.corrcoef(np.ravel(vec1), np.ravel(vec2))[0, 1]
                        similarity_score = correlation_coefficient
                else:
                    raise ValueError("Invalid metric. Choose 'pearson_correlation'.")

                similarity_matrix[i, j] = similarity_score
                if i != j:
                    similarity_matrix[j, i] = similarity_score

    return similarity_matrix

def compute_time_lag_surface(similarity_matrix: np.ndarray, filter_length: int) -> np.ndarray:
    """
    Compute the time-lag surface from the similarity matrix by applying a moving average filter along diagonals.

    Parameters:
        similarity_matrix (numpy array): Similarity matrix.
        filter_length (int): Length of the moving average filter.

    Returns:
        time_lag_matrix (numpy array): Time-lag surface.
    """
    num_frames = similarity_matrix.shape[0]
    time_lag_matrix = np.zeros((num_frames, num_frames))

    for lag in range(num_frames):
        diagonal = np.diagonal(similarity_matrix, offset=lag)
        if len(diagonal) >= filter_length:
            filtered_diagonal = np.convolve(diagonal, np.ones(filter_length) / filter_length, mode='valid')
            time_lag_matrix[lag, lag:lag + len(filtered_diagonal)] = filtered_diagonal

    return time_lag_matrix


import numpy as np

def find_thumbnail(time_lag, song_length):
    """
    Finds the optimal thumbnail position and length within a time-lag matrix.

    Parameters:
    time_lag : numpy.ndarray
        A 2D numpy array representing the time-lag matrix.
    song_length : int or float
        The length of the song in seconds.

    Returns:
    tuple
        A tuple containing the thumbnail position and length.
        thumbnail_time_position : int
            The time position of the thumbnail within the song.
        thumbnail_length : int
            The length of the thumbnail.
    """
    # Calculate constraints
    lag_threshold = int(song_length / 10)
    max_lag_position = int(3 * song_length / 4)

    # Apply constraints to time_lag matrix
    constrained_time_lag = time_lag.copy()
    constrained_time_lag[:lag_threshold, :] = 0
    constrained_time_lag[max_lag_position:, :] = 0

    # Find the maximum element in the constrained time_lag matrix
    max_index = np.unravel_index(np.argmax(constrained_time_lag), constrained_time_lag.shape)

    # Extract time-position of the maximum
    thumbnail_time_position = max_index[1]

    # Define the thumbnail length
    thumbnail_length = max_index[0]

    return thumbnail_time_position, thumbnail_length
