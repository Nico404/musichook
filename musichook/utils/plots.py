import matplotlib.pyplot as plt
import librosa
import numpy as np
import plotly.express as px


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


def display_similarity_matrix_px(
    similarity_matrix: np.ndarray,
    x_axis_title: str = "Time",
    y_axis_title: str = "Time",
) -> None:
    """
    Display a similarity matrix plot with x and y axis titles.

    Parameters:
        similarity_matrix (numpy array): Similarity matrix between pairs of arrays.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.
    """
    fig = px.imshow(similarity_matrix, origin="lower", color_continuous_scale="gray")
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=y_axis_title)
    fig.update_layout(width=750, height=750)
    fig.show()


def display_time_lag_surface_px(
    time_lag_surface: np.ndarray,
    x_axis_title: str = "Lag",
    y_axis_title: str = "Time",
) -> None:
    """
    Display a time lag surface post similarity matrix filter plot with x and y axis titles.

    Parameters:
        time_lag_surface (numpy.ndarray): The time-lag surface matrix.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.
    """
    # Transform the time-lag matrix for plotting
    time_lag_surface_transformed = np.flipud(time_lag_surface.T)

    fig = px.imshow(
        time_lag_surface_transformed, origin="lower", color_continuous_scale="gray"
    )
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=y_axis_title)
    fig.update_layout(width=750, height=750)
    fig.show()


def display_similarity_matrix(
    similarity_matrix: np.ndarray,
    x_axis_title: str = "Time",
    y_axis_title: str = "Time",
    width: int = 8,
    height: int = 8,
) -> None:
    """
    Display a similarity matrix plot with x and y axis titles.

    Parameters:
        similarity_matrix (numpy array): Similarity matrix between pairs of arrays.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.
        width (int): Width of the figure in inches.
        height (int): Height of the figure in inches.
    """
    plt.figure(figsize=(width, height))
    plt.imshow(similarity_matrix, origin="lower", cmap="gray")
    plt.colorbar()
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.show()


def display_time_lag_surface(
    time_lag_surface: np.ndarray,
    x_axis_title: str = "Lag",
    y_axis_title: str = "Time",
    width: int = 8,
    height: int = 8,
) -> None:
    """
    Display a time lag surface post similarity matrix filter plot with x and y axis titles.

    Parameters:
        time_lag_surface (numpy.ndarray): The time-lag surface matrix.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.
        width (int): Width of the figure in inches.
        height (int): Height of the figure in inches.
    """
    # Transform the time-lag matrix for plotting
    time_lag_surface_transformed = np.flipud(time_lag_surface.T)

    plt.figure(figsize=(width, height))
    plt.imshow(time_lag_surface_transformed, origin="lower", cmap="gray")
    plt.colorbar()
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.show()
