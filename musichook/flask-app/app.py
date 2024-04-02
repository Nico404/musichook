from flask import Flask, render_template, request
import os

from .audio_processing import (
    cut_song_into_segments,
    build_feature_vector,
    compute_similarity_matrix,
    correlation_filter,
    select_thumbnail,
)


app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "static/uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("index.html", alert="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", alert="No selected file")

    if file and file.filename.endswith(".mp3"):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "song.mp3")
        file.save(filepath)

        ### Audio Processing
        print("Cutting the song into segments...")
        quarter_second_segments = cut_song_into_segments(filepath, 250)
        print("Building the feature vector...")
        feature_vector = build_feature_vector(
            quarter_second_segments, True, False, False
        )
        print("Compute similarity matrix")
        similarity_matrix = compute_similarity_matrix(feature_vector, "cosine")
        print("Compute time_lag_surface matrix")
        time_lag_surface = correlation_filter(similarity_matrix, 20)
        print("Select the thumbnail")
        start_time, window_size_out = select_thumbnail(time_lag_surface)
        start_time = start_time / 4
        end_time = start_time + 120

        return render_template(
            "index.html",
            uploaded_file_url=filepath,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        return render_template("index.html", alert="Please upload a valid mp3 file")


if __name__ == "__main__":
    app.run()
