from flask import Flask, render_template, request, redirect, send_file
import os
from time import sleep
from .audio_processing import (
    cut_song_into_segments,
    build_feature_vector,
    compute_similarity_matrix,
    correlation_filter,
    select_thumbnail,
    cut_chunk_into_audio,
)


app = Flask(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "static/uploads")
CHUNK_FOLDER = os.path.join(ROOT_DIR, "static/chunks")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def root():
    return render_template("index.html")

@app.route("/chunk/<filename>/<start_time>/<end_time>", methods=["GET"])
def chunk(filename, start_time, end_time):
    song_path = os.path.join(UPLOAD_FOLDER, filename)
    chunk_filename = f"{filename[:-4]}_{int(float(start_time))}_{int(float(end_time))}.mp3"
    chunk_path = os.path.join(CHUNK_FOLDER, chunk_filename)

    print("song path:", song_path)
    print("chunk path:", chunk_path)

    # Clean the chunk folder
    for file in os.listdir(CHUNK_FOLDER):
        os.remove(os.path.join(CHUNK_FOLDER, file))

    schunk = cut_chunk_into_audio(song_path, int(float(start_time)), int(float(end_time)))
    schunk.export(chunk_path, format="mp3")

    return send_file(chunk_path, as_attachment=True)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    uploaded_file_url = None
    if "file" not in request.files:
        return render_template("index.html", alert="No file part")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", alert="No selected file")

    if file and file.filename.endswith(".mp3"):
        filename = file.filename
        print(filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print("File uploaded successfully:", filename)

        ### Audio Processing
        chunk_length_ms = 500
        window_size_seconds = 20

        print("Cutting the song into segments...")
        quarter_second_segments = cut_song_into_segments(filepath, chunk_length_ms)
        print("Building the feature vector...")
        feature_vector = build_feature_vector(
            quarter_second_segments, True, False, False
        )
        print("Compute similarity matrix")
        similarity_matrix = compute_similarity_matrix(feature_vector, "cosine")
        print("Compute time_lag_surface matrix")
        time_lag_surface = correlation_filter(similarity_matrix, window_size_seconds, chunk_length_ms)
        print("Select the thumbnail")
        start_time = select_thumbnail(time_lag_surface, chunk_length_ms)

        if start_time is not None:
            print("Start time of the selected thumbnail:", start_time)
            end_time = start_time + 30  # End time is 30 seconds after the start time
            print("End time of the selected thumbnail:", end_time)
            print('Rendering the template')
            return render_template(
            "index.html",
            uploaded_file_url=filepath,
            songname=filename,
            start_time=start_time,
            end_time=end_time,
        )
        else:
            print("No valid thumbnail found.")
            return render_template(
                "index.html",
                alert="No valid thumbnail found. Please try again.",
            )

    else:
        return render_template("index.html", alert="Please upload a valid mp3 file")


if __name__ == "__main__":
    app.run()
