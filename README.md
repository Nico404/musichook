# Music Thumbnailing Exploration

## Overview
This repository explores various techniques for music thumbnailing, a process of summarizing or representing music pieces with concise excerpts or thumbnails. The primary focus lies on two main approaches: sound similarity analysis using Spotify API and self-chorus detection based on chroma-based representations.


## Music Thumbnailing app

### Demo App [musichook-app](https://musichook-qz5vasr6bq-ew.a.run.app/upload)

![Thumbnailing-app](https://i.imgur.com/g26op22.png)


## Sound Similarity
Utilizing the Spotify API, this project downloads song previews, segments the original songs into consistent-sized pieces, and computes the Structural Similarity Index (SSIM) on the chromagram to identify start and end times within the songs.

### Demo Notebook [sound_similarity_spotify.ipynb](notebooks/sound_similarity_spotify.ipynb)

## Self-Chorus Detection
Inspired by the paper 'TO CATCH A CHORUS: USING CHROMA-BASED REPRESENTATIONS FOR AUDIO THUMBNAILING', this approach involves segmenting songs into small chunks and computing a time-by-time similarity matrix to uncover patterns. A uniform moving average filter is applied to identify windows of similarities, followed by the computation of a time-by-lag matrix to facilitate easier pattern extraction. Ultimately, the song thumbnail is selected from this matrix.

### Demo Notebook  [self_chorus_detection.ipynb](notebooks/self_chorus_detection.ipynb)
