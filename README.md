# Real-Time Speaker Clustering Using Deep Metric Learning

This project is a real-time speaker clustering tool that processes an audio file to detect, differentiate, and cluster speakers based on their voice embeddings.  
It uses **OpenAI Whisper** for transcription, **pyannote-audio** for speaker embeddings, **agglomerative clustering** for grouping speakers, and **Tkinter** for a simple graphical user interface (GUI).

## Features
- Upload any `.wav`, `.mp3`, or `.m4a` audio file.
- Perform automatic speech transcription.
- Identify and cluster different speakers without prior profiles.
- Visualize speaker clusters using PCA plots.
- Easy-to-use GUI built with Tkinter.

## Installation

Before running the project, install the required libraries:

```bash
# tkinter usually comes pre-installed. If not:
# Install via system package manager (not pip), e.g., on Ubuntu:
sudo apt-get install python3-tk

# Install required Python packages
pip install torch
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install pyannote.audio
pip install openai-whisper
pip install git+https://github.com/openai/whisper.git
