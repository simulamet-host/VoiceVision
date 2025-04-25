# VoiceVision Web Application

This web application provides a user-friendly interface for the VoiceVision AI-powered speaker diarization and smart video cropping system.

## Features

- Drag-and-drop video upload interface
- Aspect ratio selection (9:16, 1:1, 4:5) for social media optimization
- Real-time processing progress tracking
- Video preview before processing
- Side-by-side comparison of original, annotated, and cropped videos
- Easy download of processed videos

## Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the model weights:
   - Place `talknet_speaker_v1.model` in the `weights/` directory
   - Place `s3fd_facedetection_v1.pth` in the `weights/` directory

3. Run the Flask application:
   ```
   python app.py
   ```

4. Access the web interface at:
   ```
   http://localhost:5000
   ```

## Directory Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
  - `base.html`: Base template with common layout
  - `index.html`: Home page with video upload interface
  - `result.html`: Results page showing processed videos
- `static/`: Static assets
  - `css/`: CSS stylesheets
  - `img/`: Image assets
- `uploads/`: Temporary storage for uploaded videos
- `task/`: Output directory for processing results

## Processing Flow

1. User uploads a video and selects aspect ratio
2. Video is uploaded to server and stored temporarily
3. User triggers processing of the video
4. Backend processes the video in stages:
   - Download & video preparation
   - Speaker detection and analysis
   - Smart cropping generation
5. Progress is tracked and displayed to the user in real-time
6. Results page shows both annotated and cropped videos
7. User can download the processed videos

## Customization

- You can modify the aspect ratio options in `templates/index.html`
- CSS styling can be changed in `static/css/style.css`
- Processing settings can be adjusted in `app.py` 