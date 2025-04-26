import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from demo_speaker_diarization import demo_speaker_diarization
import threading
import time
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TASK_FOLDER'] = 'task'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1000  # 1GB max upload size
app.config['HISTORY_FILE'] = 'task/video_history.json'

# Ensure upload and task directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TASK_FOLDER'], exist_ok=True)

# Store processing tasks with their status
tasks = {}

# Video history functions
def load_video_history():
    """Load the video processing history from JSON file"""
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {str(e)}")
            return []
    return []

def save_video_to_history(task_id, video_data):
    """Save a processed video to the history"""
    history = load_video_history()
    
    # Add timestamp if not present
    if 'timestamp' not in video_data:
        video_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Add task_id if not present
    video_data['task_id'] = task_id
    
    # Check if entry already exists and update it, or add new entry
    for i, entry in enumerate(history):
        if entry.get('task_id') == task_id:
            history[i] = video_data
            break
    else:
        history.append(video_data)
    
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Limit history size to 50 entries
    history = history[:50]
    
    try:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving history: {str(e)}")

@app.route('/')
def index():
    # Pass video history to the template
    history = load_video_history()
    return render_template('index.html', history=history)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part'}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
        
    # Get the aspect ratio from form
    aspect_ratio = request.form.get('aspect_ratio', '9:16')
    width, height = map(int, aspect_ratio.split(':'))
    
    # Get advanced settings
    detection_threshold = float(request.form.get('detection_threshold', '0.4'))
    crop_smoothness = float(request.form.get('crop_smoothness', '0.2'))
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(file_path)
    
    # Create task entry
    tasks[task_id] = {
        'status': 'uploaded',
        'progress': 0,
        'filename': filename,
        'file_path': file_path,
        'aspect_ratio': aspect_ratio,
        'width': width,
        'height': height,
        'detection_threshold': detection_threshold,
        'crop_smoothness': crop_smoothness,
        'output_path': None,
        'annotated_path': None,
        'error': None
    }
    
    # Return the task ID to the client
    return jsonify({'task_id': task_id, 'status': 'uploaded'})

@app.route('/process_url', methods=['POST'])
def process_url():
    try:
        data = request.json
        video_url = data.get('video_url')
        aspect_ratio = data.get('aspect_ratio', '9:16')
        
        # Get advanced settings
        detection_threshold = float(data.get('detection_threshold', 0.4))
        crop_smoothness = float(data.get('crop_smoothness', 0.2))
        
        if not video_url:
            return jsonify({'error': 'No video URL provided'}), 400
            
        # Parse aspect ratio
        width, height = map(int, aspect_ratio.split(':'))
        
        # Create a unique task ID
        task_id = str(uuid.uuid4())
        
        # Extract filename from URL or use a default
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(video_url)
            url_filename = os.path.basename(parsed_url.path)
            if not url_filename or '.' not in url_filename:
                url_filename = 'video_from_url.mp4'
        except:
            url_filename = 'video_from_url.mp4'
        
        # Create task entry
        tasks[task_id] = {
            'status': 'processing',  # Start processing immediately
            'progress': 5,
            'filename': url_filename,
            'file_path': None,  # No local file yet
            'video_url': video_url,  # Store the URL
            'aspect_ratio': aspect_ratio,
            'width': width,
            'height': height,
            'detection_threshold': detection_threshold,
            'crop_smoothness': crop_smoothness,
            'output_path': None,
            'annotated_path': None,
            'error': None,
            'is_url': True
        }
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_url_task, args=(task_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'processing'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_url_task(task_id):
    task = tasks[task_id]
    
    try:
        # Setup paths
        task_temp_dir = os.path.join(app.config['TASK_FOLDER'], task_id)
        os.makedirs(task_temp_dir, exist_ok=True)
        
        output_filename = f"processed_{os.path.splitext(task['filename'])[0]}.mp4"
        output_path = os.path.join(task_temp_dir, output_filename)
        
        # Update progress for frontend
        task['progress'] = 10
        
        # Run the diarization with progress updates
        def progress_callback(stage, progress):
            if stage == 'download':
                task['progress'] = 10 + progress * 0.2  # 10-30%
            elif stage == 'detection':
                task['progress'] = 30 + progress * 0.4  # 30-70%
            elif stage == 'cropping':
                task['progress'] = 70 + progress * 0.3  # 70-100%
        
        # Call the demo_speaker_diarization function directly with the URL
        demo_speaker_diarization(
            task_id=task_id,
            video_url=task['video_url'],  # Using the URL directly
            output_path=output_path,
            target_ratio=(task['width'], task['height']),
            min_score=task['detection_threshold'],
            crop_smoothness=task['crop_smoothness'],
            progress_callback=progress_callback
        )
        
        # Update paths for the UI
        task['output_path'] = output_path
        task['annotated_path'] = output_path.replace('.mp4', '_annotated.mp4')
        
        # Update status to completed
        task['status'] = 'completed'
        task['progress'] = 100
        
        # Save to history
        save_video_to_history(task_id, {
            'filename': task['filename'],
            'aspect_ratio': task['aspect_ratio'],
            'output_path': task['output_path'],
            'annotated_path': task['annotated_path'],
            'video_url': task['video_url'],
            'detection_threshold': task['detection_threshold'],
            'crop_smoothness': task['crop_smoothness'],
            'status': 'completed',
            'thumbnail': os.path.join(task_temp_dir, 'thumbnail.jpg'),
        })
        
        # Generate thumbnail
        try:
            generate_thumbnail(task['output_path'], os.path.join(task_temp_dir, 'thumbnail.jpg'))
        except Exception as e:
            print(f"Error generating thumbnail: {str(e)}")
        
    except Exception as e:
        # Handle errors
        task['status'] = 'error'
        task['error'] = str(e)
        print(f"Error processing video URL: {str(e)}")

@app.route('/process/<task_id>', methods=['POST'])
def process_video(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    # Start processing in a background thread
    thread = threading.Thread(target=process_video_task, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'processing'})

def process_video_task(task_id):
    task = tasks[task_id]
    
    try:
        # Update status to processing
        task['status'] = 'processing'
        task['progress'] = 5
        
        # Setup paths
        task_temp_dir = os.path.join(app.config['TASK_FOLDER'], task_id)
        os.makedirs(task_temp_dir, exist_ok=True)
        
        output_filename = f"processed_{os.path.splitext(task['filename'])[0]}.mp4"
        output_path = os.path.join(task_temp_dir, output_filename)
        
        # Update progress for frontend
        task['progress'] = 10
        
        # Run the diarization with progress updates
        def progress_callback(stage, progress):
            if stage == 'download':
                task['progress'] = 10 + progress * 0.2  # 10-30%
            elif stage == 'detection':
                task['progress'] = 30 + progress * 0.4  # 30-70%
            elif stage == 'cropping':
                task['progress'] = 70 + progress * 0.3  # 70-100%
        
        # Call the demo_speaker_diarization function
        demo_speaker_diarization(
            task_id=task_id,
            video_url=task['file_path'],  # Using the local file path
            output_path=output_path,
            target_ratio=(task['width'], task['height']),
            min_score=task['detection_threshold'],
            crop_smoothness=task['crop_smoothness'],
            progress_callback=progress_callback
        )
        
        # Update paths for the UI
        task['output_path'] = output_path
        task['annotated_path'] = output_path.replace('.mp4', '_annotated.mp4')
        
        # Update status to completed
        task['status'] = 'completed'
        task['progress'] = 100
        
        # Save to history
        save_video_to_history(task_id, {
            'filename': task['filename'],
            'aspect_ratio': task['aspect_ratio'],
            'output_path': task['output_path'],
            'annotated_path': task['annotated_path'],
            'detection_threshold': task['detection_threshold'],
            'crop_smoothness': task['crop_smoothness'],
            'status': 'completed',
            'thumbnail': os.path.join(task_temp_dir, 'thumbnail.jpg'),
        })
        
        # Generate thumbnail
        try:
            generate_thumbnail(task['output_path'], os.path.join(task_temp_dir, 'thumbnail.jpg'))
        except Exception as e:
            print(f"Error generating thumbnail: {str(e)}")
        
    except Exception as e:
        # Handle errors
        task['status'] = 'error'
        task['error'] = str(e)
        print(f"Error processing video: {str(e)}")

# Function to generate thumbnail
def generate_thumbnail(video_path, thumbnail_path):
    """Generate a thumbnail from the first frame of a video"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            # Resize to a reasonable thumbnail size
            height, width = frame.shape[:2]
            max_dimension = 300
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            thumbnail = cv2.resize(frame, (new_width, new_height))
            cv2.imwrite(thumbnail_path, thumbnail)
        cap.release()
    except Exception as e:
        print(f"Error in thumbnail generation: {str(e)}")

@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    
    # Get system info
    system_info = {
        'platform': os.name,
        'memory_usage': get_memory_usage()
    }
    
    # Get any logs for this task
    logs = get_task_logs(task_id)
    
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'filename': task['filename'],
        'error': task['error'],
        'system_info': system_info,
        'logs': logs
    })

def get_memory_usage():
    """Get memory usage information for the current process"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
            'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
            'percent': process.memory_percent(),
            'total_system': psutil.virtual_memory().total / (1024 * 1024)  # Total system memory in MB
        }
    except ImportError:
        # If psutil is not available, return dummy data
        return {
            'rss': 0,
            'vms': 0,
            'percent': 0,
            'total_system': 0
        }
    except Exception as e:
        print(f"Error getting memory usage: {str(e)}")
        return {
            'rss': 0,
            'vms': 0,
            'percent': 0,
            'total_system': 0
        }

def get_task_logs(task_id):
    """Get processing logs for a specific task"""
    logs = []
    
    # Create mock logs if real logs are not available
    # In a real implementation, these would come from a log file or database
    if task_id in tasks:
        task = tasks[task_id]
        progress = task['progress']
        
        # Add logs based on processing stage
        if progress >= 10:
            logs.append({
                'timestamp': '2025-04-26 18:54:45,123',
                'level': 'INFO',
                'message': 'Starting video processing'
            })
        
        if progress >= 30:
            logs.append({
                'timestamp': '2025-04-26 18:55:36,784',
                'level': 'INFO',
                'message': 'Processing faces'
            })
            
        if progress >= 70:
            logs.append({
                'timestamp': '2025-04-26 18:56:04,130',
                'level': 'INFO',
                'message': 'Created 2 face tracks'
            })
            
        if progress >= 90:
            logs.append({
                'timestamp': '2025-04-26 18:56:45,892',
                'level': 'INFO',
                'message': 'Generating output video'
            })
    
    return logs

@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = tasks[task_id]
    
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed yet'}), 400
    
    # Get the filenames for the output videos
    output_filename = os.path.basename(task['output_path'])
    annotated_filename = os.path.basename(task['annotated_path'])
    
    # Get any logs for this task
    logs = get_task_logs(task_id)
    
    # Load video history
    history = load_video_history()
    
    return render_template('result.html', 
                          task_id=task_id, 
                          output_filename=output_filename, 
                          annotated_filename=annotated_filename,
                          logs=logs,
                          history=history,
                          aspect_ratio=task['aspect_ratio'],
                          detection_threshold=task['detection_threshold'],
                          crop_smoothness=task['crop_smoothness'])

@app.route('/video/<task_id>/<filename>')
def video(task_id, filename):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task_dir = os.path.join(app.config['TASK_FOLDER'], task_id)
    return send_from_directory(task_dir, filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    task_dir = os.path.join(app.config['TASK_FOLDER'], task_id)
    return send_from_directory(directory=task_dir, path=filename, as_attachment=True)

@app.route('/thumbnail/<task_id>')
def thumbnail(task_id):
    # Check if the thumbnail exists in the task directory
    thumbnail_path = os.path.join(app.config['TASK_FOLDER'], task_id, 'thumbnail.jpg')
    if os.path.exists(thumbnail_path):
        return send_from_directory(os.path.dirname(thumbnail_path), os.path.basename(thumbnail_path))
    else:
        # Return a default thumbnail
        return send_from_directory('static/img', 'thumbnail.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 