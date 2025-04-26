import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from demo_speaker_diarization import demo_speaker_diarization
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TASK_FOLDER'] = 'task'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1000  # 1GB max upload size

# Ensure upload and task directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TASK_FOLDER'], exist_ok=True)

# Store processing tasks with their status
tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

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
            min_score=0.4,
            progress_callback=progress_callback
        )
        
        # Update paths for the UI
        task['output_path'] = output_path
        task['annotated_path'] = output_path.replace('.mp4', '_annotated.mp4')
        
        # Update status to completed
        task['status'] = 'completed'
        task['progress'] = 100
        
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
            min_score=0.4,
            progress_callback=progress_callback
        )
        
        # Update paths for the UI
        task['output_path'] = output_path
        task['annotated_path'] = output_path.replace('.mp4', '_annotated.mp4')
        
        # Update status to completed
        task['status'] = 'completed'
        task['progress'] = 100
        
    except Exception as e:
        # Handle errors
        task['status'] = 'error'
        task['error'] = str(e)
        print(f"Error processing video: {str(e)}")

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
        return redirect(url_for('index'))
    
    task = tasks[task_id]
    if task['status'] != 'completed':
        return redirect(url_for('index'))
    
    # Extract just the filenames for the template
    output_filename = os.path.basename(task['output_path'])
    annotated_filename = os.path.basename(task['annotated_path'])
    
    # Get logs for this task
    logs = get_task_logs(task_id)
    
    return render_template('result.html', 
                          task_id=task_id,
                          filename=task['filename'],
                          output_filename=output_filename,
                          annotated_filename=annotated_filename,
                          logs=logs)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 