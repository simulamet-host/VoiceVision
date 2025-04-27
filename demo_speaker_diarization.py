import os
import logging
from cropper import SmartCropper, VideoAnnotator
from components.talknet_modules.talknet_inference import SpeakerDetector
from utils import VideoStore
from transcriber import VideoTranscriber
import subprocess

log = logging.getLogger('aiproducer')

def demo_speaker_diarization(task_id: str, video_url: str, output_path: str, target_ratio=(9, 16), min_score=0.4, crop_smoothness=0.2, enable_transcription=True, whisper_model="base", progress_callback=None):
    """Run speaker diarization demo: process video and generate cropped output.
    
    Args:
        task_id: Unique identifier for this processing task
        video_url: URL or file path to the video
        output_path: Where to save the output cropped video
        target_ratio: Aspect ratio as (width, height) tuple
        min_score: Minimum speaker detection confidence score (0-1)
        crop_smoothness: Smoothness of camera movement (0-1, higher is smoother)
        enable_transcription: Whether to generate transcription using Whisper
        whisper_model: Whisper model size to use ('tiny', 'base', 'small', 'medium', 'large')
        progress_callback: Optional callback function for progress updates
              Function signature: progress_callback(stage, progress)
              Where stage is 'download', 'detection', 'cropping', or 'transcription'
              And progress is a float between 0 and 1
    """
    try:
        # Create task directories
        task_temp_dir = os.path.join("task", task_id, "temp")
        os.makedirs(task_temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download and concatenate video
        log.info(f"Downloading video from {video_url}")
        
        # Check if input is a local file or URL
        if os.path.isfile(video_url):
            # It's a local file, just copy or use directly
            log.info("Using local video file")
            concatenated_video_path = os.path.join(task_temp_dir, "concatenated_video.ts")
            import shutil
            shutil.copy(video_url, concatenated_video_path)
            if progress_callback:
                progress_callback('download', 1.0)  # 100% complete for download stage
        else:
            # It's a URL, download it
            video_store = VideoStore(task_id, video_url, os.path.join(task_temp_dir, "segments"))
            video_store.download_video()
            
            # Update progress after download if callback provided
            if progress_callback:
                progress_callback('download', 0.5)  # 50% complete for download stage
                
            concatenated_video_path = os.path.join(task_temp_dir, "concatenated_video.ts")
            concatenated_video_path = video_store.concatenate_segments_to_file(concatenated_video_path)
            
            # Update progress after concatenation if callback provided
            if progress_callback:
                progress_callback('download', 1.0)  # 100% complete for download stage
        
        # Downsample the video for better performance and browser compatibility
        downsampled_video_path = os.path.join(task_temp_dir, "downsampled_video.mp4")
        log.info("Downsampling video for better processing performance")
        
        # First, check the dimensions of the input video
        try:
            import cv2
            input_cap = cv2.VideoCapture(concatenated_video_path)
            if not input_cap.isOpened():
                log.error("Could not open input video for dimension check")
                raise RuntimeError("Failed to open input video")
                
            input_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(input_cap.get(cv2.CAP_PROP_FPS))
            input_cap.release()
            
            log.info(f"Original video dimensions: {input_width}x{input_height}, FPS: {fps}")
            
            # Calculate target dimensions while maintaining aspect ratio
            if input_width > input_height:
                # Landscape video
                target_width = min(1280, input_width)
                target_height = int(target_width * input_height / input_width)
                target_height = target_height - (target_height % 2)  # Ensure even height (required by some codecs)
            else:
                # Portrait video
                target_height = min(720, input_height)
                target_width = int(target_height * input_width / input_height)
                target_width = target_width - (target_width % 2)  # Ensure even width
                
            log.info(f"Target dimensions for processing: {target_width}x{target_height}")
            
            scale_filter = f"scale={target_width}:{target_height}"
        except Exception as e:
            log.warning(f"Error determining video dimensions: {str(e)}")
            # Fallback to safe scaling
            scale_filter = "scale='min(1280,iw)':'-2'"
            fps = 30  # Default FPS
        
        downsample_cmd = [
            "ffmpeg", "-y",
            "-i", concatenated_video_path,
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-crf", "28",  # Higher CRF for smaller file
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            downsampled_video_path
        ]
        subprocess.run(downsample_cmd, check=True)
        
        # Use the downsampled video for processing
        processed_video_path = downsampled_video_path
        
        # Run speaker detection
        log.info("Running speaker detection...")
        detector = SpeakerDetector(model_path="weights/talknet_speaker_v1.model")
        
        # Process detection in stages to update progress
        total_frames = 0
        try:
            import cv2
            cap = cv2.VideoCapture(processed_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except:
            pass
        
        # Custom progress tracking during detection if we have a callback
        detection_progress_tracker = None
        if progress_callback and total_frames > 0:
            class ProgressTracker:
                def __init__(self):
                    self.frames_processed = 0
                
                def update(self, frame_idx):
                    self.frames_processed = max(self.frames_processed, frame_idx)
                    progress = min(1.0, self.frames_processed / total_frames)
                    progress_callback('detection', progress)
            
            detection_progress_tracker = ProgressTracker()
            
        speaker_data = detector.process_video(
            processed_video_path, 
            task_temp_dir,
            progress_tracker=detection_progress_tracker
        )
        
        # Add FPS to speaker data for transcription alignment
        speaker_data['fps'] = fps
        
        # Mark detection as complete if we have a callback
        if progress_callback:
            progress_callback('detection', 1.0)
        
        if not speaker_data or 'tracks' not in speaker_data or not speaker_data['tracks']:
            log.warning("No speaker tracks detected in the video")
            return
            
        # Generate cropped and annotated video
        log.info("Generating cropped video...")
        
        # Progress tracking for cropping stage
        cropping_progress_tracker = None
        if progress_callback:
            class CroppingProgressTracker:
                def __init__(self):
                    self.frames_processed = 0
                
                def update(self, frame_idx):
                    self.frames_processed = max(self.frames_processed, frame_idx)
                    progress = min(1.0, self.frames_processed / total_frames)
                    progress_callback('cropping', progress)
            
            cropping_progress_tracker = CroppingProgressTracker()
        
        cropper = SmartCropper(target_ratio=target_ratio)
        cropper.process_video(
            processed_video_path, 
            output_path, 
            speaker_data, 
            min_score,
            crop_smoothness=crop_smoothness,
            progress_tracker=cropping_progress_tracker
        )

        log.info("Generating annotated video...")
        annotated_output = output_path.replace('.mp4', '_annotated.mp4')
        annotator = VideoAnnotator(target_ratio=target_ratio)
        annotator.process_video(processed_video_path, annotated_output, speaker_data, min_score=min_score, crop_smoothness=crop_smoothness)
        
        # Generate transcription if enabled
        transcript_data = None
        if enable_transcription:
            log.info("Generating transcription with Whisper...")
            if progress_callback:
                progress_callback('transcription', 0.0)
                
            # Create transcription directory
            transcript_dir = os.path.join(os.path.dirname(output_path), "transcript")
            os.makedirs(transcript_dir, exist_ok=True)
            
            # Initialize transcriber and process video
            transcriber = VideoTranscriber(model_size=whisper_model)
            transcript_data = transcriber.transcribe_video(
                processed_video_path,
                speaker_data,
                transcript_dir,
                task_id
            )
            
            if progress_callback:
                progress_callback('transcription', 1.0)
                
            log.info(f"Transcription generated: {transcript_data['json_path']}")

        # Complete the progress
        if progress_callback:
            progress_callback('cropping', 1.0)

        log.info(f"Demo videos generated successfully at:")
        log.info(f"  - Cropped: {output_path}")
        log.info(f"  - Annotated: {annotated_output}")
        
        # Return the paths and data
        result = {
            "output_path": output_path,
            "annotated_path": annotated_output,
            "speaker_data": speaker_data
        }
        
        if transcript_data:
            result["transcript_data"] = transcript_data
            
        return result
        
    except Exception as e:
        log.error(f"Error in demo speaker diarization: {str(e)}")
        raise
    finally:
        # Cleanup
        if os.path.exists(concatenated_video_path):
            os.remove(concatenated_video_path)


if __name__ == '__main__':
    task_id = 'demo-speaker-diarization-task'
    video_url = 'https://api.forzasys.com/allsvenskan/playlist.m3u8/12160:0:40000/Manifest.m3u8'
    output_path = f'task/{task_id}/demo_speaker_diarization_output.mp4'

    demo_speaker_diarization(task_id, video_url, output_path, target_ratio=(9, 16), min_score=0.4, crop_smoothness=0.2)