import os
import logging
from cropper import SmartCropper, VideoAnnotator
from components.talknet_modules.talknet_inference import SpeakerDetector
from utils import VideoStore

log = logging.getLogger('aiproducer')

def demo_speaker_diarization(task_id: str, video_url: str, output_path: str, target_ratio=(9, 16), min_score=0.4):
    """Run speaker diarization demo: process video and generate cropped output."""
    try:
        # Create task directories
        task_temp_dir = os.path.join("task", task_id, "temp")
        os.makedirs(task_temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download and concatenate video
        log.info(f"Downloading video from {video_url}")
        video_store = VideoStore(task_id, video_url, os.path.join(task_temp_dir, "segments"))
        video_store.download_video()
        
        concatenated_video_path = os.path.join(task_temp_dir, "concatenated_video.ts")
        concatenated_video_path = video_store.concatenate_segments_to_file(concatenated_video_path)
        
        # Run speaker detection
        log.info("Running speaker detection...")
        detector = SpeakerDetector(model_path="weights/talknet_speaker_v1.model")
        speaker_data = detector.process_video(concatenated_video_path, task_temp_dir)
        
        if not speaker_data or 'tracks' not in speaker_data or not speaker_data['tracks']:
            log.warning("No speaker tracks detected in the video")
            return
            
        # Generate cropped and annotated video
        log.info("Generating cropped video...")
        cropper = SmartCropper(target_ratio=target_ratio)
        cropper.process_video(concatenated_video_path, output_path, speaker_data, min_score)

        log.info("Generating annotated video...")
        annotated_output = output_path.replace('.mp4', '_annotated.mp4')
        annotator = VideoAnnotator(target_ratio=target_ratio)
        annotator.process_video(concatenated_video_path, annotated_output, speaker_data, min_score=min_score)

        log.info(f"Demo videos generated successfully at:")
        log.info(f"  - Cropped: {output_path}")
        log.info(f"  - Annotated: {annotated_output}")
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

    demo_speaker_diarization(task_id, video_url, output_path, target_ratio=(9, 16), min_score=0.4)