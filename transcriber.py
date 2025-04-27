import os
import json
import logging
import numpy as np
import torch
import whisper
from datetime import timedelta
import subprocess
import tempfile

log = logging.getLogger('aiproducer')

class VideoTranscriber:
    """
    Transcribes audio from video files using OpenAI's Whisper model.
    Can associate transcription with speaker segments for speaker diarization.
    """
    
    def __init__(self, model_size="base"):
        """
        Initialize the transcriber with specified Whisper model.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        # Lazy-load the model when needed
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Initializing VideoTranscriber with model_size={model_size} on {self.device}")
    
    def load_model(self):
        """Lazy-load the Whisper model when needed"""
        if self.model is None:
            log.info(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            log.info(f"Whisper model loaded")
        return self.model
    
    def extract_audio(self, video_path, output_audio_path=None):
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_audio_path: Path to save the extracted audio (if None, uses a temporary file)
            
        Returns:
            Path to the extracted audio file
        """
        if output_audio_path is None:
            # Create a temporary WAV file
            output_audio_path = tempfile.mktemp(suffix='.wav')
        
        try:
            # Use ffmpeg to extract audio at 16kHz (Whisper's preferred sample rate)
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ar", "16000",
                "-ac", "1",
                "-vn",
                output_audio_path
            ]
            
            log.info(f"Extracting audio from {video_path} to {output_audio_path}")
            subprocess.run(cmd, check=True)
            return output_audio_path
            
        except Exception as e:
            log.error(f"Failed to extract audio: {str(e)}")
            if os.path.exists(output_audio_path):
                os.unlink(output_audio_path)
            raise
    
    def transcribe_audio(self, audio_path, language=None):
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (if None, Whisper will auto-detect)
            
        Returns:
            Whisper transcription result object
        """
        model = self.load_model()
        
        try:
            log.info(f"Transcribing audio file: {audio_path}")
            transcription_options = {}
            
            if language:
                transcription_options["language"] = language
                
            # Transcribe with word-level timestamps
            result = model.transcribe(
                audio_path, 
                word_timestamps=True,
                **transcription_options
            )
            
            return result
        except Exception as e:
            log.error(f"Transcription failed: {str(e)}")
            raise
    
    def match_segments_to_speakers(self, transcription, speaker_data):
        """
        Match transcription segments to speaker tracks.
        
        Args:
            transcription: Whisper transcription result
            speaker_data: Speaker detection data
            
        Returns:
            Enhanced transcription with speaker IDs
        """
        if 'fps' not in speaker_data:
            log.warning("FPS not found in speaker data, using default 30fps")
            fps = 30
        else:
            fps = speaker_data['fps']
        
        tracks = speaker_data.get('tracks', [])
        if not tracks:
            log.warning("No speaker tracks found in speaker data")
            return transcription
            
        # Extract segments with timestamps
        segments = transcription.get('segments', [])
        enhanced_segments = []
        
        for segment in segments:
            # Get segment start and end times in seconds
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Calculate corresponding frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Find the most active speaker during this segment
            speaker_scores = {}
            
            for track in tracks:
                track_id = track.get('id', 0)
                timestamps = track.get('timestamps', [])
                
                # Count speaker activity during this segment
                activity_score = 0
                
                for t in timestamps:
                    frame = t.get('frame', 0)
                    score = t.get('score', 0)
                    
                    if start_frame <= frame <= end_frame:
                        activity_score += score
                
                if activity_score > 0:
                    speaker_scores[track_id] = activity_score
            
            # Assign the most active speaker (if any)
            if speaker_scores:
                most_active_speaker = max(speaker_scores.items(), key=lambda x: x[1])
                segment['speaker'] = most_active_speaker[0]
            else:
                segment['speaker'] = None
                
            enhanced_segments.append(segment)
        
        # Update transcription with enhanced segments
        transcription['segments'] = enhanced_segments
        return transcription
    
    def save_transcription(self, transcription, output_dir, task_id):
        """
        Save transcription to JSON and SRT files.
        
        Args:
            transcription: Whisper transcription result with speaker data
            output_dir: Directory to save the output files
            task_id: Task identifier for filenames
            
        Returns:
            Dictionary with paths to the saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON format (full data)
        json_path = os.path.join(output_dir, f"{task_id}_transcript.json")
        with open(json_path, 'w') as f:
            json.dump(transcription, f, indent=2)
        
        # Save SRT format (for video subtitles)
        srt_path = os.path.join(output_dir, f"{task_id}_transcript.srt")
        
        with open(srt_path, 'w') as f:
            for i, segment in enumerate(transcription.get('segments', [])):
                start = self._format_timestamp(segment.get('start', 0))
                end = self._format_timestamp(segment.get('end', 0))
                text = segment.get('text', '').strip()
                speaker = segment.get('speaker')
                
                # Add speaker information if available
                if speaker is not None:
                    text = f"[Speaker {speaker}] {text}"
                
                # Write SRT entry
                f.write(f"{i+1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
        
        # Save text format (plain text)
        txt_path = os.path.join(output_dir, f"{task_id}_transcript.txt")
        
        with open(txt_path, 'w') as f:
            for segment in transcription.get('segments', []):
                text = segment.get('text', '').strip()
                speaker = segment.get('speaker')
                
                if speaker is not None:
                    f.write(f"[Speaker {speaker}] {text}\n")
                else:
                    f.write(f"{text}\n")
        
        return {
            "json_path": json_path,
            "srt_path": srt_path,
            "txt_path": txt_path
        }
    
    def _format_timestamp(self, seconds):
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        
    def transcribe_video(self, video_path, speaker_data, output_dir, task_id):
        """
        Transcribe a video file and match segments to speakers.
        
        Args:
            video_path: Path to the video file
            speaker_data: Speaker detection data
            output_dir: Directory to save output files
            task_id: Task identifier
            
        Returns:
            Dictionary with paths to transcript files and the transcription data
        """
        try:
            # Create temp directory
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract audio from video
            audio_path = os.path.join(temp_dir, f"{task_id}_audio.wav")
            audio_path = self.extract_audio(video_path, audio_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Match transcription segments to speakers
            enhanced_transcription = self.match_segments_to_speakers(transcription, speaker_data)
            
            # Save transcription files
            output_files = self.save_transcription(enhanced_transcription, output_dir, task_id)
            
            # Add transcription data to output
            output_files["transcription"] = enhanced_transcription
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
            log.info(f"Transcription complete. Files saved to {output_dir}")
            return output_files
            
        except Exception as e:
            log.error(f"Error in transcribe_video: {str(e)}")
            raise 