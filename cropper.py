import cv2
import numpy as np
import subprocess
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import platform
import psutil
import time

# Set OpenCV to use all available CPU cores
cv2.setNumThreads(multiprocessing.cpu_count())

# OpenCV is compiled without CUDA support in this environment
CUDA_AVAILABLE = False
print("Using CPU-only mode for processing")

class VideoProcessor:
    @staticmethod
    def extract_audio(input_video, output_audio):
        """Extract audio from video file"""
        print("Extracting audio...")
        extract_audio_cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn", "-acodec", "copy",
            output_audio
        ]
        subprocess.run(extract_audio_cmd, check=True)
        return output_audio
    
    @staticmethod
    def encode_final_video(temp_video, audio_file, output_video, thread_count):
        """Encode final video with audio using ffmpeg"""
        print(f"Encoding final video: {output_video}")
        
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", audio_file,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", 
            "-crf", "23",
            "-preset", "faster",  
            "-pix_fmt", "yuv420p",  
            "-c:a", "aac",
            "-af", "aresample=async=1000",  
            "-vsync", "cfr",  
            "-shortest",
            "-movflags", "+faststart",  
            "-threads", str(thread_count), 
            output_video
        ]
        subprocess.run(reencode_cmd, check=True)
        return output_video
    
    @staticmethod
    def calculate_batch_size(frame_width, frame_height):
        """Calculate optimal batch size based on available memory - cross-platform compatible"""
        try:
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024**3)
            available_memory_gb = memory_info.available / (1024**3)
            
            usable_memory_gb = min(total_memory_gb / 4, available_memory_gb / 2)
            
            # Calculate frame size in MB
            frame_size_mb = (frame_width * frame_height * 3) / (1024**2)
            
            # Calculate batch size based on available memory
            max_frames_in_memory = int(usable_memory_gb * 1024 / frame_size_mb)
            batch_size = min(200, max(50, max_frames_in_memory))  # Between 50 and 200
            
            print(f"System memory: {total_memory_gb:.1f} GB (Available: {available_memory_gb:.1f} GB)")
            print(f"Frame size: {frame_size_mb:.1f} MB, using batch size of {batch_size} frames")
            
        except Exception as e:
            # Fallback if memory detection fails
            print(f"Memory detection error: {e}, using default batch size")
            batch_size = 100
            
        return batch_size
    
    @staticmethod
    def get_video_info(input_video):
        """Get video properties in a consistent format"""
        cap = cv2.VideoCapture(input_video)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        print(f"Video properties - Width: {info['width']}, Height: {info['height']}, FPS: {info['fps']}")
        return info
    
    @staticmethod
    def create_temp_files(prefix):
        """Create temp filenames with consistent naming - platform independent"""
        temp_dir = os.path.abspath(os.path.join(".", "temp"))
        
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_video = os.path.join(temp_dir, f"temp_{prefix}_no_audio.mp4")
        audio_file = os.path.join(temp_dir, f"temp_{prefix}_audio.aac")
        
        return temp_video, audio_file

class SmartCropper:
    def __init__(self, target_ratio=(9, 16)):
        self.target_ratio = target_ratio[0] / target_ratio[1]
        self.output_width = target_ratio[0] * 100
        self.output_height = target_ratio[1] * 100
        self.is_phone_ratio = abs(self.target_ratio - (9/16)) < 0.1 
        
        # Try to load iPhone frame if we're using 9:16 ratio
        self.phone_frame = None
        if self.is_phone_ratio:
            try:
                self.phone_frame = cv2.imread('iphone.png', cv2.IMREAD_UNCHANGED)
                if self.phone_frame is None:
                    print("Warning: Could not load phone frame image.")
            except Exception as e:
                print(f"Error loading iPhone frame: {e}")
        
        # Speaker tracking
        self.current_speaker_track_id = None
        self.speaker_consistency_frames = 5
        self.potential_new_speaker = None
        self.potential_speaker_frames = 0
        
        # Movement thresholds
        self.movement_threshold = 30  # Pixel distance threshold for movement
        self.sticky_center = None     # Last stable position
        
        # Position smoothing for same speaker
        self.position_history = []
        self.history_size = 10
        self.smoothing_alpha = 0.2
        self.last_center = None
        
        # Thread count for parallel processing - limit to physical cores for better performance
        self.thread_count = max(1, min(multiprocessing.cpu_count(), 16))  # Cap at 16 threads
        print(f"Utilizing {self.thread_count} CPU threads for processing")

    def _calculate_movement(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        if pos1 is None or pos2 is None:
            return float('inf')
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) ** 0.5

    def _get_speaker_center(self, speaker_data, frame_idx, width, height, min_score):
        """Find the center of the active speaker and their track ID"""
        max_score = -float('inf')
        best_center = None
        best_track_id = None

        try:
            for track in speaker_data['tracks']:
                frame_indices = track['frame_indices']
                scores = track['scores']
                bboxes = track['bbox']
                track_id = track.get('track_id', None)

                if min(frame_indices) <= frame_idx <= max(frame_indices):
                    closest_idx = min(range(len(frame_indices)), 
                                  key=lambda i: abs(frame_indices[i] - frame_idx))
                    
                    score_idx = min(closest_idx, len(scores) - 1)
                    score = scores[score_idx]
                    
                    if score > max_score and score > min_score:
                        max_score = score
                        bbox = bboxes[closest_idx]
                        best_center = (
                            int((bbox[0] + bbox[2]) / 2),
                            int((bbox[1] + bbox[3]) / 2)
                        )
                        best_track_id = track_id

        except Exception as e:
            print(f"Error in speaker center calculation: {str(e)}")
            return None, None

        return best_center, best_track_id

    def _handle_speaker_movement(self, new_center):
        """Handle speaker movement with sticky positioning"""
        if self.sticky_center is None:
            self.sticky_center = new_center
            return new_center

        # Calculate movement from sticky position
        movement = self._calculate_movement(new_center, self.sticky_center)

        if movement < self.movement_threshold:
            # Small movement - maintain sticky position
            return self.sticky_center
        else:
            # Large movement - update sticky position with smoothing
            if not self.position_history:
                smoothed_center = new_center
            else:
                # Apply smoothing only for large movements
                smoothed_center = (
                    int(self.smoothing_alpha * new_center[0] + (1 - self.smoothing_alpha) * self.sticky_center[0]),
                    int(self.smoothing_alpha * new_center[1] + (1 - self.smoothing_alpha) * self.sticky_center[1])
                )

            self.sticky_center = smoothed_center
            return smoothed_center

    def _get_crop_window(self, frame_width, frame_height, center_x, center_y):
        """Calculate crop window dimensions"""
        if self.target_ratio < frame_width / frame_height:
            crop_height = frame_height
            crop_width = int(crop_height * self.target_ratio)
        else:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_ratio)

        # Add small margin
        margin = int(crop_width * 0.1)
        crop_width = min(crop_width + margin, frame_width)

        # Center around the speaker
        x = int(center_x - crop_width / 2)
        y = int(center_y - crop_height / 2)

        # Ensure within frame boundaries
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))

        return x, y, crop_width, crop_height

    def _get_default_center(self, frame_width, frame_height):
        """Get the default center position"""
        return (frame_width // 2, frame_height // 2)

    def _apply_phone_overlay(self, frame):
        """Apply iPhone overlay to the video frame if using 9:16 ratio"""
        if not self.is_phone_ratio or self.phone_frame is None:
            return frame
        
        try:
            # Get dimensions
            frame_h, frame_w = frame.shape[:2]
            phone_h, phone_w = self.phone_frame.shape[:2]
            
            # Calculate scale to fit phone frame around the video
            # Make it slightly larger than the video
            scale_factor = 1.05
            scale_w = (frame_w * scale_factor) / phone_w
            scale_h = (frame_h * scale_factor) / phone_h
            scale = min(scale_w, scale_h)
            
            # Resize phone frame
            new_w = int(phone_w * scale)
            new_h = int(phone_h * scale)
            resized_phone = cv2.resize(self.phone_frame, (new_w, new_h))
            
            # Center phone frame around the video
            pos_x = (frame_w - new_w) // 2
            pos_y = (frame_h - new_h) // 2
            
            # Create output frame with alpha channel
            result = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
            result[:, :, :3] = frame
            result[:, :, 3] = 255  # Full opacity
            
            # Calculate ROI and ensure it's within bounds
            if pos_x < 0:
                src_x = -pos_x
                dst_x = 0
                roi_w = min(new_w - src_x, frame_w)
            else:
                src_x = 0
                dst_x = pos_x
                roi_w = min(new_w, frame_w - pos_x)
                
            if pos_y < 0:
                src_y = -pos_y
                dst_y = 0
                roi_h = min(new_h - src_y, frame_h)
            else:
                src_y = 0
                dst_y = pos_y
                roi_h = min(new_h, frame_h - pos_y)
            
            # Skip if ROI is invalid
            if roi_w <= 0 or roi_h <= 0:
                return frame
                
            # Get alpha channel from phone frame
            phone_alpha = resized_phone[src_y:src_y+roi_h, src_x:src_x+roi_w, 3] / 255.0
            
            # Apply alpha blending
            for c in range(3):  # RGB channels
                result[dst_y:dst_y+roi_h, dst_x:dst_x+roi_w, c] = (
                    (1 - phone_alpha) * frame[dst_y:dst_y+roi_h, dst_x:dst_x+roi_w, c] + 
                    phone_alpha * resized_phone[src_y:src_y+roi_h, src_x:src_x+roi_w, c]
                )
            
            # Convert back to BGR
            return result[:, :, :3]
            
        except Exception as e:
            print(f"Error applying phone overlay: {str(e)}")
            return frame

    def _process_frame(self, frame_data):
        """Process a single frame in a worker thread"""
        frame_idx, frame, speaker_data, frame_width, frame_height, min_score = frame_data
        
        # Get speaker center and track ID
        raw_center, track_id = self._get_speaker_center(
            speaker_data, frame_idx, frame_width, frame_height, min_score
        )

        # Handle speaker change or maintain current position
        if raw_center is not None and track_id is not None:
            if self.current_speaker_track_id is None:
                # First speaker detected
                self.current_speaker_track_id = track_id
                center = raw_center
                self.sticky_center = raw_center
            elif track_id != self.current_speaker_track_id:
                # New speaker - instant change
                self.current_speaker_track_id = track_id
                center = raw_center
                self.sticky_center = raw_center  # Reset sticky position for new speaker
            else:
                # Same speaker - handle movement with threshold
                center = self._handle_speaker_movement(raw_center)
        else:
            # No speaker detected, use last known position or default
            center = self.sticky_center if self.sticky_center else self._get_default_center(frame_width, frame_height)

        # Get and apply crop window
        x, y, crop_w, crop_h = self._get_crop_window(frame_width, frame_height, center[0], center[1])
        cropped = frame[y:y+crop_h, x:x+crop_w]
        
        # Standard CPU resize
        resized = cv2.resize(cropped, (self.output_width, self.output_height))
        
        # Apply phone overlay if using 9:16 ratio
        if self.is_phone_ratio:
            resized = self._apply_phone_overlay(resized)
            
        return frame_idx, resized

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5, crop_smoothness=0.2, progress_tracker=None, cached_audio=None):
        # Update smoothing factor with the user's choice
        self.smoothing_alpha = crop_smoothness
        
        # Make sure input file exists
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        
        # Get video info using shared utility
        video_info = VideoProcessor.get_video_info(input_video)
        frame_width = video_info['width']
        frame_height = video_info['height']
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        # Get exact fps from ffprobe for more accurate frame timing
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_video
            ]
            fps_str = subprocess.check_output(probe_cmd, text=True).strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                exact_fps = num / den if den != 0 else fps
                print(f"Exact FPS from source: {exact_fps}")
                fps = exact_fps
        except Exception as e:
            print(f"Warning: Could not get exact FPS from source: {e}")
        
        # Setup temporary files
        temp_video, audio_file = VideoProcessor.create_temp_files("cropped")
        
        # Standard software encoding - ensure we use the exact same FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (self.output_width, self.output_height))
        
        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            return

        # Get optimal batch size based on memory
        batch_size = VideoProcessor.calculate_batch_size(frame_width, frame_height)

        # Read and process frames in batches to avoid memory issues
        frame_idx = 0
        total_processed = 0
        frames_skipped = 0
        
        cap = cv2.VideoCapture(input_video)
        
        # Set opencv capture to use exact fps
        cap.set(cv2.CAP_PROP_FPS, fps)
        start_time = time.time()
        
        while True:
            # Read batch of frames
            frames_to_process = []
            batch_start_idx = frame_idx
            
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_to_process.append((frame_idx, frame, speaker_data, frame_width, frame_height, min_score))
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"Read {frame_idx}/{total_frames} frames")
            
            if not frames_to_process:
                break  # No more frames to process
                
            # Process the batch in parallel
            print(f"Processing batch of {len(frames_to_process)} frames...")
            processed_frames = {}
            
            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                future_to_frame = {executor.submit(self._process_frame, frame_data): frame_data for frame_data in frames_to_process}
                
                for future in as_completed(future_to_frame):
                    try:
                        idx, processed_frame = future.result()
                        processed_frames[idx] = processed_frame
                        
                        # Update progress tracker if provided
                        if progress_tracker and idx % 10 == 0:
                            progress_tracker.update(idx)
                    except Exception as exc:
                        print(f'Frame processing generated an exception: {exc}')
            
            # Write processed frames in order
            for i in range(batch_start_idx, batch_start_idx + len(frames_to_process)):
                if i in processed_frames:
                    out.write(processed_frames[i])
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        elapsed = time.time() - start_time
                        fps_achieved = total_processed / elapsed if elapsed > 0 else 0
                        print(f"Processed {total_processed}/{total_frames} frames (Current FPS: {fps_achieved:.2f})")
                else:
                    frames_skipped += 1
            
            # Clear memory
            processed_frames.clear()
            frames_to_process.clear()
        
        end_time = time.time()
        total_time = end_time - start_time
        achieved_fps = total_processed / total_time if total_time > 0 else 0
        
        print(f"Video processing stats:")
        print(f"- Total frames processed: {total_processed}")
        print(f"- Total frames skipped: {frames_skipped}")
        print(f"- Processing time: {total_time:.2f}s")
        print(f"- Achieved FPS: {achieved_fps:.2f}")
        print(f"- Target FPS: {fps:.2f}")
        
        cap.release()
        out.release()

        # Extract audio only if we don't already have it cached
        if cached_audio and os.path.exists(cached_audio):
            print(f"Using cached audio from: {cached_audio}")
            audio_file = cached_audio
        else:
            audio_file = VideoProcessor.extract_audio(input_video, audio_file)
        
        # Make sure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_video))
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the final encoded video using shared encoder
        VideoProcessor.encode_final_video(temp_video, audio_file, output_video, self.thread_count)

        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        # Don't remove audio file if it was cached
        if (not cached_audio or cached_audio != audio_file) and os.path.exists(audio_file):
            os.remove(audio_file)

        print(f"Processing complete: {output_video}")
        return audio_file  # Return audio file path in case it's needed for caching

class VideoAnnotator:
    def __init__(self, target_ratio=(9, 16)):
        self.target_ratio = target_ratio[0] / target_ratio[1]
        
        self.phone_frame = cv2.imread('iphone.png', cv2.IMREAD_UNCHANGED)
        if self.phone_frame is None:
            print("Warning: Could not load phone frame image. Using simple frame instead.")

        self.colors = {
            'active_speaker': (0, 255, 0),    # Green
            'silent_speaker': (0, 0, 255),    # Red
            'crop_overlay': (0, 0, 0, 128),   
            'bbox_thickness': 2,
            'text_size': 0.5
        }

        # Speaker tracking
        self.current_speaker_track_id = None
        self.speaker_consistency_frames = 5
        self.potential_new_speaker = None
        self.potential_speaker_frames = 0
        
        # Movement thresholds and sticky position
        self.movement_threshold = 30  # Pixel distance threshold for movement
        self.sticky_center = None     # Last stable position
        self.smoothing_alpha = 0.2    # Smoothing factor for large movements
        
        # Dimension tracking for accurate bounding box scaling
        self.processing_width = None
        self.processing_height = None
        self.original_width = None
        self.original_height = None
        
        # Thread count for parallel processing - limit to physical cores for better performance
        self.thread_count = max(1, min(multiprocessing.cpu_count(), 16))  # Cap at 16 threads
        print(f"Utilizing {self.thread_count} CPU threads for annotation")

    def _calculate_movement(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        if pos1 is None or pos2 is None:
            return float('inf')
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) ** 0.5

    def _get_speaker_data(self, speaker_data, frame_idx, width, height, min_score):
        """Get all speakers and identify active ones"""
        active_speakers = []
        silent_speakers = []
        best_speaker = None
        best_score = -float('inf')

        try:
            for track in speaker_data['tracks']:
                frame_indices = track['frame_indices']
                scores = track['scores']
                bboxes = track['bbox']
                track_id = track.get('track_id', None)

                if min(frame_indices) <= frame_idx <= max(frame_indices):
                    closest_idx = min(range(len(frame_indices)), 
                                  key=lambda i: abs(frame_indices[i] - frame_idx))
                    
                    score_idx = min(closest_idx, len(scores) - 1)
                    score = scores[score_idx]
                    bbox = bboxes[closest_idx].copy()  # Create a copy to avoid modifying original data
                    
                    # Get original dimensions from track or from class variables
                    orig_width = track.get('orig_width', self.original_width)
                    orig_height = track.get('orig_height', self.original_height)
                    
                    # Apply correct scaling chain:
                    # 1. Scale from original detection dimensions to processing dimensions (if needed)
                    # 2. Scale from processing dimensions to output dimensions
                    
                    # First scale: If we have original detection dimensions different from processing
                    if orig_width is not None and orig_height is not None:
                        # Original detection dimensions to processing dimensions
                        scale_x1 = self.processing_width / orig_width
                        scale_y1 = self.processing_height / orig_height
                        
                        # Apply first scaling
                        bbox[0] = bbox[0] * scale_x1
                        bbox[1] = bbox[1] * scale_y1
                        bbox[2] = bbox[2] * scale_x1
                        bbox[3] = bbox[3] * scale_y1
                    
                    # Second scale: From processing to output dimensions
                    scale_x2 = width / self.processing_width
                    scale_y2 = height / self.processing_height
                    
                    # Apply second scaling
                    bbox[0] = int(bbox[0] * scale_x2)
                    bbox[1] = int(bbox[1] * scale_y2)
                    bbox[2] = int(bbox[2] * scale_x2)
                    bbox[3] = int(bbox[3] * scale_y2)
                    
                    # Ensure bounding box is within frame limits
                    bbox[0] = max(0, min(bbox[0], width-1))
                    bbox[1] = max(0, min(bbox[1], height-1))
                    bbox[2] = max(0, min(bbox[2], width-1))
                    bbox[3] = max(0, min(bbox[3], height-1))
                    
                    # Sanity check: ensure width and height are positive
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    
                    speaker_info = {
                        'bbox': bbox,
                        'score': score,
                        'track_id': track_id,
                        'center': (
                            int((bbox[0] + bbox[2]) / 2),
                            int((bbox[1] + bbox[3]) / 2)
                        )
                    }
                    
                    if score > min_score:
                        active_speakers.append(speaker_info)
                        if score > best_score:
                            best_score = score
                            best_speaker = speaker_info
                    else:
                        silent_speakers.append(speaker_info)

        except Exception as e:
            print(f"Error in speaker data extraction: {str(e)}")

        return active_speakers, silent_speakers, best_speaker

    def _handle_speaker_movement(self, new_center):
        """Handle speaker movement with sticky positioning"""
        if self.sticky_center is None:
            self.sticky_center = new_center
            return new_center

        # Calculate movement from sticky position
        movement = self._calculate_movement(new_center, self.sticky_center)

        if movement < self.movement_threshold:
            # Small movement - maintain sticky position
            return self.sticky_center
        else:
            # Large movement - update sticky position with smoothing
            smoothed_center = (
                int(self.smoothing_alpha * new_center[0] + (1 - self.smoothing_alpha) * self.sticky_center[0]),
                int(self.smoothing_alpha * new_center[1] + (1 - self.smoothing_alpha) * self.sticky_center[1])
            )
            self.sticky_center = smoothed_center
            return smoothed_center

    def _calculate_crop_window(self, frame_width, frame_height, center):
        """Calculate dynamic crop window based on speaker position"""
        if self.target_ratio < frame_width / frame_height:
            crop_height = frame_height
            crop_width = int(crop_height * self.target_ratio)
        else:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_ratio)

        x = int(center[0] - crop_width / 2)
        y = int(center[1] - crop_height / 2)

        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))

        return x, y, crop_width, crop_height

    def _draw_speaker_box(self, frame, bbox, is_active=True, track_id=None, is_main_speaker=False):
        """Draw bounding box and score for a speaker"""
        x1, y1, x2, y2 = map(int, bbox)
        base_color = self.colors['active_speaker'] if is_active else self.colors['silent_speaker']
        
        thickness = self.colors['bbox_thickness'] * 2 if is_main_speaker else self.colors['bbox_thickness']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, thickness)
        
        # Prepare text
        status = "SPEAKING" if is_active else "SILENT"
        if is_main_speaker:
            status += " (TRACKED)"
        text = f"ID: {track_id} ({status})"
        
        # Get text size
        font_scale = self.colors['text_size'] * 1.2  # Slightly larger
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        
        # Draw text background
        text_bg_color = (0, 0, 0)
        text_y = max(0, y1 - 10)
        cv2.rectangle(
            frame, 
            (x1, text_y - text_height - 5),
            (x1 + text_width, text_y + 5),
            text_bg_color, 
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            frame, 
            text, 
            (x1, text_y), 
            font,
            font_scale, 
            base_color, 
            2
        )

    def _overlay_phone_frame(self, frame, crop_x, crop_y, crop_w, crop_h):
        """Apply transparency outside the crop area and overlay phone frame"""
        # Create a mask for the cropped area
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = 255

        # Add semi-transparent overlay to areas outside the crop window
        overlay = frame.copy()
        inverse_mask = cv2.bitwise_not(mask)
        
        # Apply dimming effect to areas outside the crop window
        frame_with_alpha = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        frame_with_alpha[:, :, 0:3] = frame
        frame_with_alpha[:, :, 3] = 255
        
        # Dim the areas outside the crop window
        frame_with_alpha[inverse_mask > 0, 0:3] = frame_with_alpha[inverse_mask > 0, 0:3] * 0.5
        
        # Convert back to BGR
        frame[:] = frame_with_alpha[:, :, 0:3]
        
        # Draw a white rectangle around the crop area
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), 
                     (255, 255, 255), 2)
        
        # Calculate aspect ratio to determine if we're working with 9:16
        aspect_ratio = crop_w / crop_h if crop_h > 0 else 0
        is_phone_ratio = abs(aspect_ratio - (9/16)) < 0.1  # Check if it's close to 9:16
        
        # Add appropriate text label based on the aspect ratio
        if is_phone_ratio:
            aspect_label = "9:16 Mobile Crop"
        else:
            aspect_label = f"{crop_w/crop_h:.2f} Aspect Ratio Crop"
            
        cv2.putText(frame, aspect_label, (crop_x + 10, crop_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Only apply iPhone frame for 9:16 aspect ratio
        if is_phone_ratio:
            try:
                iphone_frame = cv2.imread("iphone.png", cv2.IMREAD_UNCHANGED)
                if iphone_frame is not None and iphone_frame.shape[2] == 4:  # Has alpha channel
                    # Resize phone frame to match crop window
                    phone_h, phone_w = iphone_frame.shape[:2]
                    scale = min(crop_w / phone_w, crop_h / phone_h) * 1.1  # Slightly larger
                    new_w, new_h = int(phone_w * scale), int(phone_h * scale)
                    
                    # Center the phone frame around the crop area
                    phone_x = crop_x + (crop_w - new_w) // 2
                    phone_y = crop_y + (crop_h - new_h) // 2
                    
                    if phone_x >= 0 and phone_y >= 0 and phone_x + new_w <= frame.shape[1] and phone_y + new_h <= frame.shape[0]:
                        resized_phone = cv2.resize(iphone_frame, (new_w, new_h))
                        
                        # Extract alpha channel
                        alpha = resized_phone[:, :, 3] / 255.0
                        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                        
                        # Get ROI from the frame
                        roi = frame[phone_y:phone_y+new_h, phone_x:phone_x+new_w]
                        
                        # Blend with alpha
                        blended = (1 - alpha) * roi + alpha * resized_phone[:, :, :3]
                        frame[phone_y:phone_y+new_h, phone_x:phone_x+new_w] = blended
            except Exception as e:
                # If phone frame overlay fails, just leave the rectangle
                print(f"Error applying phone frame: {e}")

    def _process_annotation_frame(self, frame_data):
        """Process a single annotation frame in a worker thread"""
        frame_idx, frame, speaker_data, output_width, output_height, min_score = frame_data
        
        # Standard CPU resize
        frame = cv2.resize(frame, (output_width, output_height))
        
        # Get all active speakers and their data for this frame
        active_speakers, silent_speakers, best_speaker = self._get_speaker_data(
            speaker_data, frame_idx, output_width, output_height, min_score
        )
        
        # Find the most likely speaker
        max_score = -1
        main_speaker = None
        main_speaker_id = None
        for speaker in active_speakers:
            if speaker['score'] > max_score:
                max_score = speaker['score']
                main_speaker_id = speaker['track_id']
                main_speaker = speaker
        
        # Draw boxes around all tracked faces
        for speaker in active_speakers:
            is_main = (speaker['track_id'] == main_speaker_id)
            self._draw_speaker_box(frame, speaker['bbox'], speaker['score'] > min_score, speaker['track_id'], is_main)
        
        # Draw silent speakers
        for speaker in silent_speakers:
            self._draw_speaker_box(frame, speaker['bbox'], False, speaker['track_id'], False)
        
        # Calculate crop window
        if main_speaker is not None:
            center = self._handle_speaker_movement(main_speaker['center'])
            
            # Add crop window overlay
            crop_x, crop_y, crop_w, crop_h = self._calculate_crop_window(output_width, output_height, center)
            self._overlay_phone_frame(frame, crop_x, crop_y, crop_w, crop_h)
        
        return frame_idx, frame

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5, crop_smoothness=0.2, progress_tracker=None, cached_audio=None):
        # Update smoothing factor with the user's choice
        self.smoothing_alpha = crop_smoothness
        
        # Make sure input file exists
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
            
        # Load the iPhone frame image
        try:
            iphone_frame = cv2.imread("iphone.png", cv2.IMREAD_UNCHANGED)
            if iphone_frame is None:
                print("Warning: Could not load iPhone frame image. Proceeding without overlay.")
        except Exception as e:
            print(f"Error loading iPhone frame: {e}")
            iphone_frame = None
            
        # Get video info using shared utility
        video_info = VideoProcessor.get_video_info(input_video)
        frame_width = video_info['width']
        frame_height = video_info['height']
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        # Get exact fps from ffprobe for more accurate frame timing
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_video
            ]
            fps_str = subprocess.check_output(probe_cmd, text=True).strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                exact_fps = num / den if den != 0 else fps
                print(f"Exact FPS from source: {exact_fps}")
                fps = exact_fps
        except Exception as e:
            print(f"Warning: Could not get exact FPS from source: {e}")
        
        # Store dimensions for accurate bounding box mapping
        self.processing_width = frame_width
        self.processing_height = frame_height
        
        # Get original dimensions from speaker data if available
        self.original_width = None
        self.original_height = None
        if speaker_data and 'tracks' in speaker_data and speaker_data['tracks']:
            for track in speaker_data['tracks']:
                if 'orig_width' in track and 'orig_height' in track:
                    self.original_width = track['orig_width']
                    self.original_height = track['orig_height']
                    print(f"Annotator - Original detection dimensions: {self.original_width}x{self.original_height}")
                    break
        
        # Calculate output video dimensions based on original dimensions
        if frame_width > frame_height:
            output_width = 1280
            output_height = int(output_width * frame_height / frame_width)
        else:
            output_height = 720
            output_width = int(output_height * frame_width / frame_height)
        
        print(f"Annotator - Output dimensions: {output_width}x{output_height}")
            
        # Setup temporary files
        temp_video, audio_file = VideoProcessor.create_temp_files("annotated")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            return

        # Get optimal batch size based on memory
        batch_size = VideoProcessor.calculate_batch_size(frame_width, frame_height)

        # Read and process frames in batches
        frame_idx = 0
        total_processed = 0
        frames_skipped = 0
        
        cap = cv2.VideoCapture(input_video)
        
        # Set opencv capture to use exact fps
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Track frame processing times for stats
        start_time = time.time()
        
        while True:
            # Read batch of frames
            frames_to_process = []
            batch_start_idx = frame_idx
            
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_to_process.append((frame_idx, frame, speaker_data, output_width, output_height, min_score))
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"Read {frame_idx}/{total_frames} frames for annotation")
            
            if not frames_to_process:
                break 
                
            # Process the batch in parallel
            print(f"Annotating batch of {len(frames_to_process)} frames...")
            processed_frames = {}
            
            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                future_to_frame = {executor.submit(self._process_annotation_frame, frame_data): frame_data for frame_data in frames_to_process}
                
                for future in as_completed(future_to_frame):
                    try:
                        idx, processed_frame = future.result()
                        processed_frames[idx] = processed_frame
                        
                        # Update progress tracker if provided
                        if progress_tracker and idx % 10 == 0:
                            progress_tracker.update(idx)
                    except Exception as exc:
                        print(f'Frame annotation generated an exception: {exc}')
            
            # Write processed frames in order
            for i in range(batch_start_idx, batch_start_idx + len(frames_to_process)):
                if i in processed_frames:
                    out.write(processed_frames[i])
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        elapsed = time.time() - start_time
                        fps_achieved = total_processed / elapsed if elapsed > 0 else 0
                        print(f"Annotated {total_processed}/{total_frames} frames (Current FPS: {fps_achieved:.2f})")
                else:
                    frames_skipped += 1
            
            # Clear memory
            processed_frames.clear()
            frames_to_process.clear()
                
        # Clean up
        end_time = time.time()
        total_time = end_time - start_time
        achieved_fps = total_processed / total_time if total_time > 0 else 0
        
        print(f"Annotation stats:")
        print(f"- Total frames processed: {total_processed}")
        print(f"- Total frames skipped: {frames_skipped}")
        print(f"- Processing time: {total_time:.2f}s")
        print(f"- Achieved FPS: {achieved_fps:.2f}")
        print(f"- Target FPS: {fps:.2f}")
        
        cap.release()
        out.release()

        if cached_audio and os.path.exists(cached_audio):
            print(f"Using cached audio from: {cached_audio}")
            audio_file = cached_audio
        else:
            audio_file = VideoProcessor.extract_audio(input_video, audio_file)
        
        output_dir = os.path.dirname(os.path.abspath(output_video))
        os.makedirs(output_dir, exist_ok=True)

        VideoProcessor.encode_final_video(temp_video, audio_file, output_video, self.thread_count)

        # Cleanup
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        if (not cached_audio or cached_audio != audio_file) and os.path.exists(audio_file):
            os.remove(audio_file)

        print(f"Annotated video saved to {output_video}")
        return audio_file

def process_videos(input_video, output_cropped, output_annotated, speaker_data, min_score=0.5, crop_smoothness=0.2, progress_tracker=None):
    """Process both cropped and annotated versions in one pass, sharing resources"""
    print(f"Processing input video: {input_video}")
    print(f"Generating cropped output: {output_cropped}")
    print(f"Generating annotated output: {output_annotated}")
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_cropped)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_annotated)), exist_ok=True)
    
    # Initialize processors
    cropper = SmartCropper()
    annotator = VideoAnnotator()
    
    # Create a dedicated temp directory
    temp_dir = os.path.abspath(os.path.join(".", "temp"))
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("== STEP 1: Generating cropped version ==")
        audio_file = cropper.process_video(
            input_video, 
            output_cropped, 
            speaker_data, 
            min_score, 
            crop_smoothness, 
            progress_tracker
        )
        
        print("== STEP 2: Generating annotated version ==")
        annotator.process_video(
            input_video, 
            output_annotated, 
            speaker_data, 
            min_score, 
            crop_smoothness, 
            progress_tracker,
            cached_audio=audio_file 
        )
        
        # Clean up shared resources
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        print("Both videos processed successfully!")
        return output_cropped, output_annotated
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        # Clean up any temporary files on error
        for file in os.listdir(temp_dir):
            if file.startswith("temp_"):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
        raise