import cv2
import numpy as np
import subprocess
import os

class SmartCropper:
    def __init__(self, target_ratio=(9, 16)):
        self.target_ratio = target_ratio[0] / target_ratio[1]
        self.output_width = target_ratio[0] * 100
        self.output_height = target_ratio[1] * 100
        
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

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5, progress_tracker=None):
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
        
        # Setup temporary files
        temp_video = os.path.abspath("temp_no_audio.mp4")
        audio_file = os.path.abspath("temp_audio.aac")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (self.output_width, self.output_height))
        
        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            cap.release()
            return

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
            resized = cv2.resize(cropped, (self.output_width, self.output_height))
            out.write(resized)

            frame_idx += 1
            
            # Update progress tracker if provided
            if progress_tracker and frame_idx % 10 == 0:
                progress_tracker.update(frame_idx)
                
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        out.release()

        # Handle audio and final encoding
        extract_audio_cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn", "-acodec", "copy",
            audio_file
        ]
        subprocess.run(extract_audio_cmd, check=True)

        # Create the final encoded video directly to the expected output path
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", audio_file,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", 
            "-crf", "23",
            "-pix_fmt", "yuv420p",  # Ensure browser compatibility
            "-c:a", "aac",
            "-movflags", "+faststart",  # Enable streaming
            output_video  # Use the exact output path provided
        ]
        subprocess.run(reencode_cmd, check=True)

        # Cleanup
        os.remove(temp_video)
        os.remove(audio_file)

        print(f"Processing complete: {output_video}")


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
        
        # Add text labels
        cv2.putText(frame, "9:16 Crop Area", (crop_x + 10, crop_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Try to use the phone frame if available
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

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5, progress_tracker=None):
        # Load the iPhone frame image
        try:
            iphone_frame = cv2.imread("iphone.png", cv2.IMREAD_UNCHANGED)
            if iphone_frame is None:
                print("Warning: Could not load iPhone frame image. Proceeding without overlay.")
        except Exception as e:
            print(f"Error loading iPhone frame: {e}")
            iphone_frame = None
            
        cap = cv2.VideoCapture(input_video)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Annotator - Input video dimensions: {frame_width}x{frame_height}")
        
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
        temp_video = os.path.abspath("temp_annotated_no_audio.mp4")
        audio_file = os.path.abspath("temp_annotated_audio.aac")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            cap.release()
            return

        frame_idx = 0
        main_speaker_id = None
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize the frame to output dimensions
            frame = cv2.resize(frame, (output_width, output_height))
            
            # Get all active speakers and their data for this frame
            active_speakers, silent_speakers, best_speaker = self._get_speaker_data(
                speaker_data, frame_idx, output_width, output_height, min_score
            )
            
            # Find the most likely speaker
            max_score = -1
            main_speaker = None
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
            
            # Write the frame
            out.write(frame)
            
            frame_idx += 1
            
            # Update progress tracker if provided
            if progress_tracker and frame_idx % 10 == 0:
                progress_tracker.update(frame_idx)
                
            if frame_idx % 100 == 0:
                print(f"Annotated {frame_idx}/{total_frames} frames")
                
        # Clean up
        cap.release()
        out.release()

        # Extract audio from original video
        extract_audio_cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vn", "-acodec", "copy",
            audio_file
        ]
        subprocess.run(extract_audio_cmd, check=True)

        # Create the final encoded video using FFmpeg with web-compatible settings
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", audio_file,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", 
            "-crf", "23",
            "-pix_fmt", "yuv420p",  # Ensure browser compatibility
            "-c:a", "aac",
            "-movflags", "+faststart",  # Enable streaming
            output_video
        ]
        subprocess.run(reencode_cmd, check=True)

        # Cleanup
        os.remove(temp_video)
        os.remove(audio_file)

        print(f"Annotated video saved to {output_video}")