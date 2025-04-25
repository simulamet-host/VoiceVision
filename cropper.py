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

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5):
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

        final_encoded = output_video.replace(".mp4", "_h264.mp4")
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", input_video,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264", 
            "-crf", "23",
            "-c:a", "copy",
            final_encoded
        ]
        subprocess.run(reencode_cmd, check=True)

        # Cleanup
        os.remove(temp_video)
        os.remove(audio_file)

        print(f"Processing complete: {final_encoded}")


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
                    bbox = bboxes[closest_idx]
                    
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
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, thickness)
        
        status = "SPEAKING" if is_active else "SILENT"
        if is_main_speaker:
            status += " (TRACKED)"
        text = f"ID: {track_id} ({status})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    self.colors['text_size'], base_color, 2)

    def _overlay_phone_frame(self, frame, crop_x, crop_y, crop_w, crop_h):
        """Apply transparency outside the crop area and overlay phone frame"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = 255

        overlay = frame.copy()
        inverse_mask = cv2.bitwise_not(mask)
        overlay_area = frame[inverse_mask > 0]
        overlay[inverse_mask > 0] = overlay_area * 0.5

        # Blend the overlay with original frame
        cv2.addWeighted(overlay, 1, frame, 0, 0, frame)

        # Add phone frame if available
        if self.phone_frame is not None:
            # Resize phone frame to match crop window
            resized_phone = cv2.resize(self.phone_frame, (crop_w, crop_h))
            
            if resized_phone.shape[2] == 4:
                frame_mask = resized_phone[:, :, 3] / 255.0
                phone_rgb = resized_phone[:, :, :3]
                
                roi = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - frame_mask) + phone_rgb[:, :, c] * frame_mask
        else:
            cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), 
                         (255, 255, 255), 2)

    def process_video(self, input_video, output_video, speaker_data, min_score=0.5):
        cap = cv2.VideoCapture(input_video)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = os.path.abspath("temp_annotated.mp4")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("Error: VideoWriter could not be opened.")
            cap.release()
            return

        frame_idx = 0
        default_center = (frame_width // 2, frame_height // 2)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get speaker information
            active_speakers, silent_speakers, best_speaker = self._get_speaker_data(
                speaker_data, frame_idx, frame_width, frame_height, min_score
            )

            if best_speaker is not None:
                if self.current_speaker_track_id is None:
                    # First speaker detected
                    self.current_speaker_track_id = best_speaker['track_id']
                    center = best_speaker['center']
                    self.sticky_center = center
                elif best_speaker['track_id'] != self.current_speaker_track_id:
                    if best_speaker['track_id'] == self.potential_new_speaker:
                        self.potential_speaker_frames += 1
                        if self.potential_speaker_frames >= self.speaker_consistency_frames:
                            # Confirmed new speaker - instant jump
                            self.current_speaker_track_id = best_speaker['track_id']
                            center = best_speaker['center']
                            self.sticky_center = center  # Reset sticky position for new speaker
                            self.potential_new_speaker = None
                            self.potential_speaker_frames = 0
                        else:
                            center = self.sticky_center if self.sticky_center else best_speaker['center']
                    else:
                        self.potential_new_speaker = best_speaker['track_id']
                        self.potential_speaker_frames = 1
                        center = self.sticky_center if self.sticky_center else best_speaker['center']
                else:
                    # Same speaker - handle movement with threshold
                    center = self._handle_speaker_movement(best_speaker['center'])
                    self.potential_new_speaker = None
                    self.potential_speaker_frames = 0
            else:
                center = self.sticky_center if self.sticky_center else default_center

            # Calculate crop window
            crop_x, crop_y, crop_w, crop_h = self._calculate_crop_window(
                frame_width, frame_height, center
            )

            # Draw silent speakers
            for speaker in silent_speakers:
                self._draw_speaker_box(
                    frame, 
                    speaker['bbox'], 
                    is_active=False, 
                    track_id=speaker['track_id']
                )

            # Draw active speakers
            for speaker in active_speakers:
                is_main = speaker['track_id'] == self.current_speaker_track_id
                self._draw_speaker_box(
                    frame, 
                    speaker['bbox'], 
                    is_active=True, 
                    track_id=speaker['track_id'],
                    is_main_speaker=is_main
                )

            # Add phone frame overlay
            self._overlay_phone_frame(frame, crop_x, crop_y, crop_w, crop_h)

            # Add frame information
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        out.release()

        final_encoded = output_video.replace(".mp4", "_annotated.mp4")
        reencode_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", input_video,    
            "-map", "0:v",        
            "-map", "1:a", 
            "-c:v", "libx264", 
            "-crf", "23",
            "-c:a", "copy",
            final_encoded
        ]
        subprocess.run(reencode_cmd, check=True)

        os.remove(temp_video)

        print(f"Annotation complete: {final_encoded}")