import os
import torch
import glob
import subprocess
import warnings
import cv2
import numpy
import json
from scipy.io import wavfile
from scipy.interpolate import interp1d
import python_speech_features
from components.face_detection.s3fd_detector import S3FD
from components.talknet_modules.talknet_wrapper import talkNet
from utils import device
import logging

log = logging.getLogger('aiproducer')
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class SpeakerDetector:
    def __init__(self, model_path="weights/talknet_speaker_v1.model"):
        self.device = device
        self.model_path = model_path
        self.model = talkNet()
        self.model.loadParameters("weights/talknet_speaker_v1.model")
        
    def process_video(self, video_input, tmp_dir, audio_wav_path=None, progress_tracker=None):
        """
        Process video from either VideoStore object or file path
        
        Args:
            video_input: Either a VideoStore object or a file path string
            tmp_dir: Directory for temporary files
            audio_wav_path: Optional path to audio file
            progress_tracker: Optional progress tracking object with update() method
        """
        frames = []
        
        # Handle different types of video input
        if isinstance(video_input, str):
            # It's a file path
            log.info(f"Loading video from file: {video_input}")
            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                log.error(f"Could not open video file: {video_input}")
                return None
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                
                # Update progress if tracker provided
                if progress_tracker and frame_count % 10 == 0:
                    progress_tracker.update(frame_count)
                    
            cap.release()
            log.info(f"Read {frame_count} frames from file")
            
            # Extract audio if needed and none was provided
            if audio_wav_path is None:
                audio_wav_path = os.path.join(tmp_dir, "audio.wav")
                log.info(f"Extracting audio to {audio_wav_path}")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_input,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    audio_wav_path
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    log.warning(f"Failed to extract audio: {e}")
                    audio_wav_path = None
        else:
            # Assume it's a VideoStore
            log.info("Loading video from VideoStore")
            video_input.rewind()
            frame_count = 0
            while True:
                ret, frame = video_input.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                
                # Update progress if tracker provided
                if progress_tracker and frame_count % 10 == 0:
                    progress_tracker.update(frame_count)
                    
            log.info(f"Read {frame_count} frames from VideoStore")

        log.info("Processing faces")
        faces = self._inference_video_in_memory(frames, progress_tracker)
        
        # Track faces
        face_tracks = self._track_shot(faces, min_track=5)
        log.info(f"Created {len(face_tracks)} face tracks")
        
        # Process each track
        results = {
            'video_name': "in-memory video",
            'tracks': []
        }
        
        for track_idx, track in enumerate(face_tracks):
            log.debug(f"Processing track {track_idx} with {len(track['frame'])} frames")
            
            if audio_wav_path:
                scores = self._evaluate_network(
                    face_track=track,
                    frames_list=frames,         
                    audio_path=audio_wav_path
                )
            else:
                scores = []
                
            track_info = {
                'track_id': track_idx,
                'frame_indices': track['frame'].tolist(),
                'bbox': track['bbox'].tolist(),
                'scores': scores.tolist() if len(scores) > 0 else [],
            }
            results['tracks'].append(track_info)
            
            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(track_idx + frame_count)
        
        return results

    def _inference_video_in_memory(self, frames_list, progress_tracker=None):
        DET = S3FD(device=self.device)
        dets = []
        
        # Get original frame dimensions for proper scaling
        if frames_list and len(frames_list) > 0:
            sample_frame = frames_list[0]
            orig_height, orig_width = sample_frame.shape[:2]
            log.info(f"Original frame dimensions: {orig_width}x{orig_height}")
            
        for fidx, frame in enumerate(frames_list):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = DET.detect_faces(image_rgb, conf_th=0.9, scales=[0.25])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({
                    'frame': fidx, 
                    'bbox': (bbox[:-1]).tolist(), 
                    'conf': bbox[-1],
                    'orig_width': orig_width,
                    'orig_height': orig_height
                })
                
            # Update progress if tracker provided (every 10 frames)
            if progress_tracker and fidx % 10 == 0:
                progress_tracker.update(fidx)
                
        return dets

    def _bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _track_shot(self, faces, min_track=10, num_failed_det=10):
        tracks = []
        iouThres = 0.5
        
        # Extract frame dimensions from the first face if available
        orig_width = None
        orig_height = None
        for frame_faces in faces:
            for face in frame_faces:
                if 'orig_width' in face and 'orig_height' in face:
                    orig_width = face['orig_width']
                    orig_height = face['orig_height']
                    break
            if orig_width is not None:
                break
                
        while True:
            track = []
            for frameFaces in faces:
                for face in frameFaces:
                    if track == []:
                        track.append(face)
                        frameFaces.remove(face)
                    elif face['frame'] - track[-1]['frame'] <= num_failed_det:
                        iou = self._bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        if iou > iouThres:
                            track.append(face)
                            frameFaces.remove(face)
                            continue
                    else:
                        break
            if track == []:
                break
            elif len(track) > min_track:
                frameNum = numpy.array([ f['frame'] for f in track ])
                bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
                frameI = numpy.arange(frameNum[0],frameNum[-1]+1)
                bboxesI = []
                for ij in range(0,4):
                    interpfn = interp1d(frameNum, bboxes[:,ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = numpy.stack(bboxesI, axis=1)
                track_info = {'frame':frameI,'bbox':bboxesI}
                
                # Add original frame dimensions if available
                if orig_width is not None and orig_height is not None:
                    track_info['orig_width'] = orig_width
                    track_info['orig_height'] = orig_height
                    
                tracks.append(track_info)
                
        return tracks

    def _extract_face(self, image, dets, idx, crop_scale=0.40):
        """Extract face from image with safety checks"""
        if idx >= len(dets['s']) or idx >= len(dets['x']) or idx >= len(dets['y']):
            log.warning(f"Index {idx} out of bounds for face extraction")
            return None

        try:
            bs = dets['s'][idx]
            bsi = int(bs * (1 + 2 * crop_scale))
            
            padded_image = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 
                                   'constant', constant_values=(110, 110))
            
            my = dets['y'][idx] + bsi
            mx = dets['x'][idx] + bsi
            
            h, w = padded_image.shape[:2]
            y1 = max(0, int(my-bs))
            y2 = min(h, int(my+bs*(1+2*crop_scale)))
            x1 = max(0, int(mx-bs*(1+crop_scale)))
            x2 = min(w, int(mx+bs*(1+crop_scale)))
            
            face = padded_image[y1:y2, x1:x2]
            
            if face.size == 0:
                return None
                
            return face
            
        except Exception as e:
            log.error(f"Error in face extraction: {str(e)}")
            return None

    def _evaluate_network(self, face_track, frames_list, audio_path):
        self.model.eval()

        # Process face track
        dets = {'x':[], 'y':[], 's':[]}

        for det in face_track['bbox']:
            
            half_size = max((det[3]-det[1]), (det[2]-det[0])) / 2
            dets['s'].append(half_size)
            dets['y'].append((det[1]+det[3]) / 2)
            dets['x'].append((det[0]+det[2]) / 2)

        videoFeature = []
        valid_frames = []
        for idx_in_track, frame_num in enumerate(face_track['frame']):
            if frame_num >= len(frames_list):
                log.warning(f"Frame number {frame_num} exceeds available frames")
                continue
            
            image = frames_list[frame_num]
            if image is None:
                log.warning(f"Could not read frame {frame_num}")
                continue

            face = self._extract_face(image, dets, idx_in_track)
            if face is None:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-56):int(112+56), int(112-56):int(112+56)]

            videoFeature.append(face)
            valid_frames.append(frame_num)

        if not videoFeature:
            log.warning("No valid video features extracted")
            return numpy.array([])

        videoFeature = numpy.array(videoFeature)

        # Extract audio features
        if valid_frames:
            sr, audio = wavfile.read(audio_path)

            start_time = valid_frames[0] / 25.0
            end_time = (valid_frames[-1] + 1) / 25.0
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]

            audioFeature = python_speech_features.mfcc(audio_segment, sr, numcep=13)

            length = min(
                (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
                videoFeature.shape[0] / 25)
            
            audio_frames = int(round(length * 100))
            video_frames = int(round(length * 25))

            audioFeature = audioFeature[:audio_frames,:]
            videoFeature = videoFeature[:video_frames,:,:]

            # Run inference
            durationSet = {1,1,1,2,2,2,3,3,4,5,6}
            allScore = []

            with torch.no_grad():
                for duration in durationSet:
                    batchSize = int(numpy.ceil(length / duration))
                    scores = []
                    for i in range(batchSize):
                        try:
                            inputA = torch.FloatTensor(
                                audioFeature[i * duration * 100 : (i + 1) * duration * 100, :]
                            ).unsqueeze(0).to(self.device)
                            
                            inputV = torch.FloatTensor(
                                videoFeature[i * duration * 25 : (i + 1) * duration * 25, :, :]
                            ).unsqueeze(0).to(self.device)
                            
                            embedA = self.model.model.forward_audio_frontend(inputA)
                            embedV = self.model.model.forward_visual_frontend(inputV)
                            embedA, embedV = self.model.model.forward_cross_attention(embedA, embedV)
                            out = self.model.model.forward_audio_visual_backend(embedA, embedV)
                            score = self.model.lossAV.forward(out, labels=None)
                            scores.extend(score)
                            
                        except Exception as e:
                            log.error(f"Error in batch processing: {str(e)}")
                            continue
                            
                    if scores:
                        allScore.append(scores)

            if allScore:
                return numpy.mean(numpy.array(allScore), axis=0)

        return numpy.array([])