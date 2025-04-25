import torch
import cv2
import numpy as np
import os
import json
import time
import subprocess
import requests
import m3u8
import logging
import concurrent.futures
from typing import List, Optional

# Configure logging
log = logging.getLogger('aiproducer')
if not log.handlers:
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# Create a session for TS downloads
ts_session = requests.Session()
ts_session.verify = False

# Configure device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoProperties:
    def __init__(self, nframes: int, nframes_incl_missing: int = 0, fps: float = 0, 
                n_frames_per_segment: int = 0, reference_width: int = 0, reference_height: int = 0):
        self.nframes = nframes
        self.nframes_incl_missing = nframes_incl_missing
        self.fps = fps
        self.n_frames_per_segment = n_frames_per_segment
        self.reference_width = reference_width
        self.reference_height = reference_height

class VideoStore:
    def __init__(self, task_id: str, m3u8_url: str, video_segment_path: str) -> None:
        self.m3u8_url = m3u8_url
        self.segment_manifest = None
        self.nsegments = 0
        self.task_id = task_id
        self.video_segment_path = video_segment_path
        self.video_properties: VideoProperties = VideoProperties(nframes=0, nframes_incl_missing=0, fps=0, n_frames_per_segment=0, reference_width=0, reference_height=0)
        # This stores the information about missing frames in each segment in following format:
        # [
        #   [missing_from_frame_number, nb_missing_frames],
        #   ...
        # ]
        self.missing_frames = []
        self._cap = None
        self._frame = None
        self._current_frame_number = -1
        self._current_segment_idx = -1
        self._eos = False

    @property
    def cap(self) -> cv2.VideoCapture | None:
        return self._cap

    @property
    def current_frame(self) -> object | None:
        return self._frame

    @property
    def eos(self) -> bool:
        return self._eos

    @property
    def current_frame_number(self) -> int:
        return self._current_frame_number

    @property
    def current_segment_idx(self) -> int:
        return self._current_segment_idx

    def download_video(self) -> bool:
        t0 = time.monotonic()

        # Download the manifest
        m3u8_url = self.m3u8_url.replace('forzasys', 'forzify')
        if 'Manifest.m3u8' in m3u8_url:
            base_manifest = m3u8.load(m3u8_url)
            # Pick the lowest quality stream
            lq_bw = 1000000000
            lq_manifest = None
            for manifest in base_manifest.playlists:
                if lq_manifest is None or manifest.stream_info.bandwidth < lq_bw:
                    lq_bw = manifest.stream_info.bandwidth
                    lq_manifest = manifest

            if lq_manifest is None:
                log.error('No segment manifest in base manifest')
                return False

            segment_manifest = m3u8.load(lq_manifest.absolute_uri, headers={'X-Forzify-Client': 'telenor-internal'})
        else:
            # We were given a concrete bitrate to work on
            segment_manifest = m3u8.load(m3u8_url, headers={'X-Forzify-Client': 'telenor-internal'})

        self.segment_manifest = segment_manifest
        self.nsegments = len(self.segment_manifest.segments)

        # Download the segments
        if not self.download_segments():
            return False

        t1 = time.monotonic()
        log.debug('[video download] %s downloaded in %0.2fs' % (m3u8_url, t1 - t0,))
        return True

    def download_segments(self) -> bool:
        if self.segment_manifest is None or self.segment_manifest.segments is None:
            log.error('No segment manifest found')
            return False

        def _download_segment(session, segment, file_path, retries=3, delay=1) -> int:
            for attempt in range(retries):
                try:
                    log.info(f'Downloading {segment.absolute_uri} to {file_path}')

                    r = session.get(
                        segment.absolute_uri,
                        headers={'X-Forzify-Client': 'telenor-internal'},
                        stream=True,
                        verify=False
                    )

                    bytes = 0

                    if r.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:  # filter out keep-alive new chunks
                                    f.write(chunk)
                                    bytes += len(chunk)
                        return bytes
                    else:
                        log.error(f'Failed to download segment {segment.absolute_uri}, status code: {r.status_code}')
                except requests.RequestException as e:
                    log.exception(f'Exception downloading segment {segment.absolute_uri}')

                if attempt < retries - 1:
                    log.info(f'Retrying in {delay} seconds...')
                    time.sleep(delay)

            log.error(f'Could not download segment {segment.absolute_uri} after {retries} attempts')
            return -1

        bytes_downloaded = 0
        segments_downloaded = 0

        os.makedirs(self.video_segment_path, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_to_segment = {
                executor.submit(
                    _download_segment, ts_session, segment, f'{self.video_segment_path}/{idx}.ts'):
                        (idx, segment) for idx, segment in enumerate(self.segment_manifest.segments)
            }

            for future in concurrent.futures.as_completed(future_to_segment):
                bytes = future.result()
                if bytes == -1:
                    return False

                bytes_downloaded += bytes
                segments_downloaded += 1

        return True

    def _release_cap(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def rewind(self) -> None:
        #
        # After calling this, get_next_frame() will start to return frames from the start of the video again.
        #
        self._current_segment_idx = -1
        self._current_frame_number = -1
        self._eos = False
        self._release_cap()

    def read(self):
        # opencv2.VideoCapture style read()
        ret, frame = self.get_next_frame()
        return ret is not None, frame if ret is not None else None

    def grab(self):
        # opencv2.VideoCapture style grab()
        ret, _ = self.get_next_frame()
        return ret is not None

    def retrieve(self):
        # opencv2.VideoCapture style retrieve()
        return True, self.current_frame

    def isOpened(self):
        # opencv2.VideoCapture style isOpened()
        return True

    def get(self, propId: int):
        # opencv2.VideoCapture style get()
        if propId == cv2.CAP_PROP_POS_FRAMES:
            return self._current_frame_number
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return self.video_properties.nframes
        elif propId == cv2.CAP_PROP_FPS:
            return self.video_properties.fps
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.video_properties.reference_width
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.video_properties.reference_height
        else:
            return 0

    def convert_from_linear(self, frame_number: int) -> int:
        if not self.missing_frames:
            return frame_number

        delta = 0
        for mframes in self.missing_frames:
            if frame_number > mframes[0]:
                delta += mframes[1]

        return frame_number - delta

    def get_next_frame(self) -> tuple[int | None, object | None]:
        if self._eos:
            return None, None

        if self._cap is None:
            # We don't have a cap or we've reached the end of a segment, so try to open the next segment.
            next_segment_idx = self._current_segment_idx + 1
            if next_segment_idx >= self.nsegments:
                self._eos = True
                return None, None

            segment_file = f'{self.video_segment_path}/{next_segment_idx}.ts'
            if not os.path.exists(segment_file):
                log.warning(f'Segment {segment_file} does not exist!')
                self._eos = True
                return None, None

            self._cap = cv2.VideoCapture(segment_file)
            if not self._cap.isOpened():
                log.warning(f'Could not open segment {segment_file}')
                self._eos = True
                return None, None

            self._current_segment_idx = next_segment_idx

        ret, frame = self._cap.read()
        if not ret:
            # End of segment, release cap and try to open the next segment
            self._release_cap()
            return self.get_next_frame()

        # Update the frame number
        self._current_frame_number += 1
        self._frame = frame

        return self._current_frame_number, frame

    def concatenate_segments_to_file(self, output_path: str) -> str:
        """Concatenate all segments to a single file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a list file for FFmpeg
        list_file_path = os.path.join(os.path.dirname(output_path), "segments.txt")
        with open(list_file_path, "w") as list_file:
            for idx in range(self.nsegments):
                segment_path = f'{self.video_segment_path}/{idx}.ts'
                if os.path.exists(segment_path):
                    list_file.write(f"file '{os.path.abspath(segment_path)}'\n")

        # Use FFmpeg to concatenate the segments
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c", "copy",
            output_path
        ]
        subprocess.run(cmd, check=True)

        # Clean up
        os.remove(list_file_path)
        
        return output_path
