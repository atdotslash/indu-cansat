"""
CanSat Ground Station - Stream Capture & Ingestion Module
Author: Aerospace Software Engineer
Date: July 2026

Handles the live video stream from the ESP32-CAM (MJPEG) over the network.
Implements robust buffer reading, multi-threaded architecture, automatic
reconnection with backoff, and a high-fidelity synthetic mock terrain fallback
when the camera is offline.
"""

import urllib.request
import urllib.error
import time
import threading
import socket
import numpy as np
import cv2

class ESP32CamStream:
    """
    Manages the connection and decoding of the MJPEG stream from the ESP32-CAM.
    Runs in a background thread to prevent blocking the main server threads.
    """
    def __init__(self, stream_url="http://192.168.4.1:81/stream", mock_fallback=True):
        self.stream_url = stream_url
        self.mock_fallback = mock_fallback
        self.is_running = False
        
        self.frame = None
        self.fps = 0.0
        self.connected = False
        self.is_mock_active = False
        
        self.lock = threading.Lock()
        self.thread = None
        self._check_thread_active = False

    def start(self):
        """Starts the background frame ingestion thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._run, name="ESP32CamStreamThread", daemon=True)
        self.thread.start()
        print("[STREAM] Capture thread started.")

    def stop(self):
        """Stops the background thread and cleans up resources."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[STREAM] Capture thread stopped.")

    def get_frame(self):
        """
        Thread-safe retrieval of the latest decoded frame.
        
        Returns:
            frame: np.ndarray or None
            connected: bool (True if frame is valid)
            is_mock: bool (True if frame is synthetic mock data)
        """
        with self.lock:
            if self.frame is None:
                return None, False, self.is_mock_active
            return self.frame.copy(), self.connected, self.is_mock_active

    def _run(self):
        """Main thread loop managing connection and fallbacks."""
        retry_delay = 1.0
        max_retry_delay = 15.0

        while self.is_running:
            try:
                # Attempt to read stream from ESP32-CAM
                self._read_stream()
                retry_delay = 1.0  # Reset retry delay on successful stream exit (if clean)
            except Exception as e:
                self.connected = False
                print(f"[STREAM] Connection error: {e}")
                
                if self.mock_fallback:
                    print("[STREAM] Switching to high-fidelity Mock Flight Mode...")
                    self.is_mock_active = True
                    self.connected = True
                    self._run_mock()
                else:
                    print(f"[STREAM] Retrying connection in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)

    def _read_stream(self):
        """Establishes HTTP connection and decodes multipart MJPEG stream."""
        print(f"[STREAM] Connecting to {self.stream_url}...")
        req = urllib.request.Request(self.stream_url, headers={'User-Agent': 'CanSatGroundStation/1.0'})
        
        # Open connection with a 4-second timeout to handle initial handshake failure quickly
        try:
            stream = urllib.request.urlopen(req, timeout=4.0)
        except (urllib.error.URLError, socket.timeout) as err:
            raise ConnectionError(f"Failed to reach camera stream endpoint: {err}")

        self.connected = True
        self.is_mock_active = False
        print(f"[STREAM] Successfully connected to ESP32-CAM stream.")

        bytes_buffer = bytes()
        last_fps_time = time.time()
        frame_count = 0

        while self.is_running and not self.is_mock_active:
            # Read socket data in chunks. MJPEG streams are continuous.
            try:
                chunk = stream.read(8192)
            except (socket.timeout, ConnectionResetError) as err:
                raise ConnectionError(f"Socket connection lost during stream read: {err}")

            if not chunk:
                raise ConnectionError("Zero bytes received from stream. Device may have disconnected.")

            bytes_buffer += chunk

            # Find start (SOI) and end (EOI) markers of JPEG in buffer
            # JPEG Start of Image: 0xFFD8, End of Image: 0xFFD9
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')

            if a != -1 and b != -1 and b > a:
                jpg_data = bytes_buffer[a:b+2]
                # Slice buffer to release processed frame bytes
                bytes_buffer = bytes_buffer[b+2:]

                # Decode the raw byte buffer into OpenCV image
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Enforce standardized resolution: 800 x 600 px
                    if frame.shape[1] != 800 or frame.shape[0] != 600:
                        frame = cv2.resize(frame, (800, 600))
                    
                    with self.lock:
                        self.frame = frame

                    # FPS Telemetry Calculation
                    frame_count += 1
                    now = time.time()
                    elapsed = now - last_fps_time
                    if elapsed >= 1.0:
                        self.fps = frame_count / elapsed
                        frame_count = 0
                        last_fps_time = now

    def _init_mock_world(self):
        """
        Generates a 2D synthetic scrolling landscape map.
        Contains green forest blobs, rectangular agricultural crops, roads, and structures.
        """
        world_h = 2400
        world_w = 800
        # Base terrain: light brown/dry grass (BGR: [90, 110, 130])
        world = np.zeros((world_h, world_w, 3), dtype=np.uint8)
        world[:, :] = [90, 110, 130]

        np.random.seed(42)  # Seed for visual consistency across runs

        # 1. Populate Forest patches (Dark Green: BGR [40, 160, 60])
        for _ in range(15):
            cx = np.random.randint(0, world_w)
            cy = np.random.randint(0, world_h)
            radius = np.random.randint(80, 200)
            cv2.circle(world, (cx, cy), radius, [40, 160, 60], -1)
            # Add darker tree cluster details
            for _ in range(4):
                tcx = cx + np.random.randint(-radius//2, radius//2)
                tcy = cy + np.random.randint(-radius//2, radius//2)
                tr = np.random.randint(20, 50)
                cv2.circle(world, (tcx, tcy), tr, [30, 120, 45], -1)

        # 2. Populate Agricultural Crop fields (Light Green: BGR [60, 210, 90])
        for _ in range(12):
            x1 = np.random.randint(0, world_w - 150)
            y1 = np.random.randint(0, world_h - 150)
            w = np.random.randint(120, 250)
            h = np.random.randint(120, 250)
            cv2.rectangle(world, (x1, y1), (x1+w, y1+h), [60, 210, 90], -1)

        # 3. Add Infrastructure (Gray Roads and Building roofs)
        # Vertical highway running down the canvas
        cv2.line(world, (world_w//3, 0), (world_w//3 + 120, world_h), [140, 140, 140], 15)
        # Lateral secondary road
        cv2.line(world, (0, world_h//2), (world_w, world_h//2 + 100), [140, 140, 140], 8)
        
        # Concrete roofs/buildings near roads
        for _ in range(10):
            bx = np.random.randint(world_w//3 - 80, world_w//3 + 200)
            by = np.random.randint(100, world_h - 100)
            # Draw building outline
            cv2.rectangle(world, (bx, by), (bx+40, by+40), [80, 80, 80], -1)
            # Inner roof details (reddish-orange brick color or white panels)
            cv2.rectangle(world, (bx+6, by+6), (bx+34, by+34), [50, 80, 220], -1)

        # 4. Add sensor noise simulating RF telemetry degradation
        noise = np.random.normal(0, 8, world.shape).astype(np.int16)
        self.mock_world = np.clip(world.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _run_mock(self):
        """Simulates camera flight path at target 15 FPS scrolling down the landscape."""
        if not hasattr(self, 'mock_world'):
            self._init_mock_world()

        world_h, world_w, _ = self.mock_world.shape
        y_offset = 0
        last_time = time.time()
        frame_count = 0
        frame_duration = 1.0 / 15.0

        while self.is_running and self.is_mock_active:
            start_frame_time = time.time()

            # Slice sliding 800 x 600 viewport. Handle wrapping.
            y1 = y_offset
            y2 = y_offset + 600
            
            if y2 <= world_h:
                frame = self.mock_world[y1:y2, 0:800].copy()
            else:
                slice_bottom = self.mock_world[y1:world_h, 0:800]
                slice_top = self.mock_world[0:y2 - world_h, 0:800]
                frame = np.vstack((slice_bottom, slice_top))

            # Update scrolling offset to simulate downward CanSat descent
            # 2 pixels per frame
            y_offset = (y_offset + 2) % world_h

            with self.lock:
                self.frame = frame

            # Telemetry Metrics inside Mock Mode
            frame_count += 1
            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                last_time = now

            # Periodically query if actual camera is back online (non-blocking)
            if frame_count % 30 == 0 and not self._check_thread_active:
                self._check_thread_active = True
                threading.Thread(target=self._check_camera_availability, daemon=True).start()

            # Maintain strict 15 FPS sleep
            frame_elapsed = time.time() - start_frame_time
            sleep_time = max(0.001, frame_duration - frame_elapsed)
            time.sleep(sleep_time)

    def _check_camera_availability(self):
        """Checks if the camera stream is reachable to resume hardware acquisition."""
        try:
            req = urllib.request.Request(self.stream_url, headers={'User-Agent': 'PingTest/1.0'})
            # 1.0s timeout to keep tests snappy
            with urllib.request.urlopen(req, timeout=1.0) as conn:
                conn.read(1)
            
            # If successful, deactivate mock and exit current mock loop
            print("[STREAM] ESP32-CAM stream re-established. Resuming telemetry stream...")
            self.is_mock_active = False
        except Exception:
            pass  # Fail silently, continue in mock mode
        finally:
            self._check_thread_active = False
