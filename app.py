"""
CanSat Ground Station - Main Web Server
Author: Aerospace Software Engineer
Date: July 2026

Flask-based ground control telemetry server. Consumes frames from ESP32CamStream,
sends them to image_processor, saves periodic/on-demand flight captures, appends 
telemetry values to a local CSV log, and serves an interactive web dashboard
with dual video streaming and Server-Sent Events (SSE) live telemetry.
"""

import os
import csv
import json
import time
import threading
from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import numpy as np

# Import custom modules
from stream_capture import ESP32CamStream
from image_processor import process_frame

# Setup application
app = Flask(__name__)

# System Configurations
CAMERA_URL = "http://192.168.4.1/stream"  # Default ESP32-CAM stream URL (INDU-CANSAT network)
SAVE_DIR = "capturas_guardadas"
CSV_PATH = "telemetry_log.csv"
AUTO_SAVE_ENABLED = False  # Set to True to enable periodic auto-saved snapshots
AUTO_SAVE_INTERVAL = 5.0  # Capture visual backup stack every 5 seconds (if AUTO_SAVE_ENABLED is True)
TARGET_FPS = 15

# Thread-safe global frame buffers and metrics cache
latest_processed_rgb = None
latest_processed_heatmap = None
latest_metrics = {
    "coverage_pct": 0.0,
    "mean_vari": 0.0,
    "mean_exg": 0.0,
    "is_alert": False,
    "fps": 0.0,
    "is_mock": True
}
cache_lock = threading.Lock()
csv_lock = threading.Lock()

# Initialize stream client (auto-fallback to simulation is enabled by default)
stream_client = ESP32CamStream(stream_url=CAMERA_URL, mock_fallback=True)

# HTML/CSS template inline for rapid single-file deployment
INDEX_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CanSat Ground Station - Agro-Ecological Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #10b981; /* Emerald green */
            --danger: #ef4444; /* Alert red */
            --warning: #f59e0b; /* Alert orange */
            --info: #3b82f6; /* Blue */
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 40%);
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 40px;
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-title-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo-mark {
            width: 35px;
            height: 35px;
            background: linear-gradient(135deg, var(--primary), var(--info));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            color: white;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }

        h1 {
            font-size: 1.4rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #ffffff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.04);
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            font-size: 0.85rem;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--primary);
            box-shadow: 0 0 10px var(--primary);
        }

        .status-dot.alert {
            background-color: var(--danger);
            box-shadow: 0 0 10px var(--danger);
            animation: pulse 1s infinite alternate;
        }

        .status-dot.mock {
            background-color: var(--info);
            box-shadow: 0 0 10px var(--info);
            animation: pulse 1.5s infinite alternate;
        }

        @keyframes pulse {
            from { opacity: 0.4; }
            to { opacity: 1; }
        }

        .dashboard-container {
            flex: 1;
            padding: 30px 40px;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 25px;
        }

        /* Metric Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(16px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: transparent;
        }

        .card.card-primary::before {
            background: linear-gradient(95deg, var(--primary), var(--info));
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        .card-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            display: block;
        }

        .card-value {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 5px;
        }

        .card-subtext {
            font-size: 0.85rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* Feeds Section */
        .feeds-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }

        @media (max-width: 1024px) {
            .feeds-grid {
                grid-template-columns: 1fr;
            }
        }

        .feed-card {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .feed-title-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .feed-title {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .feed-wrapper {
            position: relative;
            width: 100%;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: #000;
            aspect-ratio: 4/3;
        }

        .feed-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.5s ease;
        }

        .feed-wrapper:hover .feed-image {
            transform: scale(1.015);
        }

        /* Action Panel */
        .action-panel {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 20px 30px;
            border-radius: 16px;
            backdrop-filter: blur(16px);
        }

        @media (max-width: 768px) {
            .action-panel {
                flex-direction: column;
                gap: 15px;
                text-align: center;
            }
        }

        .system-info {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .system-info strong {
            color: var(--text-main);
        }

        .button-group {
            display: flex;
            gap: 12px;
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #059669);
            border: none;
            color: white;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #10b981, #047857);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35);
        }

        .toast-notification {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: rgba(16, 185, 129, 0.95);
            color: white;
            padding: 14px 24px;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1000;
            backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .toast-notification.show {
            transform: translateY(0);
            opacity: 1;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title-container">
            <div class="logo-mark">CS</div>
            <div>
                <h1>CANSAT Estación Terrena</h1>
                <p style="font-size: 0.75rem; color: var(--text-muted);">Monitoreo Agro-Ecológico y Cobertura de Suelo en Tiempo Real</p>
            </div>
        </div>
        <div class="status-badge">
            <div id="status-dot" class="status-dot"></div>
            <span id="status-text">Conectando...</span>
        </div>
    </header>

    <div class="dashboard-container">
        <!-- Telemetry Cards -->
        <div class="metrics-grid">
            <!-- Coverage Card -->
            <div class="card card-primary" id="card-coverage">
                <span class="card-label">Cobertura Vegetal (VARI + ExG)</span>
                <div class="card-value"><span id="metric-coverage">0.00</span>%</div>
                <div class="card-subtext" id="metric-coverage-status">
                    Cargando datos...
                </div>
            </div>

            <!-- Ingest FPS Card -->
            <div class="card">
                <span class="card-label">Rendimiento Telemetría</span>
                <div class="card-value"><span id="metric-fps">0.0</span> FPS</div>
                <div class="card-subtext">
                    Límite RF de ingesta: 15.0 FPS
                </div>
            </div>

            <!-- VARI Mean Card -->
            <div class="card">
                <span class="card-label">Índice VARI Promedio</span>
                <div class="card-value" id="metric-vari">0.0000</div>
                <div class="card-subtext">
                    Rango típico vegetación: > 0.1
                </div>
            </div>

            <!-- ExG Mean Card -->
            <div class="card">
                <span class="card-label">Índice ExG Promedio</span>
                <div class="card-value" id="metric-exg">0.0</div>
                <div class="card-subtext">
                    Contraste vegetación sana: > 15
                </div>
            </div>
        </div>

        <!-- Video Stream Layout -->
        <div class="feeds-grid">
            <div class="feed-card">
                <div class="feed-title-bar">
                    <span class="feed-title">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981;"></span>
                        Feed Original RGB (Con Segmentación)
                    </span>
                </div>
                <div class="feed-wrapper">
                    <img id="rgb-feed" src="/video_feed/original" class="feed-image" alt="Transmisión RGB">
                </div>
            </div>

            <div class="feed-card">
                <div class="feed-title-bar">
                    <span class="feed-title">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#3b82f6;"></span>
                        Mapa de Calor Espectral (VARI Colormap)
                    </span>
                </div>
                <div class="feed-wrapper">
                    <img id="heatmap-feed" src="/video_feed/heatmap" class="feed-image" alt="Heatmap VARI">
                </div>
            </div>
        </div>

        <!-- System Controls & CSV Paths -->
        <div class="action-panel">
            <div class="system-info">
                <div>Almacenamiento: <strong>/capturas_guardadas</strong> (Autoguardado: 5s)</div>
                <div>Historial del Vuelo: <strong>telemetry_log.csv</strong></div>
            </div>
            <div class="button-group">
                <button id="btn-toggle-mock" class="btn" style="border-color: rgba(59, 130, 246, 0.4);">
                    Modo Simulador
                </button>
                <button id="btn-snapshot" class="btn btn-primary">
                    Captura Instantánea
                </button>
            </div>
        </div>
    </div>

    <!-- Floating Toast Notification -->
    <div id="toast" class="toast-notification">
        <span id="toast-text">Captura guardada con éxito</span>
    </div>

    <script>
        // Init Server-Sent Events (SSE) telemetry receiver
        const eventSource = new EventSource("/telemetry");
        
        const statusDot = document.getElementById("status-dot");
        const statusText = document.getElementById("status-text");
        
        const metricCoverage = document.getElementById("metric-coverage");
        const metricCoverageStatus = document.getElementById("metric-coverage-status");
        const metricFps = document.getElementById("metric-fps");
        const metricVari = document.getElementById("metric-vari");
        const metricExg = document.getElementById("metric-exg");
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // 1. Update Connection Status Badge
            if (data.is_mock) {
                statusDot.className = "status-dot mock";
                statusText.innerText = "SIMULADOR (Modo de Vuelo)";
                statusText.style.color = "#3b82f6";
            } else {
                statusDot.className = "status-dot";
                statusText.innerText = "ONLINE (ESP32-CAM)";
                statusText.style.color = "#10b981";
            }
            
            // 2. Update Metrics Values
            metricCoverage.innerText = data.coverage_pct.toFixed(2);
            metricFps.innerText = data.fps.toFixed(1);
            metricVari.innerText = data.mean_vari.toFixed(4);
            metricExg.innerText = data.mean_exg.toFixed(1);
            
            // 3. Update Alerts / Ecological Density banner
            if (data.is_alert) {
                metricCoverageStatus.innerHTML = '<span style="color: var(--danger); font-weight: 600;">⚠️ ALERTA: Cobertura Crítica</span>';
                document.getElementById("card-coverage").style.borderColor = "rgba(239, 68, 110, 0.4)";
            } else {
                metricCoverageStatus.innerHTML = '<span style="color: var(--primary); font-weight: 600;">✅ Densidad Nominal</span>';
                document.getElementById("card-coverage").style.borderColor = "rgba(16, 185, 129, 0.4)";
            }
        };

        eventSource.onerror = function() {
            statusDot.className = "status-dot alert";
            statusText.innerText = "DESCONECTADO";
            statusText.style.color = "#ef4444";
        };

        // Snapshot Button Event Handler
        document.getElementById("btn-snapshot").addEventListener("click", async () => {
            try {
                const response = await fetch("/snapshot", { method: "POST" });
                const result = await response.json();
                if (result.status === "success") {
                    showToast(`Captura guardada: ${result.filename.split(/[\\\\/]/).pop()}`);
                } else {
                    showToast(`Error: ${result.message}`);
                }
            } catch (err) {
                showToast("Error de conexión al servidor");
            }
        });

        // Toggle Mock Mode Button Event Handler
        document.getElementById("btn-toggle-mock").addEventListener("click", async () => {
            try {
                const response = await fetch("/toggle_mock", { method: "POST" });
                const result = await response.json();
                if (result.mock_fallback) {
                    showToast("Simulador Activado");
                } else {
                    showToast("Conectando con hardware ESP32-CAM...");
                }
            } catch (err) {
                showToast("Error de comunicación");
            }
        });

        // Toast visual notifier helper
        function showToast(message) {
            const toast = document.getElementById("toast");
            const toastText = document.getElementById("toast-text");
            toastText.innerText = message;
            toast.classList.add("show");
            setTimeout(() => {
                toast.classList.remove("show");
            }, 3000);
        }
    </script>
</body>
</html>
"""

def background_processor():
    """
    Consumes raw frames from the camera client, processes them via the OpenCV pipeline,
    updates the shared thread-safe variables, records flight telemetry to the CSV log,
    and handles periodic auto-saving of processed backup images.
    """
    global latest_processed_rgb, latest_processed_heatmap, latest_metrics
    
    # Check capture folder exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize the CSV telemetry file with flight headers
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "coverage_pct", "mean_vari", "mean_exg", "is_alert", "fps", "is_mock"])
            
    last_save_time = time.time()
    frame_duration = 1.0 / TARGET_FPS
    
    while stream_client.is_running:
        start_time = time.time()
        
        # Get frame from background stream thread
        frame, connected, is_mock = stream_client.get_frame()
        
        if connected and frame is not None:
            # Perform matrix processing (indices, mask segmentation, colorization, HUD layers)
            rgb_hud, heatmap_hud, metrics = process_frame(
                frame, 
                actual_fps=stream_client.fps,
                threshold_vari=0.1,
                threshold_exg=15.0
            )
            
            # Cache processed layers safely
            with cache_lock:
                latest_processed_rgb = rgb_hud
                latest_processed_heatmap = heatmap_hud
                latest_metrics = {
                    **metrics,
                    "fps": round(stream_client.fps, 1),
                    "is_mock": is_mock,
                    "timestamp": time.strftime("%H:%M:%S")
                }
            
            # Record record to CSV log
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
            with csv_lock:
                with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str,
                        metrics["coverage_pct"],
                        metrics["mean_vari"],
                        metrics["mean_exg"],
                        1 if metrics["is_alert"] else 0,
                        round(stream_client.fps, 1),
                        1 if is_mock else 0
                    ])
                    
            # Handle automatic snapshot backup every AUTO_SAVE_INTERVAL seconds
            if AUTO_SAVE_ENABLED:
                now = time.time()
                if now - last_save_time >= AUTO_SAVE_INTERVAL:
                    timestamp_file = time.strftime('%Y%m%d_%H%M%S')
                    save_filename = os.path.join(SAVE_DIR, f"cansat_{timestamp_file}.jpg")
                    # Save a side-by-side stack: [RGB Feed, VARI Heatmap]
                    h_stack = np.hstack((rgb_hud, heatmap_hud))
                    cv2.imwrite(save_filename, h_stack)
                    last_save_time = now
                
        # Enforce exact telemetry update rate matching target FPS
        elapsed = time.time() - start_time
        sleep_time = max(0.001, frame_duration - elapsed)
        time.sleep(sleep_time)

@app.route('/')
def index():
    """Serves the main Ground Station Dashboard."""
    return render_template_string(INDEX_HTML)

def gen_stream(feed_type="rgb"):
    """
    Generates individual HTTP multipart chunks containing JPG images.
    Pushes frames at ~15 FPS to the connected web clients.
    """
    global latest_processed_rgb, latest_processed_heatmap
    frame_duration = 1.0 / TARGET_FPS
    
    while True:
        start_time = time.time()
        img = None
        
        with cache_lock:
            if feed_type == "rgb":
                img = latest_processed_rgb.copy() if latest_processed_rgb is not None else None
            else:
                img = latest_processed_heatmap.copy() if latest_processed_heatmap is not None else None
                
        if img is not None:
            # Compress decoded NumPy frame back to JPG payload (80% Quality for fast network transmit)
            ret, jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
        elapsed = time.time() - start_time
        sleep_time = max(0.001, frame_duration - elapsed)
        time.sleep(sleep_time)

@app.route('/video_feed/original')
def video_feed_original():
    """Endpoint serving original RGB with overlaid green vegetation mask and HUD."""
    return Response(gen_stream("rgb"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/heatmap')
def video_feed_heatmap():
    """Endpoint serving the colormapped VARI density representation and HUD."""
    return Response(gen_stream("heatmap"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    """
    Server-Sent Events (SSE) telemetry feed.
    Pushes JSON telemetry records every 100ms.
    """
    def event_stream():
        while True:
            with cache_lock:
                data = json.dumps(latest_metrics)
            yield f"data: {data}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/snapshot', methods=['POST'])
def take_snapshot():
    """
    Manual on-demand snapshot capture trigger.
    Saves a horizontal stack of the current RGB and Heatmap frames.
    """
    global latest_processed_rgb, latest_processed_heatmap
    with cache_lock:
        if latest_processed_rgb is None or latest_processed_heatmap is None:
            return jsonify({"status": "error", "message": "Estación Terrena: No hay frames de video disponibles."}), 400
        rgb_copy = latest_processed_rgb.copy()
        heatmap_copy = latest_processed_heatmap.copy()
        metrics = latest_metrics.copy()
        
    timestamp_file = time.strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(SAVE_DIR, f"snapshot_{timestamp_file}.jpg")
    
    # Save a horizontal stack
    h_stack = np.hstack((rgb_copy, heatmap_copy))
    cv2.imwrite(filename, h_stack)
    return jsonify({
        "status": "success",
        "filename": filename,
        "coverage_pct": metrics["coverage_pct"],
        "timestamp": metrics["timestamp"]
    })

@app.route('/toggle_mock', methods=['POST'])
def toggle_mock():
    """
    Manually forces connection retry or simulates flight.
    Changes configuration on the fly.
    """
    stream_client.mock_fallback = not stream_client.mock_fallback
    if not stream_client.mock_fallback:
        # Deactivate mock mode immediately to trigger hardware connection search
        stream_client.is_mock_active = False
    return jsonify({
        "mock_fallback": stream_client.mock_fallback,
        "is_mock_active": stream_client.is_mock_active
    })

if __name__ == '__main__':
    # Start stream capture thread
    stream_client.start()
    
    # Start processing daemon
    processor_thread = threading.Thread(target=background_processor, name="BackgroundProcessorThread", daemon=True)
    processor_thread.start()
    
    print("[SERVER] Starting CanSat telemetry server on http://localhost:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        # Cleanup threads on exit
        stream_client.stop()
        print("[SERVER] Ground Station shutdown successfully.")
