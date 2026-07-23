# Estación Terrena CanSat - Procesamiento de Video e Índices Agro-Ecológicos

Este proyecto implementa el sistema de software de la **Estación Terrena** para un CanSat. Su objetivo principal es recibir el flujo de video en vivo (MJPEG) transmitido por un módulo ESP32-CAM, procesar los fotogramas en tiempo real a 15 FPS para calcular índices de vegetación (VARI y ExG), clasificar la cobertura del suelo (vegetación sana vs. suelo desnudo/infraestructura) y desplegar un panel web interactivo local para telemetría.

---

## 🛠️ Arquitectura del Sistema

El proyecto consta de tres módulos de software principales desarrollados en Python:

```mermaid
graph TD
    ESP32[ESP32-CAM (WiFi/RF)] -->|MJPEG Stream| SC[stream_capture.py]
    Mock[Scrolling Terrain Generator] -->|Mock Frames (Fallback)| SC
    SC -->|Raw RGB Frames| IP[image_processor.py]
    IP -->|Vectorized NumPy Processing| IP
    IP -->|1. VARI Index Calculation| IP
    IP -->|2. ExG Index Calculation| IP
    IP -->|3. Soil Classification Mask| IP
    IP -->|4. JET Heatmap & HUD Overlay| IP
    IP -->|Processed Frames & Metrics| App[app.py (Flask Server)]
    App -->|Local Logging| CSV[(telemetry_log.csv)]
    App -->|Image Archiving| Disk[(capturas_guardadas/)]
    App -->|Server-Sent Events & Streams| Dash[Dashboard Web (HTML5/CSS3/JS)]
```

### Componentes de Software

1. **`stream_capture.py` (Ingesta de Video):** 
   - Se conecta al flujo de red HTTP del ESP32-CAM.
   - Decodifica el flujo MJPEG byte a byte de forma eficiente en un hilo secundario para no bloquear el servidor Flask.
   - Posee un mecanismo de reconexión automática si se producen microcortes de RF o WiFi.
   - **Simulación de Vuelo (Mock Fallback):** Si la cámara física no es detectable, el sistema genera automáticamente un mapa de terreno procedural en movimiento para realizar pruebas de laboratorio sin necesidad del hardware conectado.

2. **`image_processor.py` (Procesador Matricial NumPy/OpenCV):**
   - **Índice VARI (Visible Atmospherically Resistant Index):** 
     $$\text{VARI} = \frac{G - R}{G + R - B + 10^{-6}}$$
     Atenúa los efectos de la dispersión atmosférica utilizando únicamente el espectro visible.
   - **Índice ExG (Excess Green):**
     $$\text{ExG} = 2G - R - B$$
     Resalta el vigor de la vegetación verde frente a suelos arcillosos o rocosos.
   - **Clasificación de Suelo:** Segmenta píxeles mediante condiciones umbrales combinadas. Calcula en tiempo real el porcentaje ($\%$) de cobertura de dosel vegetal.
   - **Capas de Visualización (HUD Overlay):** Superpone un panel semitransparente con los FPS reales, fecha/hora UTC, porcentaje de cobertura y banderas de alerta ecológica si la cobertura desciende del $20\%$.

3. **`app.py` (Servidor Flask y Dashboard):**
   - Corre un servidor web local en `http://localhost:5000`.
   - Implementa *Multipart streaming* (`multipart/x-mixed-replace`) para transmitir por separado el canal RGB procesado y el mapa de calor VARI (`COLORMAP_JET`).
   - Envía datos estructurados cada $100\text{ ms}$ utilizando *Server-Sent Events* (SSE) al dashboard HTML.
   - Genera registros automatizados persistentes en un archivo `telemetry_log.csv` por cada fotograma procesado.
   - Guarda snapshots en disco (`capturas_guardadas/`) de manera periódica (cada 5 segundos) y a demanda mediante la interfaz.

---

## 📂 Estructura del Proyecto

```text
indu-cansat/
├── .gitignore              # Configuración de exclusiones de Git (logs, capturas y cache)
├── requirements.txt        # Dependencias de Python
├── README.md               # Documentación general del sistema (este archivo)
├── stream_capture.py       # Hilo de conexión y parser del buffer MJPEG
├── image_processor.py      # Cálculos matriciales VARI, ExG y máscaras de segmentación
└── app.py                  # Servidor web local de Flask y Web Dashboard
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Requisitos Previos

Se recomienda utilizar un entorno virtual de Python (por ejemplo, `env_cansat`):

```bash
# Crear entorno virtual
python -m venv env_cansat

# Activar entorno (Windows)
.\env_cansat\Scripts\activate
```

### 2. Instalar Dependencias

Instale las librerías necesarias ejecutando:

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Estación Terrena

Inicie el servidor de telemetría:

```bash
python app.py
```

Al iniciar, el sistema mostrará la dirección de enlace en la consola:
```text
[STREAM] Capture thread started.
[STREAM] Connecting to http://192.168.4.1/stream...
[STREAM] Camera connection failed: <urlopen error [WinError 10061]...>. Entering Mock Mode...
[STREAM] Switching to high-fidelity Mock Flight Mode...
[SERVER] Starting CanSat telemetry server on http://localhost:5000
```
*(Nota: Si no encuentra un ESP32-CAM activo en la IP `192.168.4.1`, entrará de manera automática en el **Modo Simulador**, generando el feed de terreno en movimiento para pruebas).*

### 4. Acceder al Panel de Control

Abra su navegador web y navegue a:
👉 **[http://localhost:5000](http://localhost:5000)**

Desde el dashboard premium con tema oscuro, podrá:
- Ver el canal RGB con el overlay verde de segmentación.
- Observar el mapa espectral VARI en falso color (`COLORMAP_JET`).
- Monitorear en tiempo real la cobertura foliar y los FPS.
- Habilitar o deshabilitar el Simulador mediante el botón **Modo Simulador**.
- Tomar capturas de imagen a demanda usando **Captura Instantánea**.

---

## 💾 Registro de Datos y Archivo local

- **`telemetry_log.csv`:** Archivo tipo log creado en la raíz del proyecto. Registra las siguientes métricas de cada fotograma para análisis posterior en software SIG o MATLAB:
  `timestamp, coverage_pct, mean_vari, mean_exg, is_alert, fps, is_mock`
- **`capturas_guardadas/`:** Directorio local donde se guardan automáticamente las imágenes compuestas en formato horizontal `[RGB | HEATMAP]` de forma periódica y cuando se solicita una captura manual.
