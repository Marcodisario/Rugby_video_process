# Rugby Video Process

Proyecto de procesamiento de imágenes aplicado a partidos de rugby del club.
El objetivo inicial es detectar y seguir jugadores en un clip corto. Después se
incorporarán pelota, equipos, identidades y reconocimiento de eventos.

## Estado actual

- Entorno local Python 3.12 en `.venv`.
- Descarga por intervalos con yt-dlp y recorte con FFmpeg.
- Inspección de videos mediante fotogramas con marcas de tiempo.
- Script de inferencia en CPU con YOLO26n preentrenado, con exportación de video,
  detecciones y tiempos. No se ha realizado entrenamiento específico de rugby.
- Entrenamiento GPU preparado en `scripts/train_yolo26n.py`; todavía no ejecutado.

## Entrenar en la PC con NVIDIA

Ver [la guía de entrenamiento GPU](docs/entrenamiento_gpu.md) para instalar el
entorno nuevo, preparar imágenes y etiquetas, y ejecutar el script.

La configuración inicial usa YOLO26n preentrenado, GPU 0, batch 4, entrada 640 y
hasta 50 épocas. Las clases de la plantilla son `player`, `ball` y `referee`.
Se necesita completar el dataset antes de entrenar; el script comprueba que
existan imágenes/etiquetas y que CUDA esté disponible. No cambia a CPU si falta
GPU. No se ejecutó el script de entrenamiento en esta PC.

## Preparación en Windows

Con Python 3.12 y `uv` disponibles:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-video.txt
```

La descarga desde YouTube usa Node.js 22 o posterior, disponible en esta PC.
`imageio-ffmpeg` incluye FFmpeg; no hace falta instalarlo globalmente.
Las versiones de descarga están fijadas para reproducibilidad, pero YouTube
puede cambiar y requerir una actualización futura de yt-dlp.

## Descargar un fragmento

```powershell
.venv\Scripts\python.exe scripts\download_clip.py 'https://www.youtube.com/watch?v=EmaaY21OwPU' --start 01:02:15 --duration 30
```

`--start` y `--duration` aceptan segundos, `MM:SS` o `HH:MM:SS`.
Por defecto se selecciona video H.264 de hasta 1080p, sin audio, y se descarga
solo el intervalo solicitado. FFmpeg recodifica para ajustar los cortes; esa
operación puede demorar más que la duración del clip en una CPU.
El programa no sobrescribe clips existentes. `--height 720` permite pedir una
versión más pequeña, aunque conviene preservar 1080p para estudiar la pelota.

## Inspeccionar un clip

```powershell
.venv\Scripts\python.exe scripts\inspect_clip.py data\clips\EmaaY21OwPU_3735-3765s_1080p.mp4 --source-start 3735 --interval 2 --output outputs\baseline
```

Genera `metadata.json`, fotogramas JPEG en resolución original y
`contact_sheet.jpg`, una plancha con una muestra cada cinco segundos.
`--source-start` indica el segundo inicial en el video de YouTube para mostrar
las marcas de tiempo originales. Las marcas no son el reloj del partido.
El muestreo de esta inspección no modifica los FPS del clip.

## Fuente inicial

- [San Patricio vs. Mariano Moreno — Primera B — URBA 2026](https://www.youtube.com/watch?v=EmaaY21OwPU)
- Duración informada: 1 h 51 min 30 s.
- Formato disponible y descargado para inspección: 1920 × 1080, 30 FPS.
- Primera muestra: 1:02:40–1:03:40. Contiene repeticiones, transiciones y una
  conversión; se conserva como ejemplo de situaciones que deben separarse del
  seguimiento continuo.
- Segunda muestra: 1:01:50–1:02:30. Se observa la preparación de un lineout y
  el inicio de la jugada; tiene bastante tiempo de espera.
- Intervalo elegido para la primera prueba: 1:02:15–1:02:45, que empieza cerca
  del lanzamiento del lineout y abarca la jugada posterior.

### Observaciones de la inspección visual

Las camisetas celestes y rojas/blancas se distinguen visualmente. Hay árbitros,
personas al costado de la cancha y público que un detector genérico de personas
también podría marcar. El paneo y el zoom cambian las posiciones y tamaños en
imagen, por lo que no deben interpretarse como desplazamientos reales en metros.
Las planchas de fotogramas sirven como inspección preliminar, no como
anotaciones de eventos ni verificación exhaustiva de cada cuadro. Los resultados
de la primera ejecución del detector se registran más abajo; aún no hay
etiquetas manuales con las que medir precisión o recall.

## Estructura

```text
scripts/                Descarga e inspección reproducibles
requirements-video.txt  Dependencias para preparar los videos
data/clips/             Fragmentos descargados (fuera de Git)
outputs/                Fotogramas, metadatos y resultados (fuera de Git)
.venv/                  Entorno local (fuera de Git)
```

## Prueba de detección de 10 segundos

El clip `data/clips/EmaaY21OwPU_3740-3750s_1080p.mp4` corresponde a
1:02:20–1:02:30 del video original: 300 cuadros a 30 FPS, 1920 × 1080.
Se recortó localmente desde el clip de 30 segundos, con FFmpeg y H.264 CRF 18.

El script requiere Ultralytics y PyTorch para CPU en el entorno del proyecto,
y los pesos oficiales descargados en `weights/yolo26n.pt`.

Instalación usada para esta prueba en Windows (la futura PC con GPU necesitará
una instalación de PyTorch apropiada para CUDA):

```powershell
uv pip install --python .venv\Scripts\python.exe 'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements-model.txt
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) '.cache\ultralytics'
New-Item -ItemType Directory -Force -Path weights | Out-Null
.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt')"
```

Ejecutar:

```powershell
.venv\Scripts\python.exe scripts\detect_video.py data\clips\EmaaY21OwPU_3740-3750s_1080p.mp4 --source-start 3740
```

Procesa todos los cuadros con `device=cpu`, `imgsz=640` y confianza mínima
0.25. El tamaño de entrada del detector se reduce manteniendo proporciones y
agregando el padding necesario; el video anotado conserva 1080p y 30 FPS.
No se filtran clases: se mantienen las clases COCO del modelo para poder
inspeccionar también errores y detecciones ajenas a rugby.

Resultados en `outputs/yolo26n_10s/`:

- `annotated.mp4`: video H.264 con cajas, clases y confianza.
- `detections.csv`: una fila por detección, con cuadro, tiempo relativo,
  segundo del video original, clase, confianza y caja en píxeles de 1080p.
- `frames.csv`: cantidad de detecciones y tiempo de inferencia por cuadro.
- `summary.json`: parámetros, versiones, tiempos y conteos por clase.
- `preview_*.jpg`: cuatro fotogramas anotados para revisión rápida.

El tiempo de procesamiento incluye predicción, dibujo, CSV, lectura del video
tras el primer cuadro y codificación de salida. No incluye instalación,
descarga, importaciones, carga del modelo ni una predicción de calentamiento.
El tiempo medio de inferencia mide solamente la ejecución del modelo.

Para repetir sin sobrescribir resultados, pasar una nueva carpeta con
`--output outputs/yolo26n_10s_run2`. La carpeta de salida debe ser nueva.
Las detecciones repetidas entre cuadros no representan jugadores únicos.
Esta prueba no asigna IDs ni equipos y no calcula pases o tackles.

### Resultado observado — 3 de noviembre de 2026

| Medida | Resultado |
| --- | --- |
| Cuadros procesados y verificados en salida | 300 de 300 |
| Duración del video | 10 segundos |
| Procesamiento, incluida escritura del MP4 | 55,21 segundos |
| Velocidad de procesamiento | 5,43 cuadros/segundo |
| Inferencia media del modelo | 143,88 ms/cuadro |
| Carga del modelo | 0,24 segundos |
| Inicialización y predicción de calentamiento | 53,85 segundos |
| Cuadros con alguna detección `person` | 300 |
| Cuadros con alguna detección `sports ball` | 13 |

Estos tiempos corresponden a esta ejecución local, con PyTorch CPU y 7 hilos.
La instalación y descarga no están incluidas. La salida conserva 30 FPS: su
reproducción dura 10 segundos aunque calcularla haya requerido más tiempo.

Se registraron 4.871 detecciones de personas acumuladas entre cuadros, unas
16,24 por cuadro; no son 4.871 personas distintas. También hubo una detección
`frisbee` y tres `car`. En la revisión visual de los cuadros 52, 57 y 125, la
pelota aparece etiquetada como `frisbee` en el 52 y como `sports ball` en los
otros dos. Hay jugadores sin detectar y personas ajenas al juego detectadas.
La detección de pelota es intermitente. La cifra 13/300 no mide recall: falta
anotar en qué cuadros la pelota es visible y cuál es su ubicación real.

Validación: decodificación completa del MP4 de salida (300 cuadros de
1920 × 1080 a 30 FPS), 300 filas en el registro por cuadro y coincidencia entre
los conteos del CSV de detecciones y `summary.json`.

## Próximo experimento

Revisar las detecciones de esta prueba y anotar una pequeña muestra para medir
errores. Después comparar una entrada de mayor resolución sobre el mismo clip
y agregar tracking para medir pérdidas e intercambios de ID. La prueba actual
confirma que la CPU permite comenzar con procesamiento offline. La otra PC con
GPU se usará después para comparar modelos y entrenar con datos etiquetados.

No se asume que un ID del tracker identifica a un jugador durante todo el
partido. Las repeticiones y cortes de cámara deben manejarse explícitamente
para no contar eventos dos veces. Entrenamiento y evaluación se separarán por
partido o por secuencias independientes.
