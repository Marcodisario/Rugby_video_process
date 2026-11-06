# Entrenar YOLO26n en la otra PC

## Reanudar y evaluar en esta PC

Seleccionar `.venv-gpu/Scripts/python.exe` en el editor.

- Correr `scripts/resume_yolo26n.py` para continuar la ejecución
  `yolo26n_rugby_20261105_232157` desde `last.pt` (26 épocas completadas de 50).
  Conserva el estado del optimizador y hace una copia de los checkpoints,
  argumentos y CSV antes de reanudar. No usar el script de entrenamiento
  inicial para continuar: ese script crea un experimento nuevo.
- Correr `scripts/evaluate_yolo26n.py` para evaluar `best.pt` de esa ejecución.
  Usa test, sin entrenar, y guarda `resumen.txt`, `metrics.json`, `per_class.csv`
  y gráficos en una carpeta nueva de `outputs/evaluation/`. Se puede evaluar
  el mejor checkpoint actual incluso antes de completar las 50 épocas.
- Ambos admiten `--check` para comprobar sin entrenar ni evaluar, y `--weights`
  para seleccionar otro checkpoint explícitamente.

Se verificó que la imagen del error vuelve a ser legible y coincide con el
original: la causa transitoria exacta no pudo certificarse. Para evitar
dependencia de imágenes sincronizadas, los tres scripts usan una copia local
en `~/AppData/Local/RugbyVideoProcess/datasets/rugby_tokyo_v3`. Se reconstruye
desde el ZIP con las mismas etiquetas adaptadas y se verifica por hashes.
No se modifican los checkpoints durante `--check`.

Precisión indica qué proporción de detecciones fue correcta; recall, qué
proporción de objetos etiquetados encontró. mAP50 y mAP50-95 resumen la calidad
de detección con distintos requisitos de coincidencia de las cajas. Los tiempos
son milisegundos por imagen medidos durante evaluación; no son FPS de un video
con lectura, tracking y escritura. Este test público no certifica desempeño
en partidos del club ni separación por partido entre train y test.

## Inicio en esta PC

La GPU detectada en esta PC es una RTX 5060 Laptop de 8 GB. El entorno local
de entrenamiento se encuentra en `.venv-gpu`.

- En el editor seleccionar `.venv-gpu/Scripts/python.exe` y usar el botón
  de correr sobre `scripts/train_yolo26n.py` para iniciar el entrenamiento.
- Comprobar sin entrenar desde PowerShell:
  `.venv-gpu\Scripts\python.exe scripts\train_yolo26n.py --check`.
- Si falta memoria, ejecutar el script con `--batch 2`.

El dataset público Rugby model v3 ya está descargado en
`data/datasets/rugby_tokyo_v3`: ZIP y exportación original preservados, y copia
`prepared` con las clases player=0, ball=1, referee=2. La configuración
`configs/rugby_dataset.yaml` apunta a esa copia (1724 train y 166 val).
Hay otras 82 imágenes de prueba en `prepared/images/test`, reservadas para
evaluación posterior; el entrenamiento no las usa.

Se conservaron los grupos originales y se excluyeron de la copia preparada
las imágenes sin cajas de las tres clases objetivo. La revisión verificó
imágenes legibles, formato, IDs, coordenadas y ausencia de duplicados exactos
entre grupos; no certifica todas las anotaciones ni independencia por partido.
Hay fotogramas cercanos y montajes en la fuente; para medir desempeño real
conviene evaluar además con un partido propio independiente.

El script está preparado, pero **no se ejecutó ningún entrenamiento**.
Los MP4 y CSV de predicciones no son por sí solos un dataset supervisado.

## Qué va a entrenar

`scripts/train_yolo26n.py` ajusta los pesos oficiales de YOLO26n a tres clases:

| ID | Clase | Qué etiquetar |
| --- | --- | --- |
| 0 | player | Jugadores de ambos equipos; todavía no se distinguen equipos ni identidades |
| 1 | ball | Pelota de rugby visible |
| 2 | referee | Árbitro y asistentes |

Se pueden cambiar las clases en `configs/rugby_dataset.yaml`, manteniendo IDs
consecutivos desde cero y su correspondencia con las etiquetas. Estos IDs son
propios del dataset: `ball` es 1, no el 32 de `sports ball` en COCO.

Configuración inicial: GPU NVIDIA 0, 50 épocas como máximo, batch 4, entrada
640, precisión mixta habilitada, sin caché de imágenes en RAM y workers 0.
Early stopping puede finalizar tras 10 épocas sin mejora de validación.
El modelo no aprenderá pases, tackles ni tracking con estas etiquetas: aprenderá
a ubicar las tres clases en imágenes.

## Preparar el entorno de la PC con GPU

Copiar el proyecto a la otra computadora, instalar Python 3.12 y tener un
controlador NVIDIA compatible con la versión CUDA elegida. Crear un entorno
nuevo: **no copiar `.venv` desde esta PC**, porque tiene PyTorch para CPU y las
rutas del entorno no son portables. Tampoco hace falta copiar `.cache`.

Desde PowerShell, en la raíz del proyecto:

```powershell
py -3.12 -m venv .venv-gpu
.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
.venv-gpu\Scripts\python.exe -m pip install -r requirements-training.txt
```

Se fijó PyTorch 2.11.0 + torchvision 0.26.0 con CUDA 12.8, combinación publicada
para Windows/Linux en las [instrucciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/).
El soporte de Blackwell se incorporó con las compilaciones CUDA 12.8;
ver [anuncio oficial](https://pytorch.org/blog/pytorch-2-7/).
La compatibilidad final también depende del controlador de la otra PC; esta
configuración no se ha probado físicamente en su RTX 5070.

Comprobar el entorno antes de entrenar:

```powershell
nvidia-smi
.venv-gpu\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU disponible:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Revisar instalación CUDA')"
```

El script exige CUDA y verifica una operación en la GPU antes de cargar el
modelo. Si no puede utilizarla, termina con un mensaje; no cambia a CPU.

## Preparar las imágenes y etiquetas

Crear esta estructura con datos reales (las carpetas no incluyen ejemplos
inventados):

```text
data/datasets/rugby/
├── images/
│   ├── train/
│   │   ├── partido_a_0001.jpg
│   │   └── ...
│   └── val/
│       ├── partido_b_0001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── partido_a_0001.txt
    │   └── ...
    └── val/
        ├── partido_b_0001.txt
        └── ...
```

Cada imagen debe tener un TXT con el mismo nombre y una línea por objeto:

```text
class_id x_center y_center width height
```

Las cuatro coordenadas están normalizadas entre 0 y 1 respecto del tamaño de
la imagen. Por ejemplo, una línea **ilustrativa**, no una etiqueta para estos
videos, sería `0 0.50 0.50 0.10 0.20`. No poner encabezados en los TXT.

Etiquetar todas las instancias visibles de las clases elegidas, incluyendo la
pelota cuando se pueda ubicar. No inventar su posición cuando esté completamente
oculta. Una imagen negativa, sin objetos de estas clases, lleva un TXT vacío.
Este script exige TXT explícitos incluso para negativos, para no confundir
imágenes pendientes de etiquetar con imágenes sin objetos.

Separar entrenamiento y validación por partido o por secuencias independientes.
No repartir aleatoriamente cuadros consecutivos del mismo video entre ambos
grupos. Reservar además otro conjunto independiente para una evaluación final
cuando haya suficientes datos; el script usa train y val durante el ajuste.

Si se exporta desde Roboflow, elegir etiquetas de detección YOLO y revisar que
los IDs y nombres coincidan. Se puede usar su YAML con `--data`, siempre que
`train` y `val` apunten a carpetas con el esquema images/labels. Este script
admite carpetas, no listas de rutas ni manifiestos TXT. Interpreta `path`
relativo al YAML, y train/val relativos a path, generando una copia con rutas
absolutas para Ultralytics. Sin `path`, toma la carpeta del YAML como raíz.

`data/` y `weights/` están fuera de Git. Si se transfiere el proyecto mediante
Git, copiar el dataset por separado. Se puede copiar `weights/yolo26n.pt` que ya
descargamos, o dejar que el script descargue los pesos oficiales al ejecutarse
con conexión. Ultralytics puede descargar recursos adicionales al inicializar.

## Ejecutar cuando esté el dataset

Desde la raíz del proyecto:

```powershell
.venv-gpu\Scripts\python.exe scripts\train_yolo26n.py
```

O ajustar el experimento sin editar el código:

```powershell
.venv-gpu\Scripts\python.exe scripts\train_yolo26n.py --epochs 30 --batch 4 --imgsz 640 --device 0
```

Si se agota la VRAM, repetir con `--batch 2` o `--batch 1`. Si luego se quiere
probar mayor resolución para la pelota, se puede pasar `--imgsz 960`; requiere
más memoria y tiempo. Cada ejecución crea una carpeta nueva para conservar las
anteriores. Los valores iniciales también están agrupados en `DEFAULTS` en el
archivo Python.

## Resultados

Cada ejecución guarda en `outputs/training/yolo26n_rugby_FECHA_HORA/`:

- `weights/best.pt`: pesos con mejor criterio de validación de Ultralytics.
- `weights/last.pt`: checkpoint de la última época guardada.
- `results.csv` y gráficos: evolución del entrenamiento y validación.
- `args.yaml`: configuración utilizada.
- `dataset.yaml`: copia de la configuración del dataset con rutas resueltas.

Para revisar el resultado conviene mirar las métricas por clase y las imágenes
de validación, especialmente las detecciones de pelota. Una pérdida menor no
garantiza buen desempeño en otro partido.

Referencia de la API utilizada: [entrenamiento con Ultralytics](https://docs.ultralytics.com/modes/train/).
