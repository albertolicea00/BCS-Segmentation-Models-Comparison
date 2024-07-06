# Proyecto de Tesis

Códigos del Proyecto de tesis donde se irán almacenando diferentes algoritmos en diferentes **Jupyter-Notebook** con un correcto **versionado** pudiendo asi lograr **mantener la integridad** de los mismos a la hora de su **futura comparación**

Nota : esto quiere decir que a pesar de que el algoritmo no funcione del todo bien y se necesite mejorar, se procederá a crear una nueva version del mismo (logrando asi su futura comparación)

## 🌳 Estructura

```bash
├───.vscode
├───db        # database BCS
│
├───doc       # documentación de esos algoritmos
│
├───code      # algoritmos
│
├───lib       # requirements correspondientes a cada algoritmo
```

- cada uno de esos algoritmops ubicados en `./code/` podrá poseer su correspondiente documentacion dentro de `./doc/`

# Requisitos

<!-- TODO
- propiedades de la maquina para su ejecución

- ?? conexión a internet estable
-->

- ver versiones de las librerias en [requirements.txt]('./requirements.txt')

# 💻 Ejecución Local

Nota: en caso de usar **visual-studio-code** se podrá los [perfiles-de-desarrollo]('./vscode-profiles/')

## Pre-requisitos

1. Necesitas tener [python 3.x](https://www.python.org/) instalado en tu maquina.
2. Entonces necesitar instalar un gestor de entornos virtuales ( **opcional** ):
   - [Anaconda](https://www.anaconda.com/) ( _include almost all necessary packages_ )
   - Version Lite de Anaconda: [miniconda](https://docs.anaconda.com/free/miniconda/index.html)
   - [virtualenv](https://pypi.org/project/virtualenv/) (incluido en python 3.3) ( **recomendado** )

## Instalación

### Clonar y navega al repositorio:

```batch
git clone https://github.com/albertolicea00/tesis.git
```

```batch
cd tesis
```

### Crea un entorno virtual (opcional)

Usando **virtualenv** :

```bash
python -m venv venv
```

o usando **conda** :

```bash
conda create --venv
```

### Activa el entorno virtual (optional)

Usando **virtualenv** :

- Mac / Linux : `source venv/bin/activate`

- WSL : `source venv/Scripts/activate`

- Windows : `venv\Scripts\activate`

o usando **conda** :

- Mac / Linux / WSL : `conda activate venv`

- Windows : `activate venv`

### Instala dependencias

```batch
pip install -r requirements.txt
```

### Recordatorio

- No ejecutar la primera celda de cada algoritmo (que representa la conexión a googleDrivel, no presente en todos los casos)

```python
from google.colab import drive
drive.mount('/content/drive')
```

<!-- TODO enlace roto -->

- Asegurarse del valor de la variable de la ubicación-BASE ; ver [NOTAS de ejecución]()

```python
drive_folder = ''
```

# ☁️ Ejecución en Google Colab

## Instalación

### Publicar el repositorio:

### Instala dependencias

Google Colab ya tiene la mayoría de estas librerías preinstaladas, pero puedes asegurarte ejecutando los siguientes comandos:

```python
!pip install [nombre de los paquetes en requirements.txt]
```

### Directorio de trabajo:

El código está diseñado para trabajar con Google Drive. Asegúrate de tener los archivos de imágenes en la carpeta especificada en Google Drive: ver [Recordatorio]()

### Recordatorio

- Recuerda incluir en la primera celda para cada algoritmo:

```python
# permisos para acceder a googleDrive
from google.colab import drive
drive.mount('/content/drive')
```

<!-- TODO enlace roto -->

- Asegurarse del valor de la variable de la ubicación-BASE ; ver [NOTAS de ejecución]()

```python
drive_folder = 'drive/MyDrive/...'
```

## NOTAS de ejecución

Ya sea en en local como en una maquina remota; asegúrate de tener las variables de ubicación a la base de datos correctamente seteadas:

```python
import os
drive_folder = ''  # ubicación-BASE

# Definición de la base de datos
data_folder = drive_folder + 'db/data/'
image_folder = drive_folder + 'db/images/cow/'
shape_folder = drive_folder + 'db/images/shape/'

# si no da error es que se cargaron bien
files = os.listdir(image_folder)
```

- en caso de que la ubicación sea remota en **googleDrive** `drive_folder = 'drive/MyDrive/...'`
- en caso de que la ubicación sea local en tu Maquina `drive_folder = ''  `

## ⚙️ Tecnologías

ver todas las librerias utilizadas para este proyecto en [requirements.txt]('./requirements.txt') con [python](https://www.python.org/) v3.09+

#### Biblioteca de machine learning

- [PyTorch](https://pytorch.org/) as torch
  - [torchvision](https://pytorch.org/vision/stable/index.html)
  - [torchvision-transforms](https://pytorch.org/vision/0.11/transforms.html)

#### Procesamiento de imagenes

- [OpenCV](https://opencv.org/get-started/) as cv2
- [Pillow](https://python-pillow.org/) as PIL

#### Procesamiento y visualización de datos

- [numpy](https://numpy.org/) as np
- [Matplotlib](https://matplotlib.org/).pyplot
