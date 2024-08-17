# Proyecto de Tesis : Aplicación de técnicas de pre-procesamiento en bases de conocimientos de la condición corporal de la vaca 🐄

Códigos del Proyecto de tesis donde se irán almacenando diferentes algoritmos en diferentes **Jupyter-Notebook** con un correcto **versionado** pudiendo asi lograr **mantener la integridad** de los mismos a la hora de su **futura comparación**

Nota : esto quiere decir que a pesar de que el algoritmo no funcione del todo bien y se necesite mejorar, se procederá a crear una nueva version del mismo (logrando asi su futura comparación)

## 🌳 Estructura

```bash
├───.vscode
├───db        # database BCS
├───OUTPUT    # output de los algoritmos
│
├───doc       # documentación de esos algoritmos
│
├───code      # algoritmos
│
├───lib       # requirements correspondientes a cada algoritmo
```

- cada uno de esos algoritmops ubicados en `./code/` podrá poseer su correspondiente documentacion dentro de `./doc/`

# Requisitos

- Verifica las versiones de las librerías necesarias en [requirements.txt](./requirements.txt).

- **Propiedades de la máquina para su ejecución:**
  - **Sistema operativo:** Compatible con Windows, macOS y Linux.
  - **Python:** Versión 3.x (recomendado Python 3.8 o superior).
  - **Espacio en disco:** Suficiente para almacenar los datos y resultados del proyecto.
  - **Memoria RAM:** Recomendado al menos 8 GB para una ejecución fluida.
  - **Procesador:** Se recomienda un procesador de múltiples núcleos (quad-core o superior) para un mejor rendimiento en el procesamiento de datos y entrenamiento de modelos.

- **Conexión a Internet:**
  - Se requiere una conexión a Internet estable para la instalación de dependencias y la ejecución de algoritmos que requieren acceso a servicios en línea o almacenamiento en la nube.
  - Para el uso de Google Drive u otros servicios en la nube, asegúrate de tener una conexión de alta velocidad para evitar interrupciones durante la descarga o carga de datos.
  - Para ejecutar en Google Colab, asegúrate de tener una cuenta de Google y acceso a Google Drive para almacenar y acceder a los datos.

- **Software adicional:**
  - **Gestor de entornos virtuales:** Opcional, pero recomendado para gestionar dependencias:
    - [Anaconda](https://www.anaconda.com/) o [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) (incluye paquetes preinstalados).
    - [virtualenv](https://pypi.org/project/virtualenv/) (incluido en Python 3.3+).

  - **Herramientas de desarrollo:**
    - **Editor de código:** Recomendado usar un editor compatible como [Visual Studio Code](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/), o [Jupyter Notebook](https://jupyter.org/).
    - **Sistema de control de versiones:** [Git](https://git-scm.com/) para clonar el repositorio y gestionar el versionado del código.

- **Hardware recomendado:**
  - **GPU (opcional):** Para entrenamiento de modelos de machine learning, se recomienda una GPU compatible con CUDA si se trabaja con [PyTorch](https://pytorch.org/).

# 💻 Ejecución Local

Nota: en caso de usar **visual-studio-code** se podrá los [perfiles-de-desarrollo](./vscode-profiles)

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

en local :
```batch
pip install -r requirements.txt
```
en remoto :
```batch
!pip install -r requirements.txt
```

### Recordatorio

- No ejecutar la (primera o segunda) celda de cada algoritmo (que puede representar la conexión a googleDrivel, no presente en todos los casos)
  
```python
from google.colab import drive
drive.mount('/content/drive')
```

- Asegurarse del valor de la variable de la ubicación-BASE ; ver [NOTAS de ejecución](#notas-de-ejecución) y otros [recordatorios](#recordatorio)


# ☁️ Ejecución en Google Colab

## Instalación

### Publicar el repositorio:

### Instala dependencias

Google Colab ya tiene la mayoría de estas librerías preinstaladas, pero puedes asegurarte ejecutando los siguientes comandos:

```python
!pip install [nombre de los paquetes en requirements.txt]
```

### Directorio de trabajo:

El código está diseñado para trabajar con Google Drive. Asegúrate de tener los archivos de imágenes en la carpeta especificada en Google Drive: ver [Recordatorio](#recordatorio)

### Recordatorio

- La primera celda de cada algoritmo puede servir exclusivamente como hack de GoogleColab para que este te brinde mas recursos en su maquina virtual den caso de ser necesario

```python
while True: sum(i*i for i in iter(int, 1))
```

- Recuerda incluir en la primera celda para cada algoritmo:

```python
# permisos para acceder a googleDrive
from google.colab import drive
drive.mount('/content/drive')
```



- Asegurarse del valor de la variable de la ubicación-BASE ; ver [NOTAS de ejecución](#notas-de-ejecución)

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

ver todas las librerias utilizadas para este proyecto en [requirements.txt](requirements.txt) con [python](https://www.python.org/) v3.09+

#### Biblioteca de machine learning

- [PyTorch](https://pytorch.org/) as `torch` : Framework de aprendizaje profundo para construir y entrenar modelos de machine learning.
  - [torchvision](https://pytorch.org/vision/stable/index.html) : Biblioteca para visión por computadora que incluye datasets, transformaciones y modelos preentrenados.
  - [torchvision-transforms](https://pytorch.org/vision/0.11/transforms.html) : Herramientas para realizar transformaciones de imágenes.

#### Procesamiento de imagenes

- [OpenCV](https://opencv.org/get-started/) as `cv2` :  Biblioteca para el procesamiento de imágenes y visión por computadora.
- [Pillow](https://python-pillow.org/) as `PIL` : Biblioteca para la manipulación de imágenes en Python.

#### Procesamiento y visualización de datos

- [numpy](https://numpy.org/) as `np` : Biblioteca para cálculos numéricos y manejo de arreglos multidimensionales.
- [Matplotlib](https://matplotlib.org/) as `plt` : Biblioteca para crear visualizaciones estáticas, animadas e interactivas en Python.
- [SciPy](https://scipy.org/) as `scipy` : Biblioteca que proporciona herramientas adicionales para cálculo científico y análisis numérico.
- [pandas](https://pandas.pydata.org/) as `pd` : Biblioteca para la manipulación y análisis de datos estructurados en Python.
