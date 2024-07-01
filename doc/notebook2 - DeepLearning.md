# Paso 1: Configurar el Entorno
Instalación de Bibliotecas:
   - Asegúrate de tener instaladas las bibliotecas necesarias:

```python
pip install torch torchvision matplotlib Pillow
```

# Paso 2: Importar Bibliotecas
Importación de Bibliotecas
```python
import os
import torch
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
```

# Paso 3: Configuración del Modelo
Definir el Directorio de Imágenes:

- Especifica la ruta a la carpeta que contiene tus imágenes

```python
image_folder = "ruta/a/tu/directorio"
```

Cargar el Modelo Preentrenado:

  - Carga el modelo de segmentación DeepLabV3 preentrenado:

```python
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.  
segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
model.eval()
```

# Paso 4: Definir Transformaciones y Funciones de Procesamiento

Definir Transformaciones de Entrada:

- Transforma las imágenes para que sean compatibles con el modelo:

```python
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Función para Redimensionar la Máscara:

- Redimensiona la máscara segmentada para que coincida con las dimensiones originales de la imagen:

```python
def resize_mask(mask, original_size):
    mask = Image.fromarray(mask.byte().cpu().numpy()).resize(original_size, resample=Image.NEAREST)
    return np.array(mask)
```

Función para Segmentar la Imagen:

- Segmenta la imagen utilizando el modelo de DeepLabV3:

```python
def segment_image(image):
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

```

Función para Extraer la Vaca:

- Aplica la máscara segmentada a la imagen original para extraer la vaca:


```python
def extract_cow(image, mask):
    original_size = image.size
    mask = resize_mask(mask, original_size)
    image_np = np.array(image)
    extracted_image = image_np * np.expand_dims(mask, axis=2)
    return Image.fromarray(extracted_image)
```

# Paso 5: Procesar y Visualizar las Imágenes

Procesar y Visualizar las Imágenes:

- Itera sobre cada imagen en el directorio, segmenta y extrae la vaca, y muestra los resultados:


```python
for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # Segmentar la imagen
        mask = segment_image(image)
        
        # Extraer la vaca
        extracted_image = extract_cow(image, mask)
        
        # Mostrar el resultado
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(extracted_image)
        ax[1].set_title('Extracted Cow')
        ax[1].axis('off')
        
        plt.show()
```


## Resumen Final
- Configuración del entorno y bibliotecas: Instalación y configuración de las bibliotecas necesarias.
- Carga del modelo preentrenado: Uso de DeepLabV3 para la segmentación de imágenes.
- Transformaciones y funciones: Definición de las transformaciones y funciones para procesar las imágenes y extraer las vacas.
- Procesamiento y visualización: Iteración sobre las imágenes en el directorio, segmentación y visualización de los resultados.


Este flujo de trabajo te permitirá extraer vacas de tus imágenes y visualizarlas utilizando `matplotlib`.






# TÉCNICA UTILIZADA  

La técnica utilizada en el código proporcionado se llama segmentación semántica con un modelo de DeepLabV3.

Segmentación semántica es una tarea de visión por computadora que implica asignar a cada píxel de una imagen una etiqueta correspondiente a la clase a la que pertenece. En este caso, el modelo DeepLabV3 está entrenado para segmentar imágenes en diferentes clases, como vacas y fondos, utilizando una red neuronal convolucional profunda para lograr este propósito.

La arquitectura DeepLabV3 en particular utiliza una red neuronal convolucional (CNN) con una estructura de codificador-decodificador para capturar información detallada y realizar predicciones de píxeles a nivel de imagen completa. Esto permite la segmentación precisa de objetos en imágenes complejas, como las vacas en paisajes variados.

En resumen, la segmentación semántica con DeepLabV3 es la técnica empleada para extraer las vacas del fondo en las imágenes, basándose en la capacidad del modelo para identificar y etiquetar diferentes partes de la imagen con precisión.
