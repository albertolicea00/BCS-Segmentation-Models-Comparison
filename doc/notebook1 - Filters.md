# 1. Cargar y Visualizar las Imágenes

En esta sección, primero se configuran los permisos para acceder a Google Drive desde Google Colab, lo que permite trabajar con datos almacenados en la nube. Se definen las rutas a las carpetas donde se encuentran los datos de la tesis, como imágenes y otros recursos.

Luego, se carga la lista de archivos de la carpeta que contiene las imágenes de vacas para asegurarse de que se han importado correctamente. Para visualizar las imágenes, se utiliza OpenCV (`cv2`) para leerlas y luego se convierten de BGR (el formato en que OpenCV carga las imágenes) a RGB antes de mostrarlas con Matplotlib. Esto es necesario porque OpenCV y Matplotlib utilizan diferentes convenciones de color.

**Puntos clave:**
- **`cv2.imread()`**: Carga la imagen desde la ruta especificada.
- **`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`**: Convierte la imagen de BGR a RGB.
- **`plt.imshow()`**: Muestra la imagen usando Matplotlib.

# 2. Preprocesamiento de las Imágenes

El preprocesamiento es un paso crucial antes de aplicar cualquier algoritmo de segmentación, ya que mejora la calidad de las imágenes y facilita la detección de características relevantes. En esta parte:

- **Escala de grises**: La imagen se convierte a escala de grises, reduciendo la cantidad de información (de tres canales de color a uno), lo que simplifica el procesamiento posterior.
  
- **Eliminación de ruido**: Se aplica un filtro bilateral para suavizar la imagen y reducir el ruido, pero conservando los bordes. Este tipo de filtro es útil en imágenes donde se quiere preservar detalles importantes como los bordes de los objetos.

**Puntos clave:**
- **`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`**: Convierte la imagen a escala de grises.
- **`cv2.bilateralFilter()`**: Aplica un filtro bilateral para suavizar la imagen.

# 3. Aplicar Diferentes Algoritmos de Segmentación

En esta sección, se implementan y aplican varios métodos de segmentación de imágenes, cada uno con su propia estrategia para dividir la imagen en regiones de interés.

## a. Umbralización (Thresholding)

La umbralización es una técnica sencilla pero efectiva para segmentar imágenes. Se implementan dos métodos:

- **Umbralización Global (Otsu)**: Este método selecciona automáticamente un umbral para dividir la imagen en dos clases (por ejemplo, objeto y fondo).
  
- **Umbralización Adaptativa**: Este método ajusta el umbral en función de la región local de la imagen, lo que es útil en imágenes con iluminación no uniforme.

**Puntos clave:**
- **`cv2.threshold()`**: Aplica la umbralización global.
- **`cv2.adaptiveThreshold()`**: Aplica la umbralización adaptativa.

## b. Detección de Bordes (Canny)

La detección de bordes es otra técnica de segmentación que identifica bordes significativos en la imagen, donde el cambio de intensidad es alto. El algoritmo de Canny es ampliamente utilizado por su precisión y capacidad para detectar bordes relevantes.

**Puntos clave:**
- **`cv2.Canny()`**: Aplica el detector de bordes Canny.

## c. Contornos Activos (Active Contours)

Los contornos activos, o "snakes", son un método iterativo que ajusta una curva inicial a los bordes de los objetos en la imagen. Este enfoque es muy útil para segmentar formas que no se pueden capturar con métodos de umbralización o detección de bordes.

**Puntos clave:**
- **`active_contour()`**: Implementa el algoritmo de contornos activos.
- **`circle_perimeter()`**: Se usa para definir un contorno inicial circular.

# 4. Evaluar los Resultados

Finalmente, se comparan los resultados visuales de los diferentes métodos de segmentación. Esto se realiza mostrando las imágenes segmentadas lado a lado para observar cómo cada técnica divide la imagen en regiones de interés. La comparación visual es una primera aproximación para evaluar qué método puede ser más adecuado para un caso de uso específico.

**Puntos clave:**
- **`plt.subplots()`**: Crea un arreglo de gráficos para comparar múltiples imágenes.
- **`axes[].imshow()`**: Muestra cada imagen segmentada en su propio gráfico.
