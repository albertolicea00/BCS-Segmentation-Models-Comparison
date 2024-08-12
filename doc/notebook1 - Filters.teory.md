### 1. **Carga y Visualización de Imágenes**

#### Carga de Imágenes
El primer paso en cualquier pipeline de procesamiento de imágenes es cargar las imágenes de la base de datos o de los archivos locales. En este caso, las imágenes se cargan desde Google Drive y se visualizan para asegurar que han sido importadas correctamente.

### 2. **Preprocesamiento de Imágenes**

El preprocesamiento es una fase crítica en el análisis de imágenes, ya que prepara los datos para ser procesados por algoritmos más avanzados. Aquí se realizaron varias técnicas de preprocesamiento:

#### a) **Conversión a Escala de Grises**
La conversión de una imagen de color (RGB) a escala de grises reduce la complejidad del procesamiento, ya que convierte una imagen tridimensional en una bidimensional. Esto es importante porque muchos algoritmos de segmentación funcionan de manera más eficiente y precisa con imágenes en escala de grises.

$$
\text{Gray} = 0.299 \times R + 0.587 \times G + 0.114 \times B
$$

donde $(R)$, $(G)$ y $(B)$ son los valores de intensidad de los canales rojo, verde y azul, respectivamente.

#### b) **Eliminación de Ruido**
El ruido en una imagen puede distorsionar los resultados de los algoritmos de segmentación. Para eliminar el ruido, se utilizó un **Filtro Bilateral**, que es una técnica avanzada que reduce el ruido mientras preserva los bordes.

$$
I_{\text{filtrado}}(x) = \frac{1}{W_p} \sum_{y \in S} I(y) \cdot \exp\left(-\frac{\|x-y\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{\|I(x) - I(y)\|^2}{2\sigma_r^2}\right)
$$

donde:
- $I(x)$ es la intensidad del píxel en la posición $(x)$.
- $(S)$ es el área de vecindad de $(x)$.
- $(\sigma_s)$ y $(\sigma_r)$ son los parámetros de la desviación estándar espacial y de intensidad.
- $(W_p)$ es un factor de normalización.

Este filtro es especialmente útil porque no difumina los bordes, lo que es esencial en la segmentación de imágenes.

### 3. **Umbralización (Thresholding)**

La **umbralización** es una técnica simple pero poderosa para la segmentación de imágenes. Se trata de convertir una imagen en escala de grises en una imagen binaria donde los píxeles son clasificados como 0 o 1 basados en un valor de umbral.

#### a) **Umbralización Global**
El método más simple de umbralización es fijar un valor umbral $(T)$. Los píxeles con valores por encima de $(T)$ se establecen en 1 (blanco), y los píxeles con valores por debajo de $(T)$ se establecen en 0 (negro).

$$
I_{\text{umbral}}(x, y) = 
\begin{cases} 
1 & \text{si } I(x, y) > T \\
0 & \text{si } I(x, y) \leq T
\end{cases}
$$

#### b) **Umbralización Adaptativa**
La **umbralización adaptativa** es una técnica más avanzada donde el valor del umbral no es fijo, sino que se calcula localmente para diferentes regiones de la imagen. Esto es útil en imágenes donde la iluminación no es uniforme.

$$
T(x, y) = \frac{1}{|S|} \sum_{(i,j) \in S} I(i,j) - C
$$

donde:
- $(S)$ es una vecindad de tamaño definido alrededor de $(x, y)$.
- $(C)$ es una constante que se resta para ajustar el umbral.

### 4. **Detección de Bordes (Canny)**

La **detección de bordes** es un proceso para identificar los puntos de una imagen donde la intensidad cambia bruscamente, lo que a menudo corresponde a los límites de los objetos dentro de la imagen.

El algoritmo de Canny es uno de los más populares y consta de varios pasos:

1. **Suavizado**: Se aplica un filtro Gaussiano para eliminar el ruido.
2. **Cálculo de Gradiente**: Se calculan los gradientes de la imagen en las direcciones $(x)$ y $(y)$ usando operadores como Sobel.

$$
G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}
$$

El gradiente de la imagen se combina para encontrar la magnitud y dirección del borde:

$$
G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)
$$

3. **Supresión No Máxima**: Se eliminan los píxeles que no están en la dirección de máxima variación.
4. **Umbralización con Histéresis**: Se aplican dos umbrales para definir los bordes más fuertes y los más débiles.

### 5. **Contornos Activos (Active Contours)**

Los **contornos activos** o **snakes** son curvas que se mueven dentro de una imagen para encontrar los contornos de los objetos. La idea es minimizar una energía asociada a la curva que la atrae hacia los bordes.

$$
E_{\text{snake}} = \int \left( \alpha \left|\frac{\partial \mathbf{s}(p)}{\partial p}\right|^2 + \beta \left|\frac{\partial^2 \mathbf{s}(p)}{\partial p^2}\right|^2 + E_{\text{ext}}(\mathbf{s}(p)) \right) dp
$$

donde:
- $(\mathbf{s}(p))$ es la posición del snake.
- $(\alpha)$ controla la tensión (suavidad) de la curva.
- $(\beta)$ controla la rigidez de la curva.
- $(E_{\text{ext}})$ es la energía externa que atrae la curva hacia los bordes.

Este método es poderoso porque se adapta a la forma de los objetos y puede segmentar estructuras complejas que otros métodos no podrían capturar tan bien.

### 6. **Evaluación de Resultados**

La evaluación de los resultados de segmentación se realizón utilizando métricas visuales.


### **Conclusión**

Este pipeline de segmentación de imágenes combina técnicas clásicas y modernas de procesamiento de imágenes. Cada método tiene sus fortalezas y debilidades, y la elección del método adecuado depende del tipo de imágenes y del objetivo específico de la segmentación. Las técnicas de preprocesamiento como la conversión a escala de grises y la eliminación de ruido son esenciales para mejorar la calidad de la segmentación. Métodos como la umbralización y la detección de bordes son efectivos en imágenes con características bien definidas, mientras que los contornos activos son más apropiados para objetos con formas complejas. La evaluación de los resultados es crucial para entender el rendimiento del modelo y ajustar los parámetros para optimizar la segmentación.


# **Elementos pendientes**

La evaluación de los resultados de segmentación se debe realizar utilizando métricas cuantitativas. Algunas de las métricas cuantitativas incluyen:

#### a) **Precisión**: 
Es la proporción de verdaderos positivos (píxeles correctamente clasificados) sobre el total de predicciones positivas.

$$
\text{Precisión} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Positivos}}
$$

#### b) **Exhaustividad** (Recall):
Es la proporción de verdaderos positivos sobre el total de positivos reales.

$$
\text{Recall} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Negativos}}
$$


#### c) **Puntuación F1**: Esta métrica es una combinación de precisión y exhaustividad (recall). La puntuación F1 se calcula como:

$$
  \text{F1} = 2 \times \frac{\text{Precisión} \times \text{Exhaustividad}}{\text{Precisión} + \text{Exhaustividad}}
$$


#### d) **IoU (Intersection over Union)**:
Índice de Jaccard es la intersección entre la predicción y el ground truth dividido por su unión. Es una métrica comúnmente utilizada en la evaluación de modelos de segmentación.

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

donde $(A)$ es la máscara predicha y $(B)$ es la máscara de ground truth.


# !! NOTA: 
Debido a los resultados insatisfactorios observados con este algoritmo, se decidió no continuar con su evaluación, quedando este método **deprecado** Los resultados mostraron una precisión deficiente en la segmentación y no cumplieron con los requisitos necesarios para el estudio. Por lo tanto, se ha optado por centrarse en métodos más prometedores y eficientes.