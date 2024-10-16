import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Ejercicio 1 (ecualizar histograma) --------------------------------------

# Implementar la ecualización local del histograma
def local_histogram_equalization(img, window_size):
    """Aplica ecualización local de histograma a la imagen con una ventana deslizante de tamaño window_size."""
    # Desempaquetar el tamaño de la ventana
    M, N = window_size
    
    # Crear una imagen de salida con el mismo tamaño que la original
    img_equalized = np.zeros_like(img)
    
    # Añadir borde para evitar problemas en los bordes de la imagen
    img_padded = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, borderType=cv2.BORDER_REPLICATE)
    
    # Recorrer cada píxel de la imagen original
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Extraer la ventana local (M x N) centrada en el píxel actual
            window = img_padded[i:i+M, j:j+N]
            
            # Aplicar ecualización de histograma a la ventana local
            img_equalized[i, j] = cv2.equalizeHist(window)[M//2, N//2]
    
    return img_equalized

# Cargar la imagen en escala de grises
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/Imagen_con_detalles_escondidos.tif',cv2.IMREAD_GRAYSCALE) 

# Aplicar la ecualización local de histograma con una ventana de MXN
window_size = (15, 15)
img_equalized = local_histogram_equalization(img, window_size)

"""
ESTE ES EL PRIMER GRÁFICO QUE USE SIN HABER PUESTO EL HISTOGRAMA 

# Mostrar la imagen original y la imagen ecualizada localmente
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Imagen original')
plt.subplot(1, 2, 2), plt.imshow(img_equalized, cmap='gray'), plt.title('Imagen con ecualización local')
plt.show()

"""

# Mostrar la imagen original y la imagen ecualizada localmente junto a sus histogramas
plt.figure(figsize=(12, 8))

# Imagen original y su histograma
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(img.flatten(), bins=256, range=[0, 256], color='gray', log=True)  # Escala logarítmica
plt.title('Histograma - Imagen original (escala log)')

# Imagen ecualizada localmente y su histograma
plt.subplot(2, 2, 3)
plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen con ecualización local')
plt.axis('off')

# Modificación del histograma para usar una escala logarítmica para poder visualizar mejor y comparar entre los 2
plt.subplot(2, 2, 4)
plt.hist(img_equalized.flatten(), bins=256, range=[0, 256], color='gray', log=True)  # Escala logarítmica
plt.title('Histograma - Imagen ecualizada local (escala log)')

plt.tight_layout()
plt.show()

# --- Ejercicio 2 (corregir examen) -------------------------------------------

# --- Cargo imagen ------------------------------------------------------------

# Función auxiliar para mostrar imágenes
def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    if cmap == 'gray':
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

image = cv2.imread('C:/Users/morena/Downloads/examen_2.png', cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(image, cmap='gray'), plt.show(block=False)

# Aplicar un desenfoque para eliminar ruido
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Umbralizar la imagen para obtener una imagen binaria
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Aplicar dilatación y erosión para limpiar la imagen
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Encontrar contornos
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copiar la imagen original para dibujar las celdas
image_with_colored_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Definir un tamaño mínimo de celda para evitar que se detecte el contorno de toda la imagen
min_width, min_height = 30, 30  # Ajustar según el tamaño esperado de las celdas
min_small_width, min_small_height = 10, 10  # Para las celdas más pequeñas

# Listas para almacenar las celdas detectadas
celdas_grandes = []
celdas_pequenas = []

# Iterar sobre los contornos detectados (celdas grandes)
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Filtrar celdas grandes
    if w > min_width and h > min_height:
        # Dibujar el rectángulo alrededor de la celda grande
        color = (0, 255, 0)  # Verde para las celdas grandes
        cv2.rectangle(image_with_colored_contours, (x, y), (x + w, y + h), color, 2)

        # Almacenar la celda grande
        celdas_grandes.append({'x': x, 'y': y, 'width': w, 'height': h})

        # Crear una máscara para la celda grande
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Aplicar la máscara a la imagen original
        cell_region = cv2.bitwise_and(image, mask)

        # Convertir a escala de grises y umbral para detectar celdas más pequeñas
        gray_cell = cv2.cvtColor(cell_region, cv2.COLOR_GRAY2BGR)
        _, thresh_cell = cv2.threshold(gray_cell, 150, 255, cv2.THRESH_BINARY_INV)

        # Encontrar contornos de las celdas más pequeñas
        small_contours, _ = cv2.findContours(thresh_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterar sobre los contornos de las celdas pequeñas
        for small_contour in small_contours:
            sx, sy, sw, sh = cv2.boundingRect(small_contour)

            # Filtrar contornos pequeños
            if sw > min_small_width and sh > min_small_height:
                # Dibujar el rectángulo alrededor de la celda pequeña
                cv2.rectangle(image_with_colored_contours, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (255, 0, 0), 2)  # Rojo para celdas pequeñas

                # Almacenar la celda pequeña
                celdas_pequenas.append({'x': x + sx, 'y': y + sy, 'width': sw, 'height': sh})

# Mostrar la imagen con las celdas grandes y pequeñas detectadas
show_image('Celdas grandes y pequeñas detectadas', image_with_colored_contours)

# Imprimir las celdas detectadas
print("Celdas grandes detectadas:", celdas_grandes)
print("Celdas pequeñas detectadas:", celdas_pequenas)


""" OPCIÓN 2 """
# Encontrar contornos
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copiar la imagen original para dibujar las celdas
image_with_colored_contours = image.copy()

# Definir un tamaño mínimo de celda para evitar que se detecte el contorno de toda la imagen
min_width, min_height = 30, 30  # Ajustar según el tamaño esperado de las celdas

# Iterar sobre los contornos detectados
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Filtrar contornos muy pequeños o muy grandes que no son celdas
    if w > min_width and h > min_height:
        # Asignar el color verde a todas las celdas
        color = (0, 255, 0)  # Verde para todas las celdas
        # Dibujar el rectángulo alrededor de la celda
        cv2.rectangle(image_with_colored_contours, (x, y), (x + w, y + h), color, 2)

# Mostrar la imagen con las celdas detectadas y coloreadas en verde
show_image('Celdas detectadas en verde', image_with_colored_contours)