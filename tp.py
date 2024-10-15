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
window_size = (31, 31)
img_equalized = local_histogram_equalization(img, window_size)

# Mostrar la imagen original y la imagen ecualizada localmente
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Imagen original')
plt.subplot(1, 2, 2), plt.imshow(img_equalized, cmap='gray'), plt.title('Imagen con ecualización local')
plt.show()


# --- Ejercicio 2 (corregir examen) -------------------------------------------

# --- Cargo imagen ------------------------------------------------------------
img2 = cv2.imread('cambiar ruta',cv2.IMREAD_GRAYSCALE) 
img2.shape
plt.figure(), plt.imshow(img2, cmap='gray'), plt.show(block=False)  
#si no se especifica vmin y vmax en imshow, toma como negro el minimo valor y como blanco el max 

# Aplicar umbral para binarizar la imagen
_, img_th = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY_INV)

# Sumar los valores de los píxeles por filas (para detectar líneas horizontales)
sum_rows = np.sum(img_th, axis=1)

# Sumar los valores de los píxeles por columnas (para detectar líneas verticales)
sum_cols = np.sum(img_th, axis=0)

# Mostrar las sumas para detectar líneas
plt.plot(sum_rows), plt.title('Suma de filas (líneas horizontales)'), plt.show()
plt.plot(sum_cols), plt.title('Suma de columnas (líneas verticales)'), plt.show()

# Umbral para identificar las posiciones de las líneas (ajustar según imagen)
th_row = 2000  # Ajusta según la imagen
th_col = 2000  # Ajusta según la imagen

# Detectar las posiciones de las líneas horizontales
lines_rows = np.where(sum_rows > th_row)[0]

# Detectar las posiciones de las líneas verticales
lines_cols = np.where(sum_cols > th_col)[0]

# Crear una copia de la imagen para dibujar las líneas detectadas
img_copy = img2.copy()

# Dibujar líneas horizontales
for row in lines_rows:
    cv2.line(img_copy, (0, row), (img_copy.shape[1], row), (0, 255, 0), 1)

# Dibujar líneas verticales
for col in lines_cols:
    cv2.line(img_copy, (col, 0), (col, img_copy.shape[0]), (0, 255, 0), 1)

# Mostrar la imagen con las líneas detectadas
plt.figure(), plt.imshow(img_copy, cmap='gray'), plt.title('Líneas detectadas'), plt.show()

# Ahora que tenemos las líneas, podemos segmentar las celdas
# Recorremos cada intersección de líneas horizontales y verticales
for i in range(len(lines_rows) - 1):
    for j in range(len(lines_cols) - 1):
        # Definir la celda como el área entre dos líneas consecutivas
        cell = img2[lines_rows[i]:lines_rows[i+1], lines_cols[j]:lines_cols[j+1]]
        
        # Mostrar cada celda (opcional, solo para visualización)
        plt.figure(), plt.imshow(cell, cmap='gray'), plt.title(f'Celda {i},{j}'), plt.show()
