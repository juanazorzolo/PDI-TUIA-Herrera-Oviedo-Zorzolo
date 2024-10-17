import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Función auxiliar para mostrar imágenes
def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    if cmap == 'gray':
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Cargar la imagen del examen en escala de grises
image = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_2.png', cv2.IMREAD_GRAYSCALE)

# Aplicar desenfoque para reducir ruido
blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Detectar bordes con Canny
edges = cv2.Canny(blurred, threshold1=20, threshold2=80)

# Umbralizar la imagen usando Otsu y umbralización adaptativa
_, thresh_otsu = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_adaptive = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Mostrar bordes y las imágenes umbralizadas
show_image('Bordes Detectados', edges)
show_image('Imagen Umbralizada Otsu', thresh_otsu)
show_image('Imagen Umbralizada Adaptativa', thresh_adaptive)

# Aplicar dilatación y erosión para resaltar los contornos
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh_adaptive, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Encontrar contornos
contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_adaptive, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Definir un tamaño mínimo de celda para evitar falsos positivos
min_width, min_height = 10, 15  # Ajustar según el tamaño de las celdas
bounding_boxes = []

# Filtrar contornos para las bounding boxes usando Otsu
for contour in contours_otsu:
    x, y, w, h = cv2.boundingRect(contour)
    if (w > min_width) and (h > 121):
        bounding_boxes.append((x, y, w, h))

# Filtrar contornos para las bounding boxes usando el método adaptativo
for contour in contours_adaptive:
    x, y, w, h = cv2.boundingRect(contour)
    if (w > min_width) and (h > min_height):
        aspect_ratio = w / float(h)
        if aspect_ratio > 3.5:  # Asumimos que el encabezado es mucho más ancho que alto
            bounding_boxes.insert(0, (x, y, w, h))  # Insertar encabezado primero
        else:
            bounding_boxes.append((x, y, w, h))

# Ordenar las bounding boxes por posición (primero por eje Y, luego por eje X)
bounding_boxes = sorted(set(bounding_boxes), key=lambda box: (box[1], box[0]))

# Crear un directorio para guardar las imágenes de las bounding boxes
output_dir = 'bounding_boxes_combined'
os.makedirs(output_dir, exist_ok=True)

# Mostrar las bounding boxes en la imagen original
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for idx, (x, y, w, h) in enumerate(bounding_boxes):
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Extraer la región de interés (ROI)
    roi = image[y:y+h, x:x+w]

    # Guardar la imagen de la ROI
    roi_filename = os.path.join(output_dir, f'bounding_box_{idx+1}.png')
    cv2.imwrite(roi_filename, roi)
    print(f'Imagen guardada: {roi_filename}')

# Mostrar la imagen con las bounding boxes
show_image('Bounding Boxes Detectadas', image_with_boxes)









"""PRUEBA ENCABEZADO"""
# Función para mostrar imágenes
def show_image(title, image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para detectar letras usando contornos
def detect_letters_using_contours(image):
    blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    # Aplicar erosión y dilatación
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(thresh_img, kernel, iterations=1)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    # Detectar contornos
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters = []

    # Filtrar contornos por área
    min_area = 20
    max_area = 5000
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_img = image[y:y+h, x:x+w]
            letters.append({
                "coords": (y, x, y+h, x+w),
                "img": letter_img
            })

    return letters

# Función para detectar letras usando líneas de texto
def detect_letters_using_lines(image):
    img_bin = image == 0  # Matriz booleana donde hay letras (pixeles negros)
    img_row_zeros = img_bin.any(axis=1)
    
    # Encontrar índices de líneas
    lines_idx = np.argwhere(np.diff(img_row_zeros))
    lines_idx = lines_idx.reshape(-1, 2) + 1  # Índices de inicio y fin de líneas
    
    letters = []
    for line_start, line_end in lines_idx:
        line_img = image[line_start:line_end, :]
        line_bin = line_img == 0
        col_zeros = line_bin.any(axis=0)

        # Índices de letras en cada línea
        letter_idx = np.argwhere(np.diff(col_zeros)).reshape(-1, 2) + 1
        
        # Extraer letras
        for start, end in letter_idx:
            letter_img = line_img[:, start:end]
            letters.append({
                "coords": [line_start, start, line_end, end],
                "img": letter_img
            })
    
    return letters

# Función para dibujar las letras detectadas en la imagen original
def draw_bounding_boxes(image, letters):
    plt.imshow(image, cmap='gray')
    for letter in letters:
        y1, x1, y2, x2 = letter["coords"]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.show()

# Cargar la imagen en escala de grises
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/bounding_boxes_combined/bounding_box_1.png', cv2.IMREAD_GRAYSCALE)

# Detección de letras usando contornos
letters_contours = detect_letters_using_contours(img)

# Detección de letras usando líneas de texto
letters_lines = detect_letters_using_lines(img)

# Combinar resultados de ambos métodos
combined_letters = letters_contours + letters_lines

# Dibujar las letras detectadas en la imagen original
draw_bounding_boxes(img, combined_letters)

# Guardar las letras recortadas en un directorio
output_dir = 'letras_recortadas/'
os.makedirs(output_dir, exist_ok=True)
for i, letter in enumerate(combined_letters):
    cv2.imwrite(f'{output_dir}letra_{i+1}.png', letter['img'])

print(f"Letras detectadas: {len(combined_letters)}")

# Normalización del tamaño de las letras
max_height = max([letra["img"].shape[0] for letra in combined_letters])
max_width = max([letra["img"].shape[1] for letra in combined_letters])

# Crear carpeta para letras normalizadas
normalized_output_dir = 'letras_normalizadas/'
os.makedirs(normalized_output_dir, exist_ok=True)

for i, letter in enumerate(combined_letters):
    normalized_img = np.zeros((max_height, max_width), dtype=np.uint8)
    letter_img = letter["img"]
    h, w = letter_img.shape
    y_offset = (max_height - h) // 2
    x_offset = (max_width - w) // 2
    normalized_img[y_offset:y_offset + h, x_offset:x_offset + w] = letter_img
    cv2.imwrite(f'{normalized_output_dir}letra_norm_{i+1}.png', normalized_img)

print(f"Letras normalizadas guardadas: {len(combined_letters)}")





""" ESTE LAS GUARDA UNA SOLA VEZ PERO NO DETECTA CADA PREGUNTA 
    Y NO LO PUDE AJUSTAR """

import cv2
import os
# Cargar la imagen principal
image_path = 'C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_2.png'
image = cv2.imread(image_path)

# Asegúrate de que la imagen se haya cargado correctamente
if image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {image_path}")

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado para reducir ruido
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicar umbral para binarizar la imagen (invirtiendo para obtener texto blanco sobre fondo negro)
_, threshold = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

# Detectar contornos
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear un directorio para guardar las imágenes de las bounding boxes
output_dir = 'preguntas_separadas'
os.makedirs(output_dir, exist_ok=True)

# Dibujar los contornos detectados sobre la imagen original para visualización
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Guardar la imagen con los contornos dibujados
cv2.imwrite(os.path.join(output_dir, 'contornos_detectados.png'), image_with_contours)
print(f'Imagen con contornos guardada: contornos_detectados.png')

# Ordenar los contornos de arriba a abajo, luego de izquierda a derecha
contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

# Variables para almacenar las posiciones del encabezado y las preguntas
header_image = None
questions_images = []

# Recorrer los contornos detectados y extraer las regiones correspondientes
for idx, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Filtrar por tamaño mínimo, ajusta estos valores si es necesario
    if w < 100 or h < 50:
        continue  # Ignorar áreas pequeñas que no sean relevantes

    # Extraer la región de interés (ROI)
    roi = image[y:y+h, x:x+w]

    # Asumimos que el primer contorno es el encabezado
    if idx == 0:
        header_image = roi
        header_filename = os.path.join(output_dir, 'encabezado.png')
        cv2.imwrite(header_filename, header_image)
        print(f'Encabezado guardado: {header_filename}')
    else:
        # Las siguientes son las preguntas
        questions_images.append(roi)
        question_filename = os.path.join(output_dir, f'pregunta_{idx}.png')
        cv2.imwrite(question_filename, roi)
        print(f'Pregunta guardada: {question_filename}')

# Verificación de cuántas preguntas se han detectado
print(f'Se detectaron {len(questions_images)} preguntas.')
