# EJERCICIO 2
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""IDENTIFICA CADA CUADRADO Y LO GENERA COMO IMAGEN"""
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
image = cv2.imread('C:/Users/morena/Desktop/FACULTAD/PDI/examenes/examen_2.png', cv2.IMREAD_GRAYSCALE)

# Aplicar desenfoque para reducir ruido
blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Detectar bordes con Canny
edges = cv2.Canny(blurred, threshold1=20, threshold2=80)

# Umbralizar la imagen usando Otsu y umbralización adaptativa
_, thresh_otsu = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh_adaptive = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

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
bounding_boxes = []

# Filtrar contornos para las bounding boxes usando Otsu PARA BOX
for contour in contours_otsu:
    x, y, w, h = cv2.boundingRect(contour)
    box = (x, y, w, h)
    if box not in bounding_boxes:
        if ((241 > w > 5) and (126 > h > 121) and ((y > 180) or (y < 70))) or (x == 324 and y == 307 and w == 242 and h == 124):
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.5:  # Asumimos que el encabezado es mucho más ancho que alto
                bounding_boxes.insert(0, (x, y, w, h))  # Insertar encabezado primero
            else:
                bounding_boxes.append((x, y, w, h))
    else:
        continue

# Filtrar contornos para las bounding boxes usando el método adaptativo PARA ENCABEZADO
for contour in contours_adaptive:
    x, y, w, h = cv2.boundingRect(contour)
    if 10 < h < 40:
        aspect_ratio = w / float(h)
        if aspect_ratio > 3.5:  # Asumimos que el encabezado es mucho más ancho que alto
            bounding_boxes.insert(0, (x, y, w, h))  # Insertar encabezado primero
        else:
            bounding_boxes.append((x, y, w, h))       

# Ordenar las bounding boxes por posición (primero por eje X, luego por eje Y)
bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0], box[1]))
print(bounding_boxes)

# Crear un directorio para guardar las imágenes de las bounding boxes
output_dir = 'bounding_boxes_combined_MORUUU'
os.makedirs(output_dir, exist_ok=True)

# Mostrar las bounding boxes en la imagen original
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for idx, (x, y, w, h) in enumerate(bounding_boxes):
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Extraer la región de interés (ROI)
    roi = image[y:y+h, x:x+w]

    # Generar el nombre del archivo
    if idx == 0:
        # Si es el primero, llamarlo "encabezado"
        roi_filename = os.path.join(output_dir, 'encabezado.png')
    else:
        # Las siguientes imágenes se llamarán "pregunta1", "pregunta2", etc.
        roi_filename = os.path.join(output_dir, f'pregunta{idx}.png')

    # Guardar la imagen de la ROI
    cv2.imwrite(roi_filename, roi)
    print(f'Imagen guardada: {roi_filename}')

# Mostrar la imagen con las bounding boxes
show_image('Bounding Boxes Detectadas', image_with_boxes)
print(f'Total de bounding boxes detectadas: {len(bounding_boxes)}')








"""IDENTIFICAR LETRAS Y VALIDAR ENCABEZADO"""
# --- Cargar la imagen del examen ---------------------------------------------
img = cv2.imread('C:/Users/morena/Desktop/FACULTAD/PDI/bounding_boxes_combined_MORUUU/encabezado.png', cv2.IMREAD_GRAYSCALE)

# --- Preprocesamiento: Umbralización Adaptativa ------------------------------
img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

# Mostrar la imagen binarizada
plt.figure(), plt.imshow(img_bin, cmap='gray'), plt.title("Imagen Binarizada"), plt.show()

# --- Detección de renglones (filas con contenido) -----------------------------
img_row_zeros = img_bin.any(axis=1)
renglones_indxs = np.argwhere(np.diff(img_row_zeros))
renglones_indxs[::2] += 1

# Filtro: renglones deben tener una altura mínima (ej. 10 píxeles)
min_renglon_height = 10
renglones = []
for i in range(0, len(renglones_indxs) - 1, 2):
    start, end = renglones_indxs[i][0], renglones_indxs[i + 1][0]
    if (end - start) >= min_renglon_height:
        renglones.append({"inicio": start, "fin": end, "img": img_bin[start:end, :]})

print(len(renglones))

# --- Detección de letras en las casillas detectadas ---------------------------
letras = []
for ir, renglon in enumerate(renglones):
    renglon_img = renglon["img"]

    # Detectar contornos dentro del renglón
    contours, _ = cv2.findContours(renglon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar contornos muy pequeños que puedan ser ruido
        if w > 1 and h > 2:
            letra = {
                "renglón": ir + 1,
                "cord": [renglon["inicio"] + y, x, renglon["inicio"] + y + h, x + w],
                "info": (x, y, w, h),
            }
            letras.append(letra)

letras = sorted(letras, key=lambda letra: letra['info'][0])
print(letras)
print(len(letras))

name = []
date = []
clase = []

for letra in letras:
    if 50 < letra['info'][0] < 245:
        name.append(letra)
    elif 290 < letra['info'][0] < 364:
        date.append(letra)
    elif 411 < letra['info'][0] < 542:
        clase.append(letra)

print(len(name))
print(len(date))
print(len(clase))

# --- Función para contar palabras en el nombre -------------------------------
def contar_palabras(name):
    palabras = 1  # Comenzamos con una palabra
    for i in range(1, len(name)):
        espacio = name[i]['info'][0] - name[i - 1]['info'][0]
        if espacio > 13:
            palabras += 1
    return palabras

# --- Validación del encabezado ------------------------------------------------
def validar_encabezado(name, date, clase):
    if len(name) <= 25 and contar_palabras(name) == 2:
        print("nombre BIEN")
    else:
        print("nombre MAL")

    if len(date) == 8:
        print("date BIEN")
    else:
        print("date MAL")

    if len(clase) == 1:
        print("class BIEN")
    else:
        print("class MAL")

# --- Visualización de las casillas detectadas en la imagen completa -----------
plt.figure(), plt.imshow(img, cmap='gray')
for l in letras:
    yi, xi, yf, xf = l["cord"]
    rect = Rectangle((xi, yi), xf - xi, yf - yi, linewidth=1, edgecolor='b', facecolor='none')
    plt.gca().add_patch(rect)

plt.title("Casillas Detectadas")
plt.show()

validar_encabezado(name, date, clase)








































"""AGARRA TODOS LOS EXAMENES, GENERA LAS IMAGENES Y DETECTA LETRAS DE ENCABEZADO""" #1 y 5 no funcionan
# Función auxiliar para mostrar imágenes
def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para procesar un examen
def process_exam(exam_file, output_dir):
    # Cargar la imagen del examen en escala de grises
    image = cv2.imread(exam_file, cv2.IMREAD_GRAYSCALE)

    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Detectar bordes con Canny
    edges = cv2.Canny(blurred, threshold1=20, threshold2=80)

    # Umbralizar la imagen usando Otsu
    _, thresh_otsu = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Aplicar dilatación y erosión para resaltar los contornos
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh_adaptive, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Encontrar contornos
    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_adaptive, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Definir un tamaño mínimo de celda para evitar falsos positivos
    min_width, min_height = 10, 15
    bounding_boxes = []

    # Filtrar contornos para las bounding boxes usando Otsu PARA BOX
    for contour in contours_otsu:
        x, y, w, h = cv2.boundingRect(contour)
        box = (x, y, w, h)
        if box not in bounding_boxes:
            if ((241 > w > 5) and (126 > h > 121) and ((y > 180) or (y < 70))) or (x == 324 and y == 307 and w == 242 and h == 124):
                aspect_ratio = w / float(h)
                if aspect_ratio > 3.5:  # Asumimos que el encabezado es mucho más ancho que alto
                    bounding_boxes.insert(0, (x, y, w, h))  # Insertar encabezado primero
                else:
                    bounding_boxes.append((x, y, w, h))
        else:
            continue

    # Filtrar contornos para las bounding boxes usando el método adaptativo PARA ENCABEZADO
    for contour in contours_adaptive:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < h < 40:
            aspect_ratio = w / float(h)
            if aspect_ratio > 3.5:  # Asumimos que el encabezado es mucho más ancho que alto
                bounding_boxes.insert(0, (x, y, w, h))  # Insertar encabezado primero
            else:
                bounding_boxes.append((x, y, w, h))       

    # Ordenar y eliminar duplicados
    bounding_boxes = sorted(set(bounding_boxes), key=lambda box: (box[1], box[0]))

    # Crear un directorio para guardar las imágenes de las bounding boxes
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

    # Identificar letras en el encabezado
    img = cv2.imread(os.path.join(output_dir, 'bounding_box_1.png'), cv2.IMREAD_GRAYSCALE)
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 3)

    # Detección de renglones
    img_row_zeros = img_bin.any(axis=1)
    renglones_indxs = np.argwhere(np.diff(img_row_zeros))
    renglones_indxs[::2] += 1

    # Filtrar renglones
    min_renglon_height = 10
    renglones = []
    for i in range(0, len(renglones_indxs) - 1, 2):
        start, end = renglones_indxs[i][0], renglones_indxs[i + 1][0]
        if (end - start) >= min_renglon_height:
            renglones.append({"inicio": start, "fin": end, "img": img_bin[start:end, :]})

    # Detección de casillas
    respuestas = []
    for ir, renglon in enumerate(renglones):
        renglon_img = renglon["img"]
        contours, _ = cv2.findContours(renglon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for ic, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 1 and h > 2:
                respuestas.append({
                    "renglón": ir + 1,
                    "casilla": ic + 1,
                    "cord": [renglon["inicio"] + y, x, renglon["inicio"] + y + h, x + w]
                })

    # Visualización de las casillas detectadas en la imagen completa
    plt.figure(), plt.imshow(img, cmap='gray')
    for resp in respuestas:
        yi, xi, yf, xf = resp["cord"]
        rect = Rectangle((xi, yi), xf - xi, yf - yi, linewidth=1, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title("Casillas Detectadas")
    plt.show()

# Listar todos los exámenes
exams = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png']
output_base_dir = 'IMAGENES_POR_EXAMEN'

# Procesar cada examen
for exam in exams:
    exam_path = f'C:/Users/morena/Desktop/FACULTAD/PDI/examenes/{exam}'
    output_dir = os.path.join(output_base_dir, f'{os.path.splitext(exam)[0]}_output')
    process_exam(exam_path, output_dir)









""" -------------DETECTAR RESPUESTAS-------------- """
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Cargar la imagen del examen ---------------------------------------------
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/bounding_boxes/pregunta1.png', cv2.IMREAD_GRAYSCALE)

# --- Preprocesamiento: Umbralización Adaptativa ------------------------------
# Esto se adapta mejor a cambios de iluminación local en la imagen.
img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

# Mostrar la imagen binarizada
plt.figure(), plt.imshow(img_bin, cmap='gray'), plt.title("Imagen Binarizada"), plt.show()

# --- Detección de bordes con Canny para encontrar la línea --------------------
edges = cv2.Canny(img_bin, 50, 150, apertureSize=3)

# Aplicar la Transformada de Hough para detectar líneas
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Visualizar las líneas detectadas sobre la imagen original
img_lines = img.copy()  # Para dibujar líneas en la imagen original
for line in lines:
    x1, y1, x2, y2 = line[0]
    # Dibujar líneas detectadas en rojo
    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.figure(), plt.imshow(img_lines, cmap='gray'), plt.title("Líneas Detectadas"), plt.show()

# --- Filtrar solo las líneas horizontales ------------------------------------
# Considerar líneas con ángulo cerca de 0 grados (horizontal)
horizontal_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y1 - y2) < 5:  # Filtrar líneas casi horizontales (diferencia mínima en Y)
        horizontal_lines.append(line[0])

# Si se detectó una línea horizontal, selecciona la primera (puedes ajustar esto)
if horizontal_lines:
    linea_horizontal = horizontal_lines[0]  # Toma la primera línea detectada
    y_linea = linea_horizontal[1]  # La coordenada Y de la línea detectada
else:
    print("No se detectaron líneas horizontales")
    y_linea = None

# --- Detección de letras en la zona de la línea detectada ---------------------
letras_sobre_linea = []
if y_linea is not None:
    margen_arriba = 12  # Puedes ajustar el margen de tolerancia
    margen_abajo = 15

    # Buscar contornos en la imagen binarizada
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar contornos que se encuentren sobre la línea
        if (y_linea - margen_abajo <= y <= y_linea - margen_arriba):
            letra = {
                "cord": [y, x, y + h, x + w],
                "info": (x, y, w, h),
            }
            letras_sobre_linea.append(letra)

# --- Visualización de las letras detectadas sobre la línea horizontal ----------
plt.figure(), plt.imshow(img, cmap='gray')
for l in letras_sobre_linea:
    yi, xi, yf, xf = l["cord"]
    rect = Rectangle((xi, yi), xf - xi, yf - yi, linewidth=1, edgecolor='b', facecolor='none')
    plt.gca().add_patch(rect)

plt.title("Letras sobre la línea detectada")
plt.show()

# Imprimir las coordenadas de las letras detectadas
print(letras_sobre_linea)
print(f"Letras detectadas: {len(letras_sobre_linea)}")




















""" --------ÚLTIMA PRUEBA CHAT--------------- """

import cv2

# Cargar cada imagen de pregunta por separado
# Cambia los nombres de los archivos de imagen según corresponda
imagenes_preguntas = ['pregunta1.png', 'pregunta2.png', 'pregunta3.png', 
                      'pregunta4.png', 'pregunta5.png', 'pregunta6.png', 
                      'pregunta7.png', 'pregunta8.png', 'pregunta9.png', 
                      'pregunta10.png']

# Configuraciones dinámicas de detección por pregunta
def procesar_pregunta(ruta_imagen):
    img = cv2.imread(ruta_imagen)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un threshold para binarizar la imagen
    _, img_bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos grandes que correspondan a respuestas
    min_area = 500  # Ajustar según el tamaño de las respuestas
    detected_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            detected_boxes.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Mostrar imagen con los contornos detectados
    cv2.imshow(f'Pregunta - {ruta_imagen}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detected_boxes

# Procesar todas las imágenes de las preguntas
for i in imagenes_preguntas:
    boxes = procesar_pregunta(f'C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/bounding_boxes/{i}') 
    print(f"Cajas detectadas: {boxes}")
