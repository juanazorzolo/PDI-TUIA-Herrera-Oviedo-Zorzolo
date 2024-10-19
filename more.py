# EJERCICIO 2
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

""" IDENTIFICA CADA CUADRADO Y LO GENERA COMO IMAGEN"""
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
image = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_3.png', cv2.IMREAD_GRAYSCALE)

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

# Ordenar las bounding boxes por posición (primero por eje Y, luego por eje X)
bounding_boxes = sorted(set(bounding_boxes), key=lambda box: (box[1], box[0]))

# Crear un directorio para guardar las imágenes de las bounding boxes
output_dir = 'bounding_boxes_combined_MORUUU'
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
print(len(bounding_boxes))









"""IDENTIFICA LETRAS DE ENCABEZADO""" #VER PARAMETROS PORQUE JUNTA ALGUNAS
# --- Cargar la imagen del examen ---------------------------------------------
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/bounding_boxes_combined_MORUUU/bounding_box_1.png', cv2.IMREAD_GRAYSCALE)

# --- Preprocesamiento: Umbralización Adaptativa ------------------------------
# Esto se adapta mejor a cambios de iluminación local en la imagen.
img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

# Mostrar la imagen binarizada
plt.figure(), plt.imshow(img_bin, cmap='gray'), plt.title("Imagen Binarizada"), plt.show()

# --- Detección de renglones (filas con contenido) -----------------------------
img_row_zeros = img_bin.any(axis=1)  # True en filas con información
renglones_indxs = np.argwhere(np.diff(img_row_zeros))
renglones_indxs[::2] += 1  # Ajustar los índices de inicio

# Filtro: renglones deben tener una altura mínima (ej. 10 píxeles)
min_renglon_height = 10
renglones = []
for i in range(0, len(renglones_indxs) - 1, 2):
    start, end = renglones_indxs[i][0], renglones_indxs[i + 1][0]
    if (end - start) >= min_renglon_height:  # Filtro de altura
        renglones.append({"inicio": start, "fin": end, "img": img_bin[start:end, :]})

# --- Detección de casillas con contornos --------------------------------------
respuestas = []
for ir, renglon in enumerate(renglones):
    renglon_img = renglon["img"]

    # Detectar contornos dentro del renglón
    contours, _ = cv2.findContours(renglon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualizar cada renglón con los contornos detectados
    plt.figure(figsize=(10, 5))
    plt.imshow(renglon_img, cmap='gray')
    plt.title(f"Renglón {ir + 1}")

    for ic, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrar contornos muy pequeños que puedan ser ruido
        if w > 1 and h > 2:
            respuestas.append({
                "renglón": ir + 1,
                "casilla": ic + 1,
                "cord": [renglon["inicio"] + y, x, renglon["inicio"] + y + h, x + w]
            })

            # Dibujar el rectángulo alrededor de cada casilla detectada
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
            plt.gca().add_patch(rect)

    plt.show()

# --- Visualización de las casillas detectadas en la imagen completa -----------
plt.figure(), plt.imshow(img, cmap='gray')
for resp in respuestas:
    yi, xi, yf, xf = resp["cord"]
    rect = Rectangle((xi, yi), xf - xi, yf - yi, linewidth=1, edgecolor='b', facecolor='none')
    plt.gca().add_patch(rect)

plt.title("Casillas Detectadas")
plt.show()









"""AGARRA TODAS LOS EXAMENES, GENERA LAS IMAGENES Y DETECTA LETRAS DE ENCABEZADO""" #1 y 5 no funcionan
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









"""PRUEBA CORRECCION 1"""
# Función auxiliar para mostrar imágenes
def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para verificar el encabezado
def verify_header(name, date, class_number):
    # Verificar el nombre
    if len(name.split()) == 2 and len(name.replace(" ", "")) <= 25:
        name_status = "BIEN"
    else:
        name_status = "MAL"

    # Verificar la fecha
    if len(date) == 10 and date.count("/") == 2:
        date_status = "BIEN"
    else:
        date_status = "MAL"

    # Verificar el número de clase
    if class_number.isdigit() and len(class_number) > 0:
        class_status = "BIEN"
    else:
        class_status = "MAL"

    return name_status, date_status, class_status

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
            if aspect_ratio > 3.5:
                bounding_boxes.insert(0, (x, y, w, h))
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

    # Suponiendo que el orden es: nombre, fecha y clase
    if len(respuestas) >= 3:
        name = " ".join([str(respuestas[0]['casilla']), str(respuestas[1]['casilla'])])  # Reemplaza con tu lógica
        date = str(respuestas[2]['casilla'])  # Asegúrate de que esto tenga la fecha correcta
        class_number = str(respuestas[3]['casilla'])  # Y lo mismo aquí para la clase
    else:
        name, date, class_number = "", "", ""

    # Verificar el encabezado
    name_status, date_status, class_status = verify_header(name, date, class_number)

    print(f"Estado del Nombre: {name_status}")
    print(f"Estado de la Fecha: {date_status}")
    print(f"Estado de la Clase: {class_status}")

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
output_base_dir = 'bounding_boxes_combined'

# Procesar cada examen
for exam in exams:
    exam_path = f'C:/Users/morena/Desktop/FACULTAD/PDI/examenes/{exam}'
    output_dir = os.path.join(output_base_dir, f'{os.path.splitext(exam)[0]}_output')
    process_exam(exam_path, output_dir)









"""PRUEBA CORRECCION 2"""
# Función auxiliar para mostrar imágenes
def show_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Función para verificar el encabezado
def verify_header(name, date, class_number):
    # Verificar el nombre
    if len(name.split()) == 2 and len(name.replace(" ", "")) <= 25:
        name_status = "BIEN"
    else:
        name_status = "MAL"

    # Verificar la fecha
    if len(date) == 10 and date.count("/") == 2:
        date_status = "BIEN"
    else:
        date_status = "MAL"

    # Verificar el número de clase
    if class_number.isdigit() and len(class_number) > 0:
        class_status = "BIEN"
    else:
        class_status = "MAL"

    return name_status, date_status, class_status

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
            if aspect_ratio > 3.5:
                bounding_boxes.insert(0, (x, y, w, h))
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

    # Mostrar las respuestas detectadas
    print("\nCasillas detectadas en el encabezado:")
    for respuesta in respuestas:
        print(respuesta)

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
output_base_dir = 'bounding_boxes_combined'

# Procesar cada examen
for exam in exams:
    exam_path = f'C:/Users/morena/Desktop/FACULTAD/PDI/examenes/{exam}'
    output_dir = os.path.join(output_base_dir, f'{os.path.splitext(exam)[0]}_output')
    process_exam(exam_path, output_dir)
