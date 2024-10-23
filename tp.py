# Importamos librerías útiles para el tp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

# --- Ejercicio 1 (ecualizar histograma) --------------------------------------

def local_histogram_equalization(img, window_size):
    """Aplicamos ecualización local de histograma a la imagen con una ventana deslizante de tamaño window_size."""
    M, N = window_size
    img_equalized = np.zeros_like(img)
    
    # Añadimos borde para evitar problemas en los bordes de la imagen
    img_padded = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, borderType=cv2.BORDER_REPLICATE)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = img_padded[i:i+M, j:j+N]
            
            # Ecualización a la ventana local
            img_equalized[i, j] = cv2.equalizeHist(window)[M//2, N//2]
    
    return img_equalized

# Función para variar tamaños de ventanas
def process_multiple_window_sizes(img, window_sizes):
    """Generamos imágenes ecualizadas localmente con diferentes tamaños de ventanas."""
    for window_size in window_sizes:
        img_equalized = local_histogram_equalization(img, window_size)
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
        plt.title(f'Imagen con ecualización local (Ventana {window_size[0]}x{window_size[1]})')
        plt.axis('off')

        # Histograma de la imagen ecualizada localmente
        plt.subplot(2, 2, 4)
        plt.hist(img_equalized.flatten(), bins=256, range=[0, 256], color='gray', log=True)  # Escala logarítmica
        plt.title('Histograma - Imagen ecualizada local (escala log)')

        plt.tight_layout()
        plt.show()

# Carga de la imagen 
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Ecualización local de histograma con múltiples tamaños de ventana
window_sizes = [(7, 7), (15, 15), (31, 31)]
process_multiple_window_sizes(img, window_sizes)


# --- Ejercicio 2 (corregir examen) -------------------------------------------

# FUNCIÓN AUXILIAR 
# ---------------------------------------------

def show_image(title, image, cmap='gray'):
    """Mostramos una imagen con título y mapa de color opcional."""
    plt.figure(figsize=(10, 10))
    if cmap == 'gray':
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# PROCESAMIENTO DE IMAGEN Y DETECCIÓN DE CONTORNOS
# ---------------------------------------------

def separar_cajas(new_box, boxes, min_distance=10):
    """Verificamos si una nueva caja está demasiado cerca de las existentes."""
    x1, y1, w1, h1 = new_box
    for (x2, y2, w2, h2) in boxes:
        if abs(x1 - x2) < min_distance and abs(y1 - y2) < min_distance:
            return True
    return False

def detectar_bounding_boxes(image_path):
    """Detectamos y guardamos las bounding boxes en la imagen de entrada."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Umbralización
    _, thresh_otsu = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dilatación y erosión
    kernel = np.ones((1, 1), np.uint8)
    eroded = cv2.erode(cv2.dilate(thresh_adaptive, kernel, 2), kernel, 1)

    contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_adaptive, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtramos y ordenamos bounding boxes
    bounding_boxes = filtrar_bounding_boxes(contours_otsu, contours_adaptive)
    guardar_bounding_boxes(image, bounding_boxes)

def filtrar_bounding_boxes(contours_otsu, contours_adaptive):
    """Filtramos contornos y generamos una lista de bounding boxes."""
    bounding_boxes = []
    for contour in contours_otsu:
        x, y, w, h = cv2.boundingRect(contour)
        if not separar_cajas((x, y, w, h), bounding_boxes) and \
           (241 > w > 5) and (126 > h > 121):
            bounding_boxes.append((x, y, w, h))
    
    for contour in contours_adaptive:
        x, y, w, h = cv2.boundingRect(contour)
        if y < 100 and 5 < h < 50 and not separar_cajas((x, y, w, h), bounding_boxes):
            bounding_boxes.append((x, y, w, h))
    
    return sorted(bounding_boxes, key=lambda box: (box[0], box[1]))

def guardar_bounding_boxes(image, bounding_boxes):
    """Guardamos las regiones de interés detectadas como imágenes individuales."""
    output_dir = 'imagenes_examenes'
    os.makedirs(output_dir, exist_ok=True)

    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y + h, x:x + w]
        roi_filename = os.path.join(output_dir, f'{"encabezado" if idx == 0 else f"pregunta{idx}"}.png')
        cv2.imwrite(roi_filename, roi)

    show_image('Bounding Boxes Detectadas', image_with_boxes)


# DETECCIÓN DE LETRAS EN EL ENCABEZADO
# ---------------------------------------------

def contar_palabras(name):
    """Contamos las palabras en el nombre detectado."""
    palabras = 1
    for i in range(1, len(name)):
        espacio = name[i]['info'][0] - name[i - 1]['info'][0]
        if espacio > 13:
            palabras += 1
    return palabras

def validar_encabezado(name, date, clase):
    """Validamos que el encabezado cumpla las condiciones requeridas."""
    if len(name) <= 25 and contar_palabras(name) == 2:
        print("nombre OK")
    else:
        print("nombre MAL")

    if len(date) == 8:
        print("date OK")
    else:
        print("date MAL")

    if len(clase) == 1:
        print("class OK")
    else:
        print("class MAL")

def detectar_letras(imagen, img_bin, renglones_indxs):
    """Detectamos las letras en cada renglón."""
    letras = []
    for i in range(len(renglones_indxs) - 1):
        start, end = renglones_indxs[i][0], renglones_indxs[i + 1][0]
        renglon_img = img_bin[start:end, :]
        contours, _ = cv2.findContours(renglon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 1 and h > 2:
                letras.append({"info": (x, y, w, h)})
                cv2.rectangle(imagen, (x, start + y), (x + w, start + y + h), (0, 255, 0), 2) 

    show_image('Letras Detectadas', imagen)
    return sorted(letras, key=lambda letra: letra['info'][0])

def separar_letras(letras):
    """Separamos las letras en listas segun correspondan a nombre, fecha o clase."""
    name = [l for l in letras if 50 < l['info'][0] < 245]
    date = [l for l in letras if 290 < l['info'][0] < 364]
    clase = [l for l in letras if 411 < l['info'][0] < 542]
    return name, date, clase

def detectar_letras_y_validar_encabezado(image_path):
    """Detectamos letras en el encabezado y validamos su formato."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

    img_row_zeros = img_bin.any(axis=1)
    renglones_indxs = np.argwhere(np.diff(img_row_zeros))[::2] + 1

    letras = detectar_letras(img, img_bin, renglones_indxs)
    name, date, clase = separar_letras(letras)
    validar_encabezado(name, date, clase)

# DETECCIÓN DE LÍNEAS Y RESPUESTAS
# ---------------------------------------------

def detectar_linea_y_extraer_respuesta(image_path, output_dir):
    """Detectamos línea horizontal en cada pregunta y extraemos el texto encima."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = image[2:, :]  
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((1, 50), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    linea_negra = None
    altura_maxima = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h < 10 and h > altura_maxima:
            linea_negra = (x, y, w, h)
            altura_maxima = h

    if linea_negra:
        x, y, w, h = linea_negra
        roi_respuesta = image[max(0, y-15):y, x:x+w]
        output_filename = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_filename, roi_respuesta)
    else:
        print(f"No se detectó línea horizontal en {os.path.basename(image_path)}")

def contar_contornos_internos(imagen_bin):
    "Detectamos y devolvemos contornos internos para la detección de letras"
    imagen_bin = cv2.bitwise_not(imagen_bin)
    contornos_internos, _ = cv2.findContours(imagen_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contornos_internos

def hay_linea_vertical(contorno):
    # Aproximamos el contorno para reducir el número de puntos
    epsilon = 5
    contorno_aproximado = cv2.approxPolyDP(contorno, epsilon, True)

    # Buscamos el bounding box del contorno
    x, y, w, h = cv2.boundingRect(contorno_aproximado)

    # Verificamos si el ancho es menor que una fracción del alto, indicando una línea vertical
    if w < h / 2:
        return True  
    return False 
 
def determinar_letra_desde_respuestas(carpeta):
    respuestas_detectadas = {}
    """Determinamos las letras a partir de las imágenes procesadas."""
    for nombre_archivo in os.listdir(carpeta):
        ruta_imagen = os.path.join(carpeta, nombre_archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        
        imagen_recortada = imagen[1:-1, 5:-5] 
        _, umbral = cv2.threshold(imagen_recortada, 127, 255, cv2.THRESH_BINARY)

        contornos_internos = contar_contornos_internos(umbral)
        letra = '?'

        if len(contornos_internos) == 0:
            letra = 'No hay respuesta'
        elif len(contornos_internos) == 1:
            letra = 'C'
        elif len(contornos_internos) == 2:
            if hay_linea_vertical(contornos_internos[1]):
                letra = 'D'
            else:
                letra = 'A'
        elif len(contornos_internos) == 3:
            letra = 'B'
        else:
            letra = "Respuesta inválida"

        respuestas_detectadas[nombre_archivo] = letra

        cv2.putText(imagen, letra, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return (respuestas_detectadas)

def validar_rtas(respuestas_detectadas):
    rtas_correctas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']
    respuestas_ordenadas = [respuestas_detectadas[f'pregunta{i+1}.png'] for i in range(10)]
    resultados = []
    nota = 0
    
    for i, (rta_detectada, rta_correcta) in enumerate(zip(respuestas_ordenadas, rtas_correctas)):
        if rta_detectada == rta_correcta:
            resultados.append("OK")
            nota += 1
        else:
            resultados.append("MAL")
        print(f"Pregunta {i+1}: Detectada = {rta_detectada}, Correcta = {rta_correcta} -> {resultados[-1]}")
    
    print("Resultados finales:")
    print(nota)
    if nota >= 6:
        estado = "APROBADO"
    else:
        estado = "DESAPROBADO"

    return estado

# GENERAR IMAGEN
# ---------------------------------------------
def extraer_nombre(image_path, output_dir, top_margin=30, bottom_margin=10):
    """Detectamos líneas en la imagen y devolvemos el contenido arriba y abajo de la línea más ancha."""
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar Gaussian Blur para reducir ruido
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Ajustar el valor de umbral
    _, img_bin = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Detectar líneas usando Hough Transform
    lineas = cv2.HoughLinesP(img_bin, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)
    widest_line = None
    max_width = 0

    if lineas is not None:
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                width = abs(y2 - y1)
                if width > max_width:
                    max_width = width
                    widest_line = (y1, y2)  

    if widest_line:
        y_top = min(widest_line) - top_margin  
        y_bottom = max(widest_line) + bottom_margin  

        y_top = max(y_top, 0)
        y_bottom = min(y_bottom, img.shape[0] - 1)

        x_start = 50
        x_end = 245
        img_content = img[y_top:y_bottom, x_start:x_end] 
        
        cv2.imwrite(output_dir, img_content)
        return img_content
    else:
        print(f"No se encontraron líneas en la imagen: {image_path}")
        return None
    
def generar_imagen_con_resultado(campo_nombre, resultado, numero_examen):
    ancho, alto = 400, 70 
    notas_finales = Image.new('RGB', (ancho, alto), color='white')
    draw = ImageDraw.Draw(notas_finales)
    
    # Convertir la imagen de OpenCV a PIL
    nombre_imagen = Image.fromarray(cv2.cvtColor(campo_nombre, cv2.COLOR_BGR2RGB))
    notas_finales.paste(nombre_imagen, (110, 5)) 
    font = ImageFont.load_default()  
    draw.text((150, 35), f'Resultado: {resultado}', fill='black', font=font)

    output_dir = 'notas'
    os.makedirs(output_dir, exist_ok=True)
   
    filename = f'notas_finales{numero_examen}.png'
    print(f"Guardando la imagen en: {os.path.join(output_dir, filename)}")
    notas_finales.save(os.path.join(output_dir, filename))

    show_image('NOTAS FINALES', notas_finales)

def unir_imagenes(ruta_carpeta):
    salida = os.path.join(ruta_carpeta, 'imagen_unida.png')

    # Eliminar la imagen unida si ya existe
    if os.path.exists(salida):
        os.remove(salida)
        print(f"Imagen anterior eliminada: {salida}")

    # Obtener todas las imágenes del directorio
    imagenes = [os.path.join(ruta_carpeta, img)
                for img in os.listdir(ruta_carpeta)
                if img.endswith(('.png', '.jpg', '.jpeg'))]

    if not imagenes:
        print("Error: No se encontraron imágenes en la carpeta.")
        return

    # Cargar las imágenes con OpenCV
    imagenes_cargadas = [cv2.imread(img) for img in imagenes]

    # Verificar si alguna imagen no se cargó correctamente
    if any(img is None for img in imagenes_cargadas):
        print("Error: No se pudo cargar una o más imágenes.")
        return

    # Calcular el ancho máximo y la altura total de la imagen combinada
    ancho_maximo = max(img.shape[1] for img in imagenes_cargadas)
    altura_total = sum(img.shape[0] for img in imagenes_cargadas)

    # Crear una imagen en blanco (RGB) con el tamaño adecuado
    imagen_final = Image.new('RGB', (ancho_maximo, altura_total), color=(255, 255, 255))

    y_offset = 0  # Desplazamiento vertical
    for img in imagenes_cargadas:
        # Convertir la imagen de OpenCV (BGR) a PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imagen_final.paste(img_pil, (0, y_offset))
        y_offset += img.shape[0]  

    imagen_final.save(salida)
    print(f"Imagen final guardada como: {salida}")
    show_image('imagen final', imagen_final)


# PROGRAMA PRINCIPAL
# ---------------------------------------------
def procesar(examen, output_dir, numero_examen):
    detectar_bounding_boxes(examen)
    detectar_letras_y_validar_encabezado('imagenes_examenes/encabezado.png')
    input_dir = 'imagenes_examenes'
    output_dir = 'respuestas'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.startswith('pregunta') and filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            detectar_linea_y_extraer_respuesta(image_path, output_dir)

    carpeta_respuestas = 'respuestas'
    respuestas_detectadas = determinar_letra_desde_respuestas(carpeta_respuestas)
    print(respuestas_detectadas)
    estado = validar_rtas(respuestas_detectadas)

    image_path = 'imagenes_examenes/encabezado.png'
    output_dir = 'imagenes_examenes/nombre_cortado.png'
    nombre_cortado = extraer_nombre(image_path, output_dir)
    
    generar_imagen_con_resultado(nombre_cortado, estado, numero_examen)

    print("Proceso completado.")

exams = ['examen_1.png', 'examen_2.png', 'examen_3.png', 'examen_4.png', 'examen_5.png']
output_base_dir = 'IMAGENES_POR_EXAMEN'

for i, exam in enumerate(exams, start=1): 
    exam_path = f'examenes/{exam}'
    output_dir = os.path.join(output_base_dir, f'{os.path.splitext(exam)[0]}_output')
    procesar(exam_path, output_dir, i) 

unir_imagenes('notas')