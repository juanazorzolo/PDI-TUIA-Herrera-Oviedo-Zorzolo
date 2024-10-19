import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Cargar imagen ------------------------------------------------------------
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_3.png', cv2.IMREAD_GRAYSCALE)
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)

# -----------------------------------------------------------------------------
# --- PARTE 1: Detección de renglones -----------------------------------------
# -----------------------------------------------------------------------------
img_zeros = img == 0  # Acondiciono la imagen para tener TRUE donde hay contenido
plt.figure(), plt.imshow(img_zeros, cmap='gray'), plt.show()

# Analizo filas para detectar renglones
img_row_zeros = img_zeros.any(axis=1)
img_row_zeros_idxs = np.argwhere(img_zeros.any(axis=1))
plt.figure(), plt.plot(img_row_zeros), plt.show()

# Visualizar los renglones sobre la imagen
xr = img_row_zeros * (img.shape[1] - 1)
yr = np.arange(img.shape[0])
plt.figure(), plt.imshow(img, cmap='gray'), plt.plot(xr, yr, c='r'), plt.title("Renglones"), plt.show(block=False)

# Detectar inicio y fin de cada renglón
x = np.diff(img_row_zeros)
renglones_indxs = np.argwhere(x)
len(renglones_indxs)

ii = np.arange(0, len(renglones_indxs), 2)
renglones_indxs[ii] += 1

# Visualización de inicio y fin de renglones
xri = np.zeros(img.shape[0])
xri[renglones_indxs] = (img.shape[1] - 1)
yri = np.arange(img.shape[0])
plt.figure(), plt.imshow(img, cmap='gray'), plt.plot(xri, yri, 'r'), plt.title("Renglones - Inicio y Fin"), plt.show(block=False)

# Recortar cada renglón y guardar la información
r_idxs = np.reshape(renglones_indxs, (-1, 2))

# Estructura de datos para renglones
renglones = []
for ir, idxs in enumerate(r_idxs):
    renglones.append({
        "ir": ir + 1,
        "cord": idxs,
        "img": img[idxs[0]:idxs[1], :]
    })

# Visualización de los renglones
plt.figure()
for renglon in renglones:
    plt.subplot(len(renglones), 1, renglon["ir"])
    plt.imshow(renglon["img"], cmap='gray')
    plt.title(f"Renglón {renglon['ir']}")
plt.suptitle("Renglones")
plt.show(block=False)

# -----------------------------------------------------------------------------
# --- PARTE 2: Detección de opciones (letras/objetos) en cada renglón ----------
# -----------------------------------------------------------------------------
letras = []
il = -1
for ir, renglon in enumerate(renglones):
    renglon_zeros = renglon["img"] == 0  # Acondiciono imagen del renglón
    
    # Analizo columnas del renglón para detectar opciones
    ren_col_zeros = renglon_zeros.any(axis=0)
    ren_col_zeros_idxs = np.argwhere(renglon_zeros.any(axis=0))
    
    # Encontramos inicio y final de cada opción
    x = np.diff(ren_col_zeros)
    letras_indxs = np.argwhere(x)
    ii = np.arange(0, len(letras_indxs), 2)
    letras_indxs[ii] += 1

    letras_indxs = letras_indxs.reshape((-1, 2))

    # Visualización de las opciones detectadas
    Nrows = letras_indxs.shape[0]
    plt.figure(), plt.suptitle(f"Renglón {ir + 1}")
    for ii, idxs in enumerate(letras_indxs):
        letra = renglon["img"][:, idxs[0]:idxs[1]]
        plt.subplot(Nrows, 4, ii + 1), plt.imshow(letra, cmap='gray'), plt.title(f"opción {ii + 1}")
    plt.show()
    
    # Estructura de datos para las opciones
    for irl, idxs in enumerate(letras_indxs):
        il += 1
        letras.append({
            "ir": ir + 1,
            "irl": irl + 1,
            "il": il,
            "cord": [renglon["cord"][0], idxs[0], renglon["cord"][1], idxs[1]],
            "img": renglon["img"][:, idxs[0]:idxs[1]]
        })

# --- Imagen final -----------------------------------------------------------
# Dibujar cajas alrededor de las opciones detectadas (Bounding Boxes)
plt.figure(), plt.imshow(img, cmap='gray')
for il, letra in enumerate(letras):
    yi = letra["cord"][0]
    xi = letra["cord"][1]
    W = letra["cord"][2] - letra["cord"][0]
    H = letra["cord"][3] - letra["cord"][1]
    rect = Rectangle((xi, yi), H, W, linewidth=1, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

plt.show()

# --- ANÁLISIS --------------------------------------------------------------
print(f"Se detectaron {len(letras)} opciones.")




""" 
# FUNCIÓN Y GRÁFICO PARA HISTOGRAMA ACUMULATIVO

# Función para graficar el histograma acumulativo
def plot_cumulative_histogram(image, ax, title):
   #Calcula y grafica el histograma acumulativo de una imagen.
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Calcula el histograma acumulativo
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalizar para graficar

    ax.plot(cdf_normalized, color='gray')
    ax.set_title(title)
    ax.set_xlim([0, 256])

# Mostrar la imagen original y la imagen ecualizada localmente junto a sus histogramas acumulativos
plt.figure(figsize=(12, 8))

# Imagen original y su histograma acumulativo
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(2, 2, 2)
plot_cumulative_histogram(img, plt.gca(), 'Histograma acumulativo - Imagen original')

# Imagen ecualizada localmente y su histograma acumulativo
plt.subplot(2, 2, 3)
plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen con ecualización local')
plt.axis('off')

plt.subplot(2, 2, 4)
plot_cumulative_histogram(img_equalized, plt.gca(), 'Histograma acumulativo - Imagen ecualizada local')

plt.tight_layout()
plt.show() """



"""PRUEBA JUANA (RECUADRA BIEN DE COLOR VERDE LAS DOS COLUMNAS)"""
import cv2
import numpy as np
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

# Cargar la imagen del examen
image = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_3.png', cv2.IMREAD_GRAYSCALE)

# Aplicar desenfoque para reducir ruido
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Umbralizar la imagen
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Aplicar dilatación y erosión para resaltar los contornos
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=2)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Encontrar contornos
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

# Definir un tamaño mínimo de celda para evitar falsos positivos
min_width, min_height = 15, 15  # Ajustar según el tamaño de las celdas
bounding_boxes = []

# Iterar sobre los contornos y obtener bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > min_width and h > min_height:
        bounding_boxes.append((x, y, w, h))

print(bounding_boxes)


# Ordenar las bounding boxes por posición para garantizar el orden de las preguntas
bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

# Mostrar las bounding boxes en la imagen original para ver las detecciones
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mostrar imagen con bounding boxes
show_image('Bounding Boxes Detectadas', image_with_boxes, cmap=None)


"""ÚLTIMA PRUEBA (TOMA BASTANTE BIEN EL ENCABEZADO)"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Cargo imagen ------------------------------------------------------------
img = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_3.png', cv2.IMREAD_GRAYSCALE) 
img.shape
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)

# --- Parte 1: Detección de renglones -----------------------------------------
img_zeros = img == 0  # Genero una matriz booleana (TRUE donde hay letras, que son pixeles negros)
img_row_zeros = img_zeros.any(axis=1)
x = np.diff(img_row_zeros)
renglones_indxs = np.argwhere(x)
ii = np.arange(0, len(renglones_indxs), 2)
renglones_indxs[ii] += 1
r_idxs = np.reshape(renglones_indxs, (-1, 2))  # Índices de inicio y fin de cada renglón

# Generar una estructura de datos para los renglones
renglones = []
for ir, idxs in enumerate(r_idxs):
    renglones.append({
        "ir": ir + 1,
        "cord": idxs,
        "img": img[idxs[0]:idxs[1], :]
    })

# --- Parte 2: Detección de letras en cada renglón -----------------------------
letras = []
il = -1
for ir, renglon in enumerate(renglones):
    renglon_zeros = renglon["img"] == 0  # Acondicionamiento de la imagen del renglón
    ren_col_zeros = renglon_zeros.any(axis=0)
    x = np.diff(ren_col_zeros)
    letras_indxs = np.argwhere(x)
    
    # Modifico índices de inicio
    ii = np.arange(0, len(letras_indxs), 2)
    letras_indxs[ii] += 1
    letras_indxs = letras_indxs.reshape((-1, 2))  # Índices de inicio y fin de cada letra
    
    # Estructura de datos para las letras
    for irl, idxs in enumerate(letras_indxs):
        il += 1
        letras.append({
            "ir": ir + 1,
            "irl": irl + 1,
            "il": il,
            "cord": [renglon["cord"][0], idxs[0], renglon["cord"][1], idxs[1]],  # Coordenadas en la imagen original
            "img": renglon["img"][:, idxs[0]:idxs[1]]  # Subimagen de la letra
        })

# --- Visualizar bounding boxes ------------------------------------------------
plt.figure(), plt.imshow(img, cmap='gray')
for letra in letras:
    yi = letra["cord"][0]  # Coordenada Y (inicio del renglón)
    xi = letra["cord"][1]  # Coordenada X (inicio de la letra)
    W = letra["cord"][3] - letra["cord"][1]  # Ancho de la letra
    H = letra["cord"][2] - letra["cord"][0]  # Alto del bounding box
    rect = plt.Rectangle((xi, yi), W, H, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
plt.show()

# --- Guardar letras recortadas ------------------------------------------------
output_dir = 'letras/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for letra in letras:
    idx = letra["il"] + 1
    filename = f'{output_dir}/{idx:02}.png'
    cv2.imwrite(filename, letra["img"])

# --- Normalización de tamaño de las letras ------------------------------------
max_height = max([letra["img"].shape[0] for letra in letras])
max_width = max([letra["img"].shape[1] for letra in letras])

# Crear carpeta para letras normalizadas
normalized_output_dir = 'letras_norm/'
if not os.path.exists(normalized_output_dir):
    os.makedirs(normalized_output_dir)

for letra in letras:
    idx = letra["il"] + 1
    normalized_img = np.zeros((max_height, max_width), dtype=np.uint8)
    
    # Centrar la letra en la nueva imagen
    letter_height, letter_width = letra["img"].shape
    y_offset = (max_height - letter_height) // 2
    x_offset = (max_width - letter_width) // 2
    
    # Insertar la letra centrada
    normalized_img[y_offset:y_offset + letter_height, x_offset:x_offset + letter_width] = letra["img"]
    
    # Guardar la letra normalizada
    normalized_filename = f'{normalized_output_dir}/{idx:02}n.png'
    cv2.imwrite(normalized_filename, normalized_img)

# --- Visualización final ------------------------------------------------------
print(f"Letras detectadas: {len(letras)}")



"""PRUEBA"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Cargar la imagen del examen
image = cv2.imread('C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examen_3.png', cv2.IMREAD_GRAYSCALE)

# Aplicar desenfoque para reducir ruido
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Detectar bordes con Canny
edges = cv2.Canny(blurred, 40, 110)

# Encontrar contornos a partir de los bordes detectados
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar los contornos por tamaño y forma (bounding boxes)
min_width, min_height = 100, 100  # Ajustar según el tamaño de los cuadros
bounding_boxes = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filtrar por tamaño para asegurarnos de que sea una pregunta
    if w > min_width and h > min_height:
        bounding_boxes.append((x, y, w, h))

# Ordenar las bounding boxes por posición en la imagen
bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

# Crear una carpeta para almacenar las imágenes recortadas
output_dir = "/mnt/data/preguntas_recortadas"
os.makedirs(output_dir, exist_ok=True)

# Recortar y guardar cada una de las preguntas como una imagen separada
for i, (x, y, w, h) in enumerate(bounding_boxes):
    # Recortar el recuadro detectado
    question_img = image[y:y+h, x:x+w]
    
    # Detectar líneas horizontales dentro del recuadro usando un kernel morfológico
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Kernel ancho y bajo para detectar líneas horizontales
    horizontal_lines = cv2.morphologyEx(edges[y:y+h, x:x+w], cv2.MORPH_OPEN, horizontal_kernel)

    # Encontrar contornos de las líneas horizontales dentro del recuadro
    line_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar las líneas horizontales en la imagen de la pregunta recortada
    question_img_with_lines = cv2.cvtColor(question_img, cv2.COLOR_GRAY2BGR)
    for contour in line_contours:
        lx, ly, lw, lh = cv2.boundingRect(contour)
        cv2.rectangle(question_img_with_lines, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)

    # Guardar la imagen recortada
    output_path = os.path.join(output_dir, f"pregunta_{i+1}.png")
    cv2.imwrite(output_path, question_img_with_lines)

    # Mostrar la imagen recortada con las líneas horizontales detectadas (opcional)
    show_image(f'Pregunta {i+1}', question_img_with_lines, cmap=None)

# Mostrar los contornos en la imagen original
image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Mostrar imagen con bounding boxes
show_image('Bounding Boxes Detectadas', image_with_boxes, cmap=None)


