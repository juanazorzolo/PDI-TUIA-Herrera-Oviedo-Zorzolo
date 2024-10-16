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