# Instrucciones para Correr el Ejercicio 1

Este repositorio contiene un programa en Python que implementa la técnica de ecualización local de histograma. Esta técnica permite resaltar detalles en diferentes zonas de una imagen, especialmente útil cuando las intensidades de los píxeles son similares al fondo local.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas de Python:

- OpenCV
- NumPy
- Matplotlib

## Descripción del Problema

La ecualización del histograma se puede extender para un análisis local, definiendo una ventana cuadrada o rectangular (vecindario) que se desplaza píxel a píxel. En cada ubicación, se calcula el histograma de los puntos dentro de la ventana, lo que permite obtener una transformación local de ecualización del histograma. Este proceso es útil para resaltar detalles en imágenes donde la ecualización global no proporciona buenos resultados.

### Procedimiento

1. **Definición de Ventana**: Se define una ventana de tamaño \( M \times N \).
2. **Desplazamiento de la Ventana**: La ventana se desplaza un píxel a la vez, procesando cada región de la imagen.
3. **Ecualización Local**: En cada posición de la ventana, se calcula el histograma y se aplica la transformación de ecualización para el píxel centrado en la ventana.

## Funcionalidades

- **Ecualización Local**: Implementa la ecualización local del histograma.
- **Análisis de Diferentes Tamaños de Ventana**: Permite probar la ecualización con múltiples tamaños de ventana y comparar los resultados.
- **Visualización**: Genera visualizaciones de la imagen original, la imagen ecualizada y sus respectivos histogramas.

## Ejemplo de Uso

Para utilizar el programa, carga una imagen en escala de grises y aplica la función de ecualización local con diferentes tamaños de ventana. Observa cómo varían los resultados y los detalles que se destacan en cada caso.

## Notas

- Reemplaza `'C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/Imagen_con_detalles_escondidos.tif'` por la ruta de la imagen a procesar.

---

# Instrucciones para Correr el Ejercicio 2

Este repositorio contiene un programa en Python para la corrección automática de exámenes en formato de imagen. A continuación, se detallan los pasos necesarios para ejecutar el código correctamente.

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

- OpenCV
- NumPy
- Matplotlib
- Pillow

## Uso

1. **Descargar Carpeta Exámenes**: Asegúrate de que la carpeta se descargue correctamente, copia la ruta de la carpeta y pégala en el código, reemplazando `C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examenes/`.

2. **Ejecutar el Código**: Abre tu terminal o consola de comandos, navega al directorio del proyecto y ejecuta el siguiente comando: `python codigo.py`. Esto ejecutará el script que procesa las imágenes.

3. **Resultados**: Las imágenes de las áreas detectadas (encabezado y preguntas) se guardarán en la carpeta `imagenes_examenes/`. El resultado final (con el nombre del alumno y la nota) se guardará como `notas_finales.png` en la raíz del proyecto.

## Funcionalidades

- **Detección de Bounding Boxes**: Se detectan y guardan las regiones de interés en la imagen (encabezado y preguntas).
- **Validación de Encabezado**: Verifica que el nombre, la fecha y la clase cumplan con las condiciones establecidas.
- **Extracción e Identificación de Respuestas**: Detecta la línea donde se encuentran las respuestas, las extrae y determina la letra.
- **Resultados**: Compara las respuestas detectadas con las respuestas correctas y genera un resumen con el resultado final.

## Ejemplo de Uso

A continuación, se muestra un ejemplo de cómo se vería el proceso de detección en acción:

Pregunta 1: Detectada = C, Correcta = C -> BIEN  
Pregunta 2: Detectada = B, Correcta = B -> BIEN  
...  
Resultados finales:  
8  

## Notas

- Asegúrate de reemplazar:

  `'C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/imagenes_examenes/nombre_cortado.png'` por `'ruta carpeta propia.../imagenes_examenes/nombre_cortado.png'`

  `'C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/notas_finales.png'` por `'ruta carpeta propia.../notas_finales.png'`

