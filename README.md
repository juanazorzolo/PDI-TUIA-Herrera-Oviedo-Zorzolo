# Instrucciones para Correr el Ejercicio 2

Este repositorio contiene un programa de Python para la corrección automática de exámenes en formato de imagen. A continuación, se detallan los pasos necesarios para ejecutar el código correctamente.

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas:

- OpenCV
- NumPy
- Matplotlib
- Pillow

## Uso

1. **Descargar carpeta Examenes**: Asegurarse de que la carpeta se descargue correctamente, copiar la ruta de la carpeta y pegarla en el codigo remplzando "C:/Users/juana/OneDrive/Documentos/PDI1/TP PDI/examenes/".

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

- Asegúrate de remplzar:

  'D:/Users/mvovi/OneDrive/Escritorio/TUIA/2024/2so cuatri/PDI/imagenes_examenes/nombre_cortado.png' por ' ruta caprta propia..../imagenes_examenes/nombre_cortado.png'

  'D:/Users/mvovi/OneDrive/Escritorio/TUIA/2024/2so cuatri/PDI/notas_finales.png' por por ' ruta caprta propia..../notas_finales.png'
