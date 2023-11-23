# Clasificador de Mínima Distancia 📸

En este trabajo, se trabajará con el conjunto de datos Iris el cual es comúnmente utilizado para ilustrar y evaluar diferentes métodos de clasificación debido a que tiene tres clases (tres tipos de flores), por lo que la aplicación del clasificador de distancia mínima es relevante debido a la separación relativamente clara entre las clases. 

## Etapas de la clasificación de mínima distancia
A continuación se describen las fases de procesamiento de los datos de Iris, sobre el enfoque de la clasificación de distancia mínima.

1. **Entrenamiento:** Se calculan los centroides (medias) de cada clase (Setosa, Versicolor, y Virginica) utilizando las muestras de entrenamiento. Cada centroide representa el "punto medio" de una clase en el espacio de características. 

2. **Predicción:** Cuando se presenta una nueva flor para clasificar, se calcula la distancia euclidiana entre la flor y los centroides de todas las clases. De este modo la flor se asigna a la clase cuyo centroide está más cerca.
Es decir, el método de predicción hará la evaluación de un conjunto de datos de prueba con el límite de decisión,

# Analisis de Componentes Princípales (PCA) y t-SNE📸

Además se hace uso del enfoque de Análisis de Componentes Principales (PCA) así como de la técnica t-distributed Stochastic Neighbor Embedding (t-SNE) con el objetivo de reducir la dimensionalidad del conjunto de datos original.

## Algoritmo PCA
PCA transforma las características originales en un conjunto nuevo de características, llamadas componentes principales, que son combinaciones lineales de las características originales. Estos componentes principales están ordenados de manera que la primera captura la mayor varianza en los datos, la segunda la segunda mayor varianza, y así sucesivamente.

## Algoritmo t-SNE
A diferencia de PCA, t-SNE se centra en preservar las relaciones de vecindario entre las muestras, lo que significa que las muestras que son similares entre sí en el espacio original deberían estar cerca en el espacio reducido.
Al aplicar t-SNE al conjunto de datos Iris, se espera que se destaquen las relaciones no lineales o no linealidades en los datos que podrían no ser fácilmente capturadas por métodos lineales como PCA.

## Visualización de Resultados
Para que puedas evaluar fácilmente los resultados de cada técnicas, hemos utilizado la biblioteca Matplotlib para mostrarlos en una sola imagen. Cada resultado se presenta en un subplot separado, lo que facilita la comparación y evaluación de las diferentes técnicas utilizadas.


#### Resultados obtenidos a partir de implementar el algoritmo PCA extrayendo las dos características más relevantes, con dos clases.

<table>
  <tr>
    <td align="center">
      <img src="/ImagenesReadme/Imagen1.png" alt="Resultado 1" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Conectividad 4
    </td>
  </tr>
</table>

#### Resultados obtenidos a partir de implementar el algoritmo PCA extrayendo las dos características más relevantes, con tres clases.

<table>
  <tr>
    <td align="center">
      <img src="/ImagenesReadme/Imagen3.png" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Conectividad 8
    </td>
  </tr>
</table>

#### Resultados obtenidos a partir de implementar el algoritmo PCA extrayendo las tres características más relevantes, con dos clases.

<table>
  <tr>
    <td align="center">
      <img src="/ImagenesReadme/Imagen5.png" alt="Resultado 2" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Conectividad 8
    </td>
  </tr>
</table>

#### Resultados obtenidos a partir de implementar el algoritmo t-SNE extrayendo las dos características más relevantes, con dos clases.

<table>
  <tr>
    <td align="center">
      <img src="/ImagenesReadme/Imagen2.png" alt="Resultado 3" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Conectividad 4
    </td>
  </tr>
</table>

#### Resultados obtenidos a partir de implementar el algoritmo t-SNE extrayendo las dos características más relevantes, con tres clases.

<table>
  <tr>
    <td align="center">
      <img src="/ImagenesReadme/Imagen4.png" alt="Resultado 4" width="400"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      Conectividad 8
    </td>
  </tr>
</table>


## Cómo Usar el Programa
Aquí te proporcionamos instrucciones sobre cómo utilizar nuestro programa:
1. Clona este repositorio en tu máquina local.
2. Asegúrate de tener Python y las bibliotecas necesarias instaladas.
3. Ejecuta el programa y proporciona una imagen en escala de grises como entrada.
4. El programa aplicará las técnicas de segmentación y mostrará los resultados utilizando Matplotlib.

## Autores
Este proyecto fue realizado por un equipo de estudiantes:
| [<img src="https://avatars.githubusercontent.com/u/113084234?v=4" width=115><br><sub>Aranza Michelle Gutierrez Jimenez</sub>](https://github.com/AranzaMich) |  [<img src="https://avatars.githubusercontent.com/u/113297618?v=4" width=115><br><sub>Evelyn Solano Portillo</sub>](https://github.com/Eveeelyyyn) |  [<img src="https://avatars.githubusercontent.com/u/112792541?v=4" width=115><br><sub>Marco Castelan Rosete</sub>](https://github.com/marco2220x) | [<img src="https://avatars.githubusercontent.com/u/113079687?v=4" width=115><br><sub>Daniel Vega Rodríguez</sub>](https://github.com/DanVer2002) |
| :---: | :---: | :---: | :---: |