import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from Clasificador import MinimaDistanciaClasificador
from PCA import pca_manual

if __name__ == "__main__":

    # Cargar el conjunto de datos
    iris = load_iris()
    X, y = iris.data, iris.target

    clasificador = MinimaDistanciaClasificador()

    ''' Las dos características más relevantes según PCA, con dos clases. '''
    y_setosa_versicolor = y[0:100]
    X_setosa_versicolor = X[0:100]

    X_pca_doscaracteristicas = pca_manual(X_setosa_versicolor, 2)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_pca_doscaracteristicas, y_setosa_versicolor, test_size=0.2, random_state=42)

    # Crear y entrenar el clasificador
    clasificador.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = clasificador.predict(X_test)

    # Calcular accuracy y matriz de confusión
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Las dos características más relevantes según PCA, con dos clases")
    print("Accuracy:", accuracy)
    print("Matriz de Confusión:\n", conf_matrix)

    # Graficar los resultados de PCA
    plt.figure(figsize=(8, 6))
    for clase in np.unique(y_setosa_versicolor):
        plt.scatter(X_pca_doscaracteristicas[y_setosa_versicolor == clase, 0], X_pca_doscaracteristicas[y_setosa_versicolor == clase, 1], label=f'Clase {clase}')
    plt.title('Resultados del conjunto de datos Iris utilizando PCA con dos componentes')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()

    ''' Las dos características más relevantes según t-SNE, con dos clases. '''
    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne_doscaracteristicas = tsne.fit_transform(X_setosa_versicolor)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_tsne_doscaracteristicas, y_setosa_versicolor, test_size=0.2, random_state=42)
    
    # Crear y entrenar el clasificador
    clasificador.fit(X_train1, y_train1)

    # Realizar predicciones
    y_pred1 = clasificador.predict(X_test1)

    # Calcular accuracy y matriz de confusión
    accuracy1 = accuracy_score(y_test1, y_pred1)
    conf_matrix1 = confusion_matrix(y_test1, y_pred1)

    print("Las dos características más relevantes según t-SNE, con dos clases.")
    print("Accuracy:", accuracy1)
    print("Matriz de Confusión:\n", conf_matrix1)

    # Graficar t-SNE con 2 componentes
    plt.figure(figsize=(8, 6))
    for clase in np.unique(y_setosa_versicolor):
        plt.scatter(X_tsne_doscaracteristicas[y_setosa_versicolor == clase, 0], X_tsne_doscaracteristicas[y_setosa_versicolor == clase, 1], label=f'Clase {clase}')
    plt.title('Resultados del conjunto de datos Iris utilizando t-SNE con 2 Componentes')
    plt.xlabel('t-SNE Característica 1')
    plt.ylabel('t-SNE Característica 2')
    plt.legend()
    plt.show()