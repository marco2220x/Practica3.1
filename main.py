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

    '''Las dos características más relevantes según PCA, con tres clases. '''

    X_pca_doscaracteristicas1 = pca_manual(X, 2)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pca_doscaracteristicas1, y, test_size=0.2, random_state=42)

    # Crear y entrenar el clasificador
    clasificador.fit(X_train2, y_train2)

    # Realizar predicciones
    y_pred2 = clasificador.predict(X_test2)

    # Calcular accuracy y matriz de confusión
    accuracy2 = accuracy_score(y_test2, y_pred2)
    conf_matrix2 = confusion_matrix(y_test2, y_pred2)

    print("Las dos características más relevantes según PCA, con tres clases.")
    print("Accuracy:", accuracy2)
    print("Matriz de Confusión:\n", conf_matrix2)

    # Graficar los resultados de PCA
    plt.figure(figsize=(8, 6))
    for clase in np.unique(y):
        plt.scatter(X_pca_doscaracteristicas1[y == clase, 0], X_pca_doscaracteristicas1[y == clase, 1], label=f'Clase {clase}')
    plt.title('Resultados del conjunto de datos Iris utilizando PCA con tres clases')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()

    '''Las dos características más relevantes según t-SNE, con tres clases.'''

    # Aplicar t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne_doscaracteristicas1 = tsne.fit_transform(X)

    # Dividir en conjuntos de entrenamiento y prueba
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_tsne_doscaracteristicas1, y, test_size=0.2, random_state=42)

    # Crear y entrenar el clasificador
    clasificador.fit(X_train3, y_train3)

    # Realizar predicciones
    y_pred3 = clasificador.predict(X_test3)

    # Calcular accuracy y matriz de confusión
    accuracy3 = accuracy_score(y_test3, y_pred3)
    conf_matrix3 = confusion_matrix(y_test3, y_pred3)

    print("Las dos características más relevantes según t-SNE, con tres clases.")
    print("Accuracy:", accuracy3)
    print("Matriz de Confusión:\n", conf_matrix3)

    # Graficar t-SNE con 2 componentes
    plt.figure(figsize=(8, 6))
    for clase in np.unique(y):
        plt.scatter(X_tsne_doscaracteristicas1[y == clase, 0], X_tsne_doscaracteristicas1[y == clase, 1], label=f'Clase {clase}')
    plt.title('Resultados del conjunto de datos Iris utilizando t-SNE con 2 Componentes')
    plt.xlabel('t-SNE Característica 1')
    plt.ylabel('t-SNE Característica 2')
    plt.legend()
    plt.show()

    '''Las tres características más relevantes según PCA, con dos clases.'''

    # Aplicar PCA al conjunto de datos Iris para 3 componentes
    X_pca_trescaracteristicas = pca_manual(X_setosa_versicolor, 3)

    # Graficar los resultados de PCA con 3 componentes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for clase in np.unique(y_setosa_versicolor):
        ax.scatter(X_pca_trescaracteristicas[y_setosa_versicolor == clase, 0], X_pca_trescaracteristicas[y_setosa_versicolor == clase, 1], X_pca_trescaracteristicas[y_setosa_versicolor == clase, 2], label=f'Clase {clase}')

    ax.set_title('Resultados del conjunto de datos Iris utilizando PCA con 3 Componentes')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    ax.legend()

    plt.show()

    # Dividir en conjuntos de entrenamiento y prueba
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X_pca_trescaracteristicas, y_setosa_versicolor, test_size=0.2, random_state=42)

    # Crear y entrenar el clasificador
    clasificador.fit(X_train4, y_train4)

    # Realizar predicciones
    y_pred4 = clasificador.predict(X_test4)

    # Calcular accuracy y matriz de confusión
    accuracy4 = accuracy_score(y_test4, y_pred4)
    conf_matrix4 = confusion_matrix(y_test4, y_pred4)

    print("Las tres características más relevantes según PCA, con dos clases.")
    print("Accuracy:", accuracy4)
    print("Matriz de Confusión:\n", conf_matrix4)