import numpy as np

def pca_manual(X, n_components):
    # Centrar los datos
    X_cent = X - np.mean(X, axis=0)

    # Calcular la matriz de covarianza
    cov_matrix = np.cov(X_cent.T)

    # Obtener los vectores propios y valores propios
    valores_propios, vectores_propios = np.linalg.eig(cov_matrix)

    # Ordenar y seleccionar los principales vectores propios
    idx = np.argsort(valores_propios)[::-1]
    vectores_seleccionados = vectores_propios[:, idx[:n_components]]

    # Transformar los datos
    X_pca = X_cent.dot(vectores_seleccionados)
    return X_pca