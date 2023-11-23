import numpy as np

class MinimaDistanciaClasificador:
    def __init__(self):
        self.centroides = {}

    def fit(self, X, y):
        # Calcula los centroides para cada clase
        clases = np.unique(y)
        for clase in clases:
            self.centroides[clase] = np.mean(X[y == clase], axis=0)

    def predict(self, X):
        predicciones = []
        for x in X:
            distancias = [np.linalg.norm(x - self.centroides[clase]) for clase in self.centroides]
            clasificacion = min(enumerate(distancias), key=lambda x: x[1])[0]
            predicciones.append(clasificacion)
        return np.array(predicciones)