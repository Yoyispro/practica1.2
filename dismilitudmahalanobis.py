import pandas as pd
import random
import numpy as np
from collections import Counter
from scipy.spatial import distance

# Cargar la base de datos Iris desde Scikit-Learn
iris = datasets.load_iris()

# Obtener las características y las etiquetas de especies
data = iris.data
species = iris.target

# Función para calcular la distancia Mahalanobis entre dos puntos
def mahalanobis_distance(x, y, cov_inv):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff))

# Calcular la matriz de covarianza de los datos
cov_matrix = np.cov(data, rowvar=False)
cov_inv = np.linalg.inv(cov_matrix)

# Obtener índices de las filas correspondientes a cada clase
indices_clase_1 = range(1, 51)  # Clase 1: filas del 2 al 51
indices_clase_2 = range(51, 101)  # Clase 2: filas del 52 al 101
indices_clase_3 = range(101, 151)  # Clase 3: filas del 102 al 151

# Seleccionar de forma aleatoria dos puntos de cada clase
random.seed(42)  # Para reproducibilidad
puntos_seleccionados = []
puntos_seleccionados.extend(random.sample(indices_clase_1, 2))
puntos_seleccionados.extend(random.sample(indices_clase_2, 2))
puntos_seleccionados.extend(random.sample(indices_clase_3, 2))

# Calcular las distancias Euclidianas entre los puntos seleccionados y el resto de los puntos
distancias = {}
for punto_seleccionado in puntos_seleccionados:
    distancias[punto_seleccionado] = []
    for i in range(len(data)):
        if punto_seleccionado != i:
            distancia = distance.euclidean(data[punto_seleccionado], data[i])
            distancias[punto_seleccionado].append((i, distancia))

# Imprimir las distancias
for punto_seleccionado, distancias_punto in distancias.items():
    print(f'Distancias para el punto {punto_seleccionado}:')
    for i, distancia in distancias_punto:
        print(f'  Punto {i}: {distancia}')

# Definir el valor de k para la clasificación
k = 3

# Realizar la clasificación por votación para cada punto seleccionado
class_votes = []
for punto_seleccionado, distancias_punto in distancias.items():
    neighbors = sorted(distancias_punto, key=lambda x: x[1])[:k]
    neighbor_classes = [species[i] for i, _ in neighbors]
    class_counter = Counter(neighbor_classes)
    most_common_class = class_counter.most_common(1)[0][0]
    class_votes.append((punto_seleccionado, most_common_class))

# Ordenar los resultados por punto seleccionado
class_votes = sorted(class_votes, key=lambda x: x[0])

# Imprimir la clase estimada para cada punto seleccionado
for punto_seleccionado, clase_estimada in class_votes:
    print(f'Punto {punto_seleccionado}: Clase estimada {clase_estimada}')
