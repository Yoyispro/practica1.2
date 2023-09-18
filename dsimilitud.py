import matplotlib.pyplot as plt
from sklearn import datasets

# Cargar el conjunto de datos Iris desde Scikit-Learn
iris = datasets.load_iris()

# Obtener las características de sépalo y las etiquetas de especies
sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]
species = iris.target

# Crear un gráfico de dispersión con colores según las especies
plt.figure(figsize=(10, 6))
plt.scatter(sepal_length[species == 0], sepal_width[species == 0], label='Setosa', color='red')
plt.scatter(sepal_length[species == 1], sepal_width[species == 1], label='Versicolor', color='blue')
plt.scatter(sepal_length[species == 2], sepal_width[species == 2], label='Virginica', color='green')

# Etiquetas de los ejes y título del gráfico
plt.xlabel('Longitud del Sépalo (cm)')
plt.ylabel('Ancho del Sépalo (cm)')
plt.title('Gráfico de Sépalos de Iris por Especie')

# Leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
