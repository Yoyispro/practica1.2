import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

petal_length = iris.data[:, 2]
petal_width = iris.data[:, 3]
species = iris.target

plt.figure(figsize=(10, 6))
plt.scatter(petal_length[species == 0], petal_width[species == 0], label='Setosa', color='red')
plt.scatter(petal_length[species == 1], petal_width[species == 1], label='Versicolor', color='blue')
plt.scatter(petal_length[species == 2], petal_width[species == 2], label='Virginica', color='green')

plt.xlabel('Longitud del Pétalo (cm)')
plt.ylabel('Ancho del Pétalo (cm)')
plt.title('Gráfico de Pétalos')

plt.legend()
plt.show()