import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SkPerceptron


class Perceptron:
    def __init__(self, random_weights=True, epochs=100, activation='step'):
        self.random_weights = random_weights
        self.epochs = epochs
        self.activation = activation
        self.weights = None
        self.bias = None
        self.errors_ = []

    def _initialize_weights(self, n_features):
        if self.random_weights:
            self.weights = np.random.rand(n_features)
            self.bias = np.random.rand()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

    def _activate(self, x):
        if self.activation == 'step':
            return 1 if x >= 0 else 0
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return max(0, x)
        else:
            raise ValueError("Неизвестная функция активации")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self._activate(linear_output)
                update = (target - prediction)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        if self.activation == 'step':
            return np.where(linear_output >= 0, 1, 0)
        elif self.activation == 'sigmoid':
            return np.where(linear_output >= 0.5, 1, 0)
        elif self.activation == 'relu':
            return np.where(linear_output >= 0, 1, 0)


# Применение перцептрона для бинарной классификации
n_samples = 500
data, labels = make_blobs(n_samples=n_samples,
                          centers=([1.1, 3], [4.5, 6.9]),
                          cluster_std=1.3,
                          random_state=0)

# Визуализация данных
colours = ('green', 'orange')
fig, ax = plt.subplots()
for n_class in range(2):
    ax.scatter(data[labels == n_class][:, 0],
               data[labels == n_class][:, 1],
               c=colours[n_class],
               s=50,
               label=str(n_class))
plt.title("Исходные данные")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()

perceptron = Perceptron(random_weights=True, epochs=100, activation='step')
perceptron.fit(data, labels)
predictions = perceptron.predict(data)
accuracy = accuracy_score(labels, predictions)
print(f"Точность нашего перцептрона: {accuracy:.2f}")

# Визуализация разделяющей границы
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
for n_class in range(2):
    plt.scatter(data[labels == n_class][:, 0],
                data[labels == n_class][:, 1],
                c=colours[n_class],
                s=50,
                label=str(n_class))
plt.title("Разделяющая граница нашего перцептрона")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()

# Использование перцептрона из Scikit-Learn
sk_perceptron = SkPerceptron(max_iter=100, random_state=0)
sk_perceptron.fit(data, labels)
sk_predictions = sk_perceptron.predict(data)
sk_accuracy = accuracy_score(labels, sk_predictions)
print(f"Точность перцептрона из Scikit-Learn: {sk_accuracy:.2f}")
print(f"Разница в точности: {abs(accuracy - sk_accuracy):.2f}")

# Работа с датасетом Iris
iris = load_iris()
X_iris = iris.data[-100:]
y_iris = iris.target[-100:]
iris_perceptron = SkPerceptron(max_iter=100, random_state=0)
iris_perceptron.fit(X_iris, y_iris)
iris_perceptron_pred = iris_perceptron.predict(X_iris)
iris_perceptron_acc = accuracy_score(y_iris, iris_perceptron_pred)
svm = SVC(kernel='linear')
svm.fit(X_iris, y_iris)
svm_pred = svm.predict(X_iris)
svm_acc = accuracy_score(y_iris, svm_pred)

print(f"\nРезультаты для датасета Iris:")
print(f"Точность перцептрона: {iris_perceptron_acc:.2f}")
print(f"Точность SVM: {svm_acc:.2f}")
print(f"Разница в точности: {abs(iris_perceptron_acc - svm_acc):.2f}")
