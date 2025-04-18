import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target  # уровень глюкозы
X = df[['bmi']].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)


# реализация линейной регрессии
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_predictions_sklearn = sklearn_model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_predictions_sklearn)


# реализация через градиентный спуск
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iter):
            y_predictions = np.dot(X, self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_predictions - y))
            db = (2 / n_samples) * np.sum(y_predictions - y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


custom_model = CustomLinearRegression()
custom_model.fit(X_train, y_train)
y_predictions_custom = custom_model.predict(X_test)
mse_custom = mean_squared_error(y_test, y_predictions_custom)


print("\nРезультаты моделей:")
print(f"Sklearn Mean Squared Error: {mse_sklearn:.2f}")
print(f"Коэффициенты sklearn: w={sklearn_model.coef_[0]:.2f}, b={sklearn_model.intercept_:.2f}")
print(f"\nCustom Mean Squared Error: {mse_custom:.2f}")
print(f"Коэффициенты custom: w={custom_model.w[0]:.2f}, b={custom_model.b:.2f}")

# визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Реальные значения')
plt.plot(X_test, y_predictions_sklearn, color='red', linewidth=2, linestyle='dashed', label='Sklearn')
plt.plot(X_test, y_predictions_custom, color='yellow', linestyle='dashed', label='Custom')
plt.xlabel('BMI')
plt.ylabel('Уровень глюкозы')
plt.title('Сравнение линейных регрессий: Sklearn vs Custom')
plt.legend()
plt.grid(True)
plt.show()

# табличка
results = pd.DataFrame({
    'Реальные значения': y_test,
    'Предсказание Sklearn': y_predictions_sklearn,
    'Предсказание Custom': y_predictions_custom
})

print("\nПредсказания (первые 20 штук):")
print(results.head(20))
