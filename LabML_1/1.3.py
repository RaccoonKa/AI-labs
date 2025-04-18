import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# функция для вычисления Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_prediction):
    return np.mean(np.abs((y_true - y_prediction) / y_true)) * 100


diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
X = df[['bmi']].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_prediction_sklearn = sklearn_model.predict(X_test)


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
            y_prediction = np.dot(X, self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_prediction - y))
            db = (2 / n_samples) * np.sum(y_prediction - y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


custom_model = CustomLinearRegression()
custom_model.fit(X_train, y_train)
y_prediction_custom = custom_model.predict(X_test)


# вычисление метрик
def print_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\nМетрики для {model_name}:")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")


print_metrics(y_test, y_prediction_sklearn, "Sklearn LinearRegression")
print_metrics(y_test, y_prediction_custom, "Custom LinearRegression")

# визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Реальные значения')
plt.plot(X_test, y_prediction_sklearn, color='red', linewidth=2, label='Sklearn')
plt.plot(X_test, y_prediction_custom, color='green', linestyle='dashed', label='Custom')
plt.xlabel('BMI')
plt.ylabel('Уровень глюкозы')
plt.title('Сравнение моделей линейной регрессии')
plt.legend()
plt.grid(True)
plt.show()

# табличка
results = pd.DataFrame({
    'Реальные значения': y_test,
    'Sklearn': y_prediction_sklearn,
    'Custom': y_prediction_custom
})
print("\nПредсказания (первые 20 штук):")
print(results.head(20))
