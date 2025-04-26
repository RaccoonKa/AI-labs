import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Iris и DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
for target, color in zip([0, 1, 2], ['r', 'g', 'b']):
    plt.scatter(df[df['target'] == target]['sepal length (cm)'],
                df[df['target'] == target]['sepal width (cm)'],
                c=color, label=iris.target_names[target])
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.legend()


plt.subplot(1, 2, 2)
for target, color in zip([0, 1, 2], ['r', 'g', 'b']):
    plt.scatter(df[df['target'] == target]['petal length (cm)'],
                df[df['target'] == target]['petal width (cm)'],
                c=color, label=iris.target_names[target])
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='target', palette='viridis')
plt.suptitle('Pairplot всех признаков', y=1.02)
plt.show()


df1 = df[df['target'].isin([0, 1])]
df2 = df[df['target'].isin([1, 2])]


# Обучение и оценка модели
def train_and_evaluate(dataframe, description):
    X = dataframe.drop('target', axis=1)
    y = dataframe['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность для {description}: {accuracy:.4f}")

    return clf


model1 = train_and_evaluate(df1, "setosa vs versicolor")
model2 = train_and_evaluate(df2, "versicolor vs virginica")

# Генерация данных
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1, n_clusters_per_class=1)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Сгенерированный датасет для бинарной классификации')
plt.colorbar()
plt.show()

# Обучение модели на синтетических данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
syn_clf = LogisticRegression(random_state=0)
syn_clf.fit(X_train, y_train)
y_pred = syn_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность на синтетических данных: {accuracy:.4f}")
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Предсказания модели на тестовых данных')
plt.colorbar()
plt.show()