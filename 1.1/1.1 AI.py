import csv
import matplotlib.pyplot as plt
import numpy as np


def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                data.append((float(row[0]), float(row[1])))
    return data


def print_statistics(data, x_col, y_col):
    x_data = [point[x_col] for point in data]
    y_data = [point[y_col] for point in data]

    print("\nStatistics X:")
    print(f"quantity: {len(x_data)}")
    print(f"min: {min(x_data)}")
    print(f"max: {max(x_data)}")
    print(f"average: {sum(x_data) / len(x_data)}")

    print("\nstatistics Y:")
    print(f"quantity: {len(y_data)}")
    print(f"min: {min(y_data)}")
    print(f"max: {max(y_data)}")
    print(f"average: {sum(y_data) / len(y_data)}")


def manual_linear_regression(x_data, y_data):
    n = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x * y for x, y in zip(x_data, y_data))
    sum_x2 = sum(x ** 2 for x in x_data)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b


def plot_regression_line(ax, x_data, a, b, color='red'):
    x_min, x_max = min(x_data), max(x_data)
    y_min = a * x_min + b
    y_max = a * x_max + b
    ax.plot([x_min, x_max], [y_min, y_max], color=color, label='regression line')


def plot_error_squares(ax, x_data, y_data, a, b):
    for x, y in zip(x_data, y_data):
        y_pred = a * x + b
        left = min(x, x)
        right = max(x, x)
        bottom = min(y, y_pred)
        top = max(y, y_pred)

        ax.add_patch(plt.Rectangle((left, bottom),
                                   right - left,
                                   top - bottom,
                                   fill=False,
                                   edgecolor='purple',
                                   linestyle='-',
                                   alpha=0.5))
        ax.plot([x, x], [y, y_pred], color='purple', linestyle=':', alpha=0.7)


def main():
    filename = 'student_scores.csv'
    data = read_csv_file(filename)
    if not data:
        print("The file is empty or contains insufficient data")
        return

    print("Available columns: 0 и 1")
    x_col = int(input("Select a column for X (0 or 1): "))
    y_col = 1 - x_col if x_col in (0, 1) else 0

    print_statistics(data, x_col, y_col)
    x_data = [point[x_col] for point in data]
    y_data = [point[y_col] for point in data]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Data analysis and linear regression')

    ax1.scatter(x_data, y_data, color='black', label='Initial data')
    ax1.set_title('Initial data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)

    a, b = manual_linear_regression(x_data, y_data)
    print(f"\nRegression line parameters: a = {a:.4f}, b = {b:.4f}")

    ax2.scatter(x_data, y_data, color='black', label='Initial data')
    plot_regression_line(ax2, x_data, a, b)
    ax2.set_title('Initial data from the regression line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    ax3.scatter(x_data, y_data, color='black', label='Initial data')
    plot_regression_line(ax3, x_data, a, b)
    plot_error_squares(ax3, x_data, y_data, a, b)
    ax3.set_title('Regression with squared errors')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
