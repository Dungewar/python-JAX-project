import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
NUM_TERMS: int = 11
x_np = np.linspace(0, 1, NUM_TERMS)
y_np = np.tan(x_np)
plt.plot(x_np, y_np, 'o', linestyle='')

# y = b + wx (b, w)
regression_func: tuple[float, float] = (0, 0)
STEP: float = 0.1


def loss_at(n: int) -> float:
    expected_value: float = regression_func[0] + x_np[n] * regression_func[1]
    actual_value: float = y_np[n]
    return expected_value - actual_value

"""Returns MAD of the last 5 terms of the list"""
def mad(data: list[float]) -> float:
    mean: float = 0
    for i in range(data.__len__() - 5, data.__len__()):
        mean += data[i]
    mean /= 5
    mad: float = 0
    for i in range(data.__len__() - 5, data.__len__()):
        mad += abs(data[i] - mean)
    return mad

prev_mses: list[float] = []
while True:
    # print(regression_func)

    mse: float = 0
    for i in range(0, NUM_TERMS):
        mse += loss_at(i)
    mse /= NUM_TERMS
    prev_mses.append( mse)
    print(f"MSE: {mse}")

    if len(prev_mses) > 5 and mad(prev_mses) < 0.00001:
        print("Goodbye")
        break
    elif len(prev_mses) > 5:
        print(f"MAD: {mad(prev_mses)}")

    weight_derivative: float = 0
    for i in range(0, NUM_TERMS):
        weight_derivative += loss_at(i) * 2 * x_np[i]
    weight_derivative /= NUM_TERMS

    bias_derivative: float = 0
    for i in range(0, NUM_TERMS):
        bias_derivative += loss_at(i) * 2
    bias_derivative /= NUM_TERMS

    regression_func = (regression_func[0] - STEP * bias_derivative, regression_func[1] - STEP * weight_derivative)


# Plot regression line
x = np.linspace(0, 1, 100)
y = regression_func[0] + regression_func[1] * x
plt.plot(x, y, '-r', label='Regression line')
plt.legend()

plt.show()
