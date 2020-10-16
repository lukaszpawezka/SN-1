import numpy as np
from enum import Enum



class Adaline:
    def __init__(self, bias: float,
                 weights: np.ndarray,
                 alfa: float,
                 dynamic_bias: bool,
                 max_error: float
                 ):
        self.bias = bias
        self.w = weights
        self.alfa = alfa
        self.max_error = max_error
        if dynamic_bias:
            self.bias_corection_scalar = 1
        else:
            self.bias_corection_scalar = 0
        print(weights)
        self.function = lambda value: 1 if value > 0 else -1

    def predict(self, x: np.ndarray) -> float:
        return self.function(x.dot(self.w) + self.bias)

    def learn(self, D):
        # print("D",D)
        epoch = 0
        error = float('inf')
        while error > self.max_error:
            error = 0
            epoch += 1
            print(f"Epoka: {epoch}")
            for test in D:
                x, d = test
                d = self.function(d) #konieczne jest dostosowanie wartości do typu funkcji
                y = self.predict(x)
                err = (d-y)
                error += err**2

                self.bias += 2*self.alfa*err*self.bias_corection_scalar
                self.w = self.w + 2*self.alfa*err*x

            error /= len(D)

            print(f"   Błąd: {error}")

    def test(self, T):
        error_count = 0
        for test in T:
            x, d = test
            d = self.function(d)  # konieczne jest dostosowanie wartości do typu funkcji
            y = self.predict(x)
            if d - y != 0:
                error_count += 1
        print(f"Error: {error_count}/{len(T)}")




# test = True
# max = 5
# while test:
#     for i in range(max):
#         print(i)
#         if i == 9:
#             test = False
#     max += 1
#
# print()

# print(np.array([0, 1]) @ np.array([5, 1]))
print(np.array([2, 1]) * np.array([5, 6]))
