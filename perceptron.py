import numpy as np
from enum import Enum


class ThresholdType(Enum):
    UNIPOLAR = "unipolar"
    BIPOLAR = "bipolar"


class Perceptron:
    def __init__(self, bias: float,
                 weights: np.ndarray,
                 function: ThresholdType,
                 alfa: float,
                 dynamic_bias: bool,
                 ):
        self.bias = bias
        self.w = weights
        self.alfa = alfa
        if dynamic_bias:
            self.bias_corection_scalar = 1
        else:
            self.bias_corection_scalar = 0
        print(weights)
        if function == ThresholdType.UNIPOLAR:
            self.function = lambda value: 1 if value > 0 else 0
        else:
            self.function = lambda value: 1 if value > 0 else -1

    def predict(self, x: np.ndarray) -> float:
        return self.function(x.dot(self.w) + self.bias)

    def learn(self, D):
        # print("D",D)
        epoch = 0
        has_error = True
        while has_error:
            has_error = False
            epoch += 1
            error_count = 0
            print(f"Epoka: {epoch}")
            for test in D:
                x, d = test
                d = self.function(d) #konieczne jest dostosowanie wartości do typu funkcji
                y = self.predict(x)
                err = d-y
                delta_w = err*x
                self.bias += err*self.alfa*self.bias_corection_scalar
                self.w = self.w + self.alfa*delta_w

                if err != 0:
                    has_error = True
                    error_count += 1
            print(f"   Błąd: {float(error_count)/len(D)*101}%")

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
