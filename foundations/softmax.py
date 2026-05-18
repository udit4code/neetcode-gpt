import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        e_x = np.exp(x - np.max(x))
        denominator = np.sum(e_x, axis=-1, keepdims=True)
        result = e_x / denominator
        return np.round(result, 4)
