class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        derivative = lambda x : 2*x 
        function = lambda x : x**2
        x_old = init
        for iteration_count in range(iterations):
            x_new = x_old - learning_rate * derivative(x_old)
            x_old = x_new 
        return round(x_old, 5)