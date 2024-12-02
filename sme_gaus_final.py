import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# System data
x_ele = 100
true_a = 100  # real value of 'a'
x = np.linspace(0, 10, x_ele)
x[1] = 0
x[2] = 0
n_up = 25
n_low = -15
mu = 0 #mean
sigma = 10 # standard deviation
c = (n_low - mu) / sigma
d = (n_up - mu) / sigma
noise = truncnorm.rvs(c, d, loc=mu, scale=sigma, size=x_ele)

y_true = true_a * x + noise

'''
H * (y - a*x) <= h_n
H * y - a * H * x <= h_n
- a * H * x <= h_n - H * y
A = - H * x 
b = h_n - H * y
A * a <= b
'''

# Constants
H = np.array([[1], [-1]])  # dim: 2,1
h_n = np.array([[n_up], [-n_low]])  # dim: 2,1

# Initialize variables
Ai_minus1 = - H * x[0]
bi_minus1 = h_n - H * y_true[0]

# Iterative calculation
for i in range(1, len(x)):  # Iterate over all elements
    x_i = np.array([[x[i]]])  # dim 1,1
    y_i = np.array([[y_true[i]]])  # dim 1,1
    Ai = - H * x_i  # dim 2,1
    bi = h_n - H * y_i  # dim 2,1

    # Controlla se ci sono valori di A pari a 0 e salta l'iterazione se presenti
    if np.any(Ai == 0):
        print(f"Iteration {i}: Skipping due to zero in A")
        continue

    # Concatenate to form A and b
    A = np.vstack((Ai, Ai_minus1))  # Concatenate vertically
    b = np.vstack((bi, bi_minus1))  # Concatenate vertically

    # Solve for a: element-wise ratio b / A
    a_values = b / A
    valid_a = []
    valid_A = []
    valid_b = []

    for idx, a in enumerate(a_values.flatten()):  # Iterate over all potential 'a' values
        satisfy_all = True

        for j in range(len(A)):  # Check if all inequalities are satisfied
            if not (A[j] * a <= b[j]):
                satisfy_all = False
                break

        if satisfy_all:  # If valid, save the values
            valid_a.append(a)
            valid_A.append(A[idx])  # Ensure 2D shape
            valid_b.append(b[idx])  # Ensure 2D shape
    valid_a = np.sort(valid_a)
    # Update Ai_minus1 and bi_minus1 with the valid values for the next iteration
    if valid_A and valid_b:  # Only update if valid values exist
        Ai_minus1 = np.vstack(valid_A)  # Stack all valid A
        bi_minus1 = np.vstack(valid_b)  # Stack all valid b
    print(f" Iteration {i}")
    print(f"a ∈ [{', '.join(f'{v:.4f}' for v in valid_a)}]")

