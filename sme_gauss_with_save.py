import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import os
from scipy.stats import truncnorm

# System data
x_ele = 100
true_a = 100  # real value of 'a'
x = np.linspace(0.1, 10, x_ele)
# x[1] = 0
# x[2] = 0
mu = 2 #mean
sigma = 10 # standard deviation
n_up = mu + 3*sigma
n_low = mu - 3*sigma

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
prev_ai = np.array([])  # To store ai from the previous iteration
prev_valid_a = []  # To store valid a from the previous iteration

# Initialize the plot
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(x[0] - 1, x[-1] + 1)
ax.set_ylim(np.min(y_true) - 100, np.max(y_true) + 100)
ax.set_title("")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Plot lines for ai, ai-1, and valid_a
lines_ai = [ax.plot([], [], label=f"ai[{j}]", color="blue")[0] for j in range(2)]
lines_prev_ai = [ax.plot([], [], label=f"ai-1[{j}]", color="red")[0] for j in range(2)]
lines_valid_a = [ax.plot([], [], label=f"valid_a[{j}]", color="green", linestyle=":")[0] for j in range(2)]

# Initialize the fill areas and points
fill_ai = None
fill_prev_ai = None
fill_valid_a = None

# Points that scroll
point1, = ax.plot([], [], 'yo', label=r"$y = a_{\mathrm{low}} \cdot x$")  # Yellow dot for valid_a[0]
point2, = ax.plot([], [], 'ro', label=r"$y = a_{\mathrm{up}} \cdot x$")  # Red dot for valid_a[1]

# Add legend entries for the areas
area_handles = [
    plt.Line2D([0], [0], color="blue", alpha=0.35, lw=10, label=r"$\Delta_k$"),
    plt.Line2D([0], [0], color="red", alpha=0.35, lw=10, label=r"$\Delta_{k-1}$"),
    plt.Line2D([0], [0], color="green", alpha=0.7, lw=10, label=r"$\Theta_k$")
]

ax.legend(handles=area_handles + [point1, point2], loc="upper left")

# Create output directory
output_dir = r"c:/Users/acer/Desktop/PYTHON/set_memb/PLOT"
os.makedirs(output_dir, exist_ok=True)

# Define file name
file_name = os.path.join(output_dir, f"Gaussian_truncated_noise_n_[{n_low:.2f},{n_up:.2f}]_mu_{mu}_sigma_{sigma}.gif")

# Create a PillowWriter
writer = PillowWriter(fps=30)

# Save the animation
with writer.saving(fig, file_name, dpi=100):
    for i in range(1, len(x)):  # Iterate over all elements
        if x[i] == 0:  # Skip invalid x values
            print(f"Skipping iteration {i} due to x[i] = 0")
            continue

        # Remove previous fill areas (only if they exist)
        if fill_ai:
            fill_ai.remove()
        if fill_prev_ai:
            fill_prev_ai.remove()
        if fill_valid_a:
            fill_valid_a.remove()

        # Reset the fill areas
        fill_ai = None
        fill_prev_ai = None
        fill_valid_a = None

        x_i = np.array([[x[i]]])  # dim 1,1
        y_i = np.array([[y_true[i]]])  # dim 1,1
        Ai = - H * x_i  # dim 2,1
        bi = h_n - H * y_i  # dim 2,1
        ai = (bi / Ai).flatten()  # Flatten to 1D array for easier handling
        ai = np.sort(ai)  # Sort ai

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

        # Prepare x values for the plot
        x_vals = np.linspace(np.min(x), np.max(x), x_ele)


        # Add new fill areas for this iteration
        if len(ai) == 2:
            fill_ai = ax.fill_between(x_vals, np.clip(ai[0] * x_vals, -1e6, 1e6),
                                             np.clip(ai[1] * x_vals, -1e6, 1e6),
                                             alpha=0.35, color="blue")

        if prev_ai.size == 2:
            fill_prev_ai = ax.fill_between(x_vals, np.clip(prev_ai[0] * x_vals, -1e6, 1e6),
                                                   np.clip(prev_ai[1] * x_vals, -1e6, 1e6),
                                                   alpha=0.35, color="red")

        if len(valid_a) == 2:
            fill_valid_a = ax.fill_between(x_vals, np.clip(valid_a[0] * x_vals, -1e6, 1e6),
                                                   np.clip(valid_a[1] * x_vals, -1e6, 1e6),
                                                   alpha=0.7, color="green")
            a_up = valid_a[1]
            a_low = valid_a[0]
            y_min = valid_a[0] * x[i]
            y_max = valid_a[1] * x[i]
            ax.set_title(f"Gaussian truncated noise $n \\in [{n_low:.2f}, {n_up:.2f}]$ with μ = {mu} and σ = {sigma}\n"
                         f"$a(i) \\; \\in \\; [{a_low:.2f}, {a_up:.2f}]$\n"
                         f"$x(i)$ = {x[i]:.2f} → $y \\in [{y_min:.2f}, {y_max:.2f}]$\n"
                         f"$x(i)$ = {x[i]:.2f} → $y_{{\\text{{real}}}}$: {true_a * x[i]:.2f}")

            # Update points for valid_a[0] and valid_a[1]
            y_point1 = valid_a[0] * x[i]
            y_point2 = valid_a[1] * x[i]
            point1.set_data([x[i]], [y_point1])
            point2.set_data([x[i]], [y_point2])

        # Add frame to GIF
        writer.grab_frame()

        # Save current values for the next iteration
        prev_ai = ai.copy()
        prev_valid_a = valid_a.copy()
        print(valid_a)

plt.ioff()


