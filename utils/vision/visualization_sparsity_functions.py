import os

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from utils.general.helper_path import FIGURES_PATH
plt.style.use("ggplot")

# Here I want to reconstruct and reproduce the Figure 1 from the following paper:
# Sparse Signal Recovery via Generalized Entropy Function Minimization
# Available here: https://ieeexplore.ieee.org/ielaam/78/8611004/8590738-aam.pdf?tag=1


def generate_2d_samples(n_samples, min, max):
    # Generate grid points
    n_points = n_samples
    x1_values = np.linspace(min, max, n_points)
    x2_values = np.linspace(min, max, n_points)
    X1, X2 = np.meshgrid(x1_values, x2_values)

    return X1, X2


# Define the functions here
def shannon_entropy_function(x, p):
    """
    Imports an array of n x d, where n is number of samples and d is dimensionality
    :param x: array of n x d, n = number of samples, d is dimensionality
    :param p: order of the norm
    :return: shannon's entropy
    """
    assert p > 0., "Please input a p greater than 0!"
    norm = np.sum(np.abs(x) ** p)
    numerator = np.abs(x) ** p
    ratio = numerator / norm

    shannon_entropy = - np.sum(ratio * np.log(ratio))

    return np.round(shannon_entropy, 2)


def lp_norm(x, p):
    """
    Imports an array of n x d, where n is number of samples and d is dimensionality
    :param x: array of n x d, n = number of samples, d is dimensionality
    :param p: order of the norm
    :return: lp norm
    """
    assert p > 0
    norm = np.sum(np.abs(x) ** p)
    return np.round(norm, 2)


def renyi_alpha_function(x, alpha=0.9, p=0.8):
    """
    Imports an array of n x d, where n is number of samples and d is dimensionality
    :param x: array of n x d, n = number of samples, d is dimensionality
    :param alpha: alpha
    :param p: order of the norm
    :return: lp no

    Remark paper: alpha usually [0.9, 1) (1, 1.1]
    Remark paper: p range [0.8, 1.3]
    """
    assert p > 0
    assert alpha > 0 and alpha != 1

    norm = LA.norm(x, p) ** p # across all dimensions
    numerator = np.abs(x) ** p
    ratio = (numerator / norm) ** alpha

    renyi_entropy = (1 / (1 - alpha)) * np.log(np.sum(ratio))

    return np.round(renyi_entropy, 2)


def logarithm_of_energy(x):

    #As x needs to be != 0
    numerical_constant = 1e-7
    energy = 2 * np.sum(np.log(np.abs(x) + numerical_constant))

    return np.round(energy, 2)


# Hyperparameters
n_samples = 100
min = -1
max = 1
X1, X2 = generate_2d_samples(n_samples, min, max)

p_val_shannon = [1., 2.]
p_val_norm = [1, 0.5]

# Create a single figure with multiple subplots
fig, axes = plt.subplots(3, 2, figsize=(9, 9))

#Calculate Shannon entropy for each point on the grid for each p value and plot
#Numpy version
for i, p in enumerate(p_val_shannon):
    shannon_entropy_values = np.zeros_like(X1)
    for j in range(n_samples):
        for k in range(n_samples):
            shannon_entropy_values[j, k] = shannon_entropy_function([X1[j, k], X2[j, k]], p)

    # Plot the results
    contour = axes[i, 0].contourf(X1, X2, shannon_entropy_values, levels=50, cmap='magma')
    axes[i, 0].set_xlabel("$x_{1}$")
    axes[i, 0].set_ylabel("$x_{2}$")
    axes[i, 0].set_title(f'Generalized Shannon Entropy Function (SEF) \n$p={p}$', fontsize=10)

    # Add levels on the right side of each subplot
    cb = fig.colorbar(contour, ax=axes[i, 0], orientation='vertical', label='value')
    cb.ax.yaxis.set_label_position('right')  # Move label to the right side
    cb.ax.yaxis.label.set_fontsize(10)
    cb.set_ticks(contour.levels[::5])  # Set ticks to contour levels
    # Set fontsize for colorbar labels and ticks
    cb.ax.tick_params(labelsize=10)  # Adjust the fontsize as needed

# Calculate Lp norm for each point on the grid for each p value and plot
for i, p in enumerate(p_val_norm):
    lp_norm_values = np.zeros_like(X1)
    for j in range(n_samples):
        for k in range(n_samples):
            lp_norm_values[j, k] = lp_norm([X1[j, k], X2[j, k]], p)

    # Plot the results
    contour = axes[i, 1].contourf(X1, X2, lp_norm_values, levels=50, cmap='magma')
    axes[i, 1].set_xlabel("$x_{1}$")
    axes[i, 1].set_ylabel("$x_{2}$")
    axes[i, 1].set_title(f'$L_{{{p}}}$-norm', fontsize=10)

    # Add levels on the right side of each subplot
    cb = fig.colorbar(contour, ax=axes[i, 1], orientation='vertical', label='value')
    cb.ax.yaxis.set_label_position('right')  # Move label to the right side
    cb.ax.yaxis.label.set_fontsize(10)
    cb.set_ticks(contour.levels[::5])  # Set ticks to contour levels
    # Set fontsize for colorbar labels and ticks
    cb.ax.tick_params(labelsize=10)  # Adjust the fontsize as needed

# Plot additional functions in the last two subplots
for i, func_title in enumerate(['Renyi entropy function (REF) \n$\\alpha$=0.9 $p$=0.8', 'Logarithm of energy']):
    values = np.zeros_like(X1)
    for j in range(n_samples):
        for k in range(n_samples):
            if i == 0:
                values[j, k] = renyi_alpha_function([X1[j, k], X2[j, k]])
            else:
                values[j, k] = logarithm_of_energy([X1[j, k], X2[j, k]])

    # Plot the results
    contour = axes[2, i].contourf(X1, X2, values, levels=50, cmap='magma')
    axes[2, i].set_xlabel("$x_{1}$")
    axes[2, i].set_ylabel("$x_{2}$")
    axes[2, i].set_title(func_title, fontsize=10)

    # Add levels on the right side of each subplot
    cb = fig.colorbar(contour, ax=axes[2, i], orientation='vertical', label='value')
    cb.ax.yaxis.set_label_position('right')  # Move label to the right side
    cb.ax.yaxis.label.set_fontsize(10)
    cb.set_ticks(contour.levels[::5])  # Set ticks to contour levels
    # Set fontsize for colorbar labels and ticks
    cb.ax.tick_params(labelsize=10)  # Adjust the fontsize as needed

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "visualization sparsity-promoting regularization functions.png"), dpi=400, format="png")
plt.show()
plt.close()


# Let's do it in 1D plots
# Function to calculate logarithm of energy
def one_d_logarithm_of_energy(x):
    numerical_constant = 1e-7
    return 2 * np.log(x+numerical_constant)


def one_d_log_penalty(x):
    return np.log((np.abs(x) / 0.01) + 1) / np.log((1. / 0.01) + 1)


# Function to calculate Lp norm
def one_d_lp_norm(x, p):
    return np.power(np.abs(x), p)


def one_d_l0_norm(x):
    return np.where(x == 0., 0., 1.)


def gaussian_entropy(x):
    # This is basically the log energy!
    numerical_constant = 1e-15
    return np.log(np.abs(x + numerical_constant) ** 2)


x_min = 0.
x_max = 1.
n_samples = 100
samples = np.linspace(x_min, x_max, n_samples)
log_energy_values = one_d_logarithm_of_energy(samples)
log_penalty = one_d_log_penalty(samples)
l1_norm_values = one_d_lp_norm(samples, 1.)
l2_norm_values = one_d_lp_norm(samples, 2.)
l05_norm_values = one_d_lp_norm(samples, 0.5)
l0_norm_values = one_d_l0_norm(samples)
gaussian_values = gaussian_entropy(samples)

#plt.plot(samples, log_energy_values, color="yellow", label="log_energy")
plt.plot(samples, log_penalty, color="green", label="$log_{0.01}$")
plt.plot(samples, l1_norm_values, color="red", label="$l_{1}$")
plt.plot(samples, l2_norm_values, color="purple", label="$l_{2}$")
plt.plot(samples, l05_norm_values, color="blue", label="$l_{{0.5}}$")
#plt.plot(samples, gaussian_values, color='purple', label="gaussian")
plt.scatter(samples, l0_norm_values, color="black", label="$l_{0}$")  # Plotting a horizontal line at y=1
plt.legend()
plt.xlabel('coefficient value', fontsize=10)
plt.ylabel('penalty', fontsize=10)
plt.title('Comparison different penalty functions', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "visualization l1 l0.5 and log penalty functions.png"), dpi=400, format="png")
plt.show()
plt.close()

# Make the third figure
plt.plot(samples, log_energy_values, color="orange", label="$gaussian$ $entropy$")
plt.plot(samples, log_penalty, color="green", label="$log_{0.01}$")
plt.plot(samples, l1_norm_values, color="red", label="$l_{1}$")
plt.plot(samples, l2_norm_values, color="purple", label="$l_{2}$")
plt.plot(samples, l05_norm_values, color="blue", label="$l_{{0.5}}$")
#plt.plot(samples, gaussian_values, color='purple', label="gaussian")
plt.scatter(samples, l0_norm_values, color="black", label="$l_{0}$")  # Plotting a horizontal line at y=1
plt.legend()
plt.xlabel('coefficient value', fontsize=10)
plt.ylabel('penalty', fontsize=10)
plt.title('Comparison different penalty functions', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "visualization log energy.png"), dpi=400, format="png")
plt.show()
plt.close()

