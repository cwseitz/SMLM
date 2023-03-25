import sympy as sp

# Define the variables
x, y, x0, y0, sigma, N0, B0 = sp.symbols('x y x0 y0 sigma N0 B0')

# Define the functions
Lambda_x = sp.erf((x + 1/2 - x0)/(sp.sqrt(2)*sigma)) - sp.erf((x - 1/2 - x0)/(sp.sqrt(2)*sigma))
Lambda_y = sp.erf((y + 1/2 - y0)/(sp.sqrt(2)*sigma)) - sp.erf((y - 1/2 - y0)/(sp.sqrt(2)*sigma))
L = 0.25*Lambda_x*Lambda_y

# Compute the common factors
j_x0 = L.diff(x0)
j_y0 = L.diff(y0)
j_sigma = L.diff(sigma)

# Generate code for the Jacobian function
code = """import numpy as np\n
from numpy import sqrt\n
from numpy import exp\n
from numpy import pi\n
from scipy.special import erf\n\n
"""

code += "def jacobian1(x, y, x0, y0, sigma, N0, B0):\n"
code += f"    j_x0 = {j_x0}\n"
code += f"    j_y0 = {j_y0}\n"
code += f"    j_sigma = {j_sigma}\n"
code += f"    j_N0 = np.zeros_like(j_x0)\n"
code += f"    j_B0 = np.zeros_like(j_x0)\n"
code += "    jac = np.array([j_x0, j_y0, j_sigma, j_N0, j_B0], dtype=np.float64)\n"
code += "    return jac\n"

# Save code to file
with open("jac1.py", "w") as f:
    f.write(code)

