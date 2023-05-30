import sympy as sp

# Define the variables
x, y, x0, y0, z0, sigma, N0, eta, texp, gain, var = sp.symbols('x y x0 y0 z0 sigma N0 eta texp gain var')

pixel_size = 108.3
zmin = 413.741/pixel_size
a = 5.349139e-7*pixel_size**2
b = 6.016703e-7*pixel_size**2
sigma_x = sigma + a*(z0+zmin)**2
sigma_y = sigma + b*(z0-zmin)**2
 
Lambda_x = sp.erf((x + 1/2 - x0)/(sp.sqrt(2)*sigma_x)) - sp.erf((x - 1/2 - x0)/(sp.sqrt(2)*sigma_x))
Lambda_y = sp.erf((y + 1/2 - y0)/(sp.sqrt(2)*sigma_y)) - sp.erf((y - 1/2 - y0)/(sp.sqrt(2)*sigma_y))
L = 0.25*Lambda_x*Lambda_y
Mu = gain*eta*texp*N0*L + var

# Compute the common factors
j_x0 = Mu.diff(x0)
j_y0 = Mu.diff(y0)
j_z0 = Mu.diff(z0)
j_s0 = Mu.diff(sigma)
j_N0 = Mu.diff(N0)

# Generate code for the Jacobian function
code = """import numpy as np\n
from numpy import sqrt\n
from numpy import exp\n
from numpy import pi\n
from scipy.special import erf\n\n
"""

code += "def jacobian1(x, y, x0, y0, z0, sigma, N0, eta, texp, gain, var):\n"
code += f"    j_x0 = {j_x0}\n"
code += f"    j_y0 = {j_y0}\n"
code += f"    j_z0 = {j_z0}\n"
code += f"    j_s0 = {j_s0}\n"
code += f"    j_N0 = {j_N0}\n"
code += "    jac = np.array([j_x0, j_y0, j_z0, j_s0, j_N0], dtype=np.float64)\n"
code += "    return jac\n"

# Save code to file
with open("jac1.py", "w") as f:
    f.write(code)

