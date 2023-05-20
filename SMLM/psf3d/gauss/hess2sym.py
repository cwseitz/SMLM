import sympy as sp

# Define the variables
x, y, x0, y0, z0, sigma, N0, eta, texp, gain, var = sp.symbols('x y x0 y0 z0 sigma N0 eta texp gain var')

sigma_x = sigma + 5.349139e-7*(z0+413.741)**2
sigma_y = sigma + 6.016703e-7*(z0-413.741)**2
Lambda_x = sp.erf((x + 1/2 - x0)/(sp.sqrt(2)*sigma_x)) - sp.erf((x - 1/2 - x0)/(sp.sqrt(2)*sigma_x))
Lambda_y = sp.erf((y + 1/2 - y0)/(sp.sqrt(2)*sigma_y)) - sp.erf((y - 1/2 - y0)/(sp.sqrt(2)*sigma_y))
L = 0.25*Lambda_x*Lambda_y
Mu = gain*eta*texp*N0*L + var

# Compute elements of the Hessian
h_xx = Mu.diff(x0).diff(x0)
h_xy = Mu.diff(x0).diff(y0)
h_xz = Mu.diff(x0).diff(z0)
h_xN = Mu.diff(x0).diff(N0)

h_yx = Mu.diff(y0).diff(x0)
h_yy = Mu.diff(y0).diff(y0)
h_yz = Mu.diff(y0).diff(z0)
h_yN = Mu.diff(y0).diff(N0)

h_zx = Mu.diff(z0).diff(x0)
h_zy = Mu.diff(z0).diff(y0)
h_zz = Mu.diff(z0).diff(z0)
h_zN = Mu.diff(z0).diff(N0)

h_Nx = Mu.diff(N0).diff(x0)
h_Ny = Mu.diff(N0).diff(y0)
h_Nz = Mu.diff(N0).diff(z0)
h_NN = Mu.diff(N0).diff(N0)

# Generate code for the Hessian function
code = """import numpy as np\n
from numpy import sqrt\n
from numpy import exp\n
from numpy import pi\n
from scipy.special import erf\n\n
"""

code += "def hessian2(x, y, x0, y0, z0, sigma, N0, eta, texp, gain, var):\n"
code += f"    h_xx = {h_xx}\n"
code += f"    h_xy = {h_xy}\n"
code += f"    h_xz = {h_xz}\n"
code += f"    h_xN = {h_xN}\n"
code += f"    h_yx = {h_yx}\n"
code += f"    h_yy = {h_yy}\n"
code += f"    h_yz = {h_yz}\n"
code += f"    h_yN = {h_yN}\n"
code += f"    h_zx = {h_zx}\n"
code += f"    h_zy = {h_zy}\n"
code += f"    h_zz = {h_zz}\n"
code += f"    h_zN = {h_zN}\n"
code += f"    h_Nx = {h_Nx}\n"
code += f"    h_Ny = {h_Ny}\n"
code += f"    h_Nz = {h_Nz}\n"
code += f"    h_NN = np.zeros_like(h_xx)\n"
code += "    H = np.array([[h_xx,h_xy,h_xz,h_xN], [h_yx,h_yy,h_yz,h_yN], [h_zx,h_zy,h_zz,h_zN],[h_Nx,h_Ny,h_Nz,h_NN]], dtype=np.float64)\n"
code += "    return H\n"

# Save code to file
with open("hess2.py", "w") as f:
    f.write(code)
