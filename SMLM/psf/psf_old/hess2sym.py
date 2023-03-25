import sympy as sp

# Define the variables
x, y, x0, y0, sigma, N0, B0 = sp.symbols('x y x0 y0 sigma N0 B0')

# Define the functions
Lambda_x = sp.erf((x + 1/2 - x0)/(sp.sqrt(2)*sigma)) - sp.erf((x - 1/2 - x0)/(sp.sqrt(2)*sigma))
Lambda_y = sp.erf((y + 1/2 - y0)/(sp.sqrt(2)*sigma)) - sp.erf((y - 1/2 - y0)/(sp.sqrt(2)*sigma))
L = 0.25*Lambda_x*Lambda_y

# Compute elements of the Hessian
h_xx = L.diff(x0).diff(x0)
h_xy = L.diff(x0).diff(y0)
h_xs = L.diff(x0).diff(sigma)
#h_xN = L.diff(x0).diff(N0)
#h_xB = L.diff(x0).diff(B0)

h_yx = L.diff(y0).diff(x0)
h_yy = L.diff(y0).diff(y0)
h_ys = L.diff(y0).diff(sigma)
#h_yN = L.diff(y0).diff(N0)
#h_yB = L.diff(y0).diff(B0)

h_sx = L.diff(sigma).diff(x0)
h_sy = L.diff(sigma).diff(y0)
h_ss = L.diff(sigma).diff(sigma)
#h_sN = L.diff(sigma).diff(N0)
#h_sB = L.diff(sigma).diff(B0)


h_Nx = L.diff(N0).diff(x0)
h_Ny = L.diff(N0).diff(y0)
h_Ns = L.diff(N0).diff(sigma)
#h_NN = L.diff(N0).diff(N0)
#h_NB = L.diff(N0).diff(B0)


h_Bx = L.diff(B0).diff(x0)
h_By = L.diff(B0).diff(y0)
h_Bs = L.diff(B0).diff(sigma)
#h_BN = L.diff(B0).diff(N0)
#h_BB = L.diff(B0).diff(B0)


# Generate code for the Hessian function
code = """import numpy as np\n
from numpy import sqrt\n
from numpy import exp\n
from numpy import pi\n
from scipy.special import erf\n\n
"""

code += "def hessian2(x, y, x0, y0, sigma, N0, B0):\n"
code += f"    h_xx = {h_xx}\n"
code += f"    h_xy = {h_xy}\n"
code += f"    h_xs = {h_xs}\n"
code += f"    h_xN = np.zeros_like(h_xx)\n"
code += f"    h_xB = np.zeros_like(h_xx)\n"
code += f"    h_yx = {h_yx}\n"
code += f"    h_yy = {h_yy}\n"
code += f"    h_ys = {h_ys}\n"
code += f"    h_yN = np.zeros_like(h_xx)\n"
code += f"    h_yB = np.zeros_like(h_xx)\n"
code += f"    h_sx = {h_sx}\n"
code += f"    h_sy = {h_sy}\n"
code += f"    h_ss = {h_ss}\n"
code += f"    h_sN = np.zeros_like(h_xx)\n"
code += f"    h_sB = np.zeros_like(h_xx)\n"
code += f"    h_Nx = np.zeros_like(h_xx)\n"
code += f"    h_Ny = np.zeros_like(h_xx)\n"
code += f"    h_Ns = np.zeros_like(h_xx)\n"
code += f"    h_NN = np.zeros_like(h_xx)\n"
code += f"    h_NB = np.zeros_like(h_xx)\n"
code += f"    h_Bx = np.zeros_like(h_xx)\n"
code += f"    h_By = np.zeros_like(h_xx)\n"
code += f"    h_Bs = np.zeros_like(h_xx)\n"
code += f"    h_BN = np.zeros_like(h_xx)\n"
code += f"    h_BB = np.zeros_like(h_xx)\n"
code += "    H = np.array([[h_xx,h_xy,h_xs,h_xN,h_xB], [h_yx,h_yy,h_ys,h_yN,h_yB], [h_sx,h_sy,h_ss,h_sN,h_sB],[h_Nx,h_Ny,h_Ns,h_NN,h_NB],[h_Bx,h_By,h_Bs,h_BN,h_BB]], dtype=np.float64)\n"
code += "    return H\n"

# Save code to file
with open("hess2.py", "w") as f:
    f.write(code)

