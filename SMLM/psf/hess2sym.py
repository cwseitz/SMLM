import sympy as sp

# Define the variables
x, y, x0, y0, sigma, N0, B0, eta, texp = sp.symbols('x y x0 y0 sigma N0 B0 eta texp')

# Define the functions
Lambda_x = sp.erf((x + 1/2 - x0)/(sp.sqrt(2)*sigma)) - sp.erf((x - 1/2 - x0)/(sp.sqrt(2)*sigma))
Lambda_y = sp.erf((y + 1/2 - y0)/(sp.sqrt(2)*sigma)) - sp.erf((y - 1/2 - y0)/(sp.sqrt(2)*sigma))
L = 0.25*Lambda_x*Lambda_y
Mu = N0*eta*texp*L + B0*eta*texp

# Compute elements of the Hessian
h_xx = Mu.diff(x0).diff(x0)
h_xy = Mu.diff(x0).diff(y0)
h_xs = Mu.diff(x0).diff(sigma)
h_xN = Mu.diff(x0).diff(N0)
h_xB = Mu.diff(x0).diff(B0)

h_yx = Mu.diff(y0).diff(x0)
h_yy = Mu.diff(y0).diff(y0)
h_ys = Mu.diff(y0).diff(sigma)
h_yN = Mu.diff(y0).diff(N0)
h_yB = Mu.diff(y0).diff(B0)

h_sx = Mu.diff(sigma).diff(x0)
h_sy = Mu.diff(sigma).diff(y0)
h_ss = Mu.diff(sigma).diff(sigma)
h_sN = Mu.diff(sigma).diff(N0)
h_sB = Mu.diff(sigma).diff(B0)


h_Nx = Mu.diff(N0).diff(x0)
h_Ny = Mu.diff(N0).diff(y0)
h_Ns = Mu.diff(N0).diff(sigma)
h_NN = Mu.diff(N0).diff(N0)
h_NB = Mu.diff(N0).diff(B0)


h_Bx = Mu.diff(B0).diff(x0)
h_By = Mu.diff(B0).diff(y0)
h_Bs = Mu.diff(B0).diff(sigma)
h_BN = Mu.diff(B0).diff(N0)
h_BB = Mu.diff(B0).diff(B0)


# Generate code for the Hessian function
code = """import numpy as np\n
from numpy import sqrt\n
from numpy import exp\n
from numpy import pi\n
from scipy.special import erf\n\n
"""

code += "def hessian2(x, y, x0, y0, sigma, N0, B0, eta, texp):\n"
code += f"    h_xx = {h_xx}\n"
code += f"    h_xy = {h_xy}\n"
code += f"    h_xs = {h_xs}\n"
code += f"    h_xN = {h_xN}\n"
code += f"    h_xB = np.zeros_like(h_xx)\n"
code += f"    h_yx = {h_yx}\n"
code += f"    h_yy = {h_yy}\n"
code += f"    h_ys = {h_ys}\n"
code += f"    h_yN = {h_yN}\n"
code += f"    h_yB = np.zeros_like(h_xx)\n"
code += f"    h_sx = {h_sx}\n"
code += f"    h_sy = {h_sy}\n"
code += f"    h_ss = {h_ss}\n"
code += f"    h_sN = {h_sN}\n"
code += f"    h_sB = np.zeros_like(h_xx)\n"
code += f"    h_Nx = {h_Nx}\n"
code += f"    h_Ny = {h_Ny}\n"
code += f"    h_Ns = {h_Ns}\n"
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

