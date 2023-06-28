import sympy as sp

# Define the parameter symbols
k12, k23, k34, k21, k31, k41 = sp.symbols('k12 k23 k34 k21 k31 k41')

# Define the stationary distribution symbols as a list
Pfa = [sp.symbols(f'Pfa{i}') for i in range(4)]

# Define the stationary distribution vector
Pfa[0] = (k21*k31*k41 + k21*k34*k41 + k23*k31*k41 + k23*k34*k41)/(k12*k23*k34)
Pfa[1] = (k31*k41 + k34*k41)/(k23*k34)
Pfa[2] = k41/k34
Pfa[3] = 1

# Calculate the total sum of the stationary distribution
total_sum = sum(Pfa)

# Normalize the stationary distribution vector
Pfa_normalized = [p / total_sum for p in Pfa]

# Define the gradient functions
gradients = []
param_list = [k12, k23, k34, k21, k31, k41]
var_names = ['k12', 'k23', 'k34', 'k21', 'k31', 'k41']
for i in range(4):
    for j in range(6):
        gradient = sp.diff(Pfa_normalized[i], param_list[j])
        func_name = f"gradient_Pfa{i}_{var_names[j]}"
        gradients.append((func_name, gradient))

# Generate the Python code
file_content = ""
for idx, (func_name, gradient) in enumerate(gradients):
    func_code = sp.printing.pycode(gradient)
    func_def = f"def {func_name}({', '.join(var_names)}):\n    return {func_code}\n\n"
    file_content += func_def

# Write the Python file
with open("gradients.py", "w") as f:
    f.write(file_content)

