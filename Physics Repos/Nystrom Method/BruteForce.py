import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
from sympy import Matrix

scalar_parameter = 1
a = -1
b = 1

#K(t,s)
def kernel_term(t,s):
    return np.cosh(t + s)

#y(t)
def driving_term(t):
    return -np.cosh(t)

#f(t)
def solution(t):
    return np.cosh(t) / (0.5 * np.sinh(2))

#generate the vector "g", the driving term evaluated at each quadrature point
def generate_driving_vector(driver, quadrature_points):
    driving_vector = []
    for point in quadrature_points:
        driving_vector.append(driver(point))
    return np.array(driving_vector)

#generate the matrix K_ij, the kernel evaluated at each abscissa and quadrature point
#then scale it by the scalar parameter, and subtract it from the identity matrix
def generate_k_matrix(kernel, abscissas, quadrature_points):
    k_matrix = []
    for point in quadrature_points:
        new_row = []
        for abscissa in abscissas:
            new_row.append(kernel(point, abscissa))
        k_matrix.append(new_row)
    k_matrix = scalar_parameter * np.array(k_matrix)
    identity_matrix = np.identity(len(quadrature_points))
    k_matrix = identity_matrix - k_matrix
    return k_matrix

def Nystrom(kernel, driver, start, end, a, b, n):
    abscissas = [a]
    quadrature_points = [start]
    for i in range(n - 1):
        abscissas.append(abscissas[i] + (b-a)/n)
    for i in range(n - 1):
        quadrature_points.append(quadrature_points[i] + (end - start)/n)
    driving_vector = generate_driving_vector(driver, quadrature_points)
    k_matrix = generate_k_matrix(kernel, abscissas, quadrature_points)
    augmented_matrix = Matrix(k_matrix).col_insert(6 , Matrix(driving_vector))
    reduced_matrix = augmented_matrix.rref()[0]
    return_points = []
    for i in range(n):
        return_points.append(reduced_matrix.row(i)[5])
    return return_points, quadrature_points

test = Nystrom(kernel_term, driving_term, 0, 10, -1, 1, 5)
print(test)
plt.plot(test[1], test[0], 'o')
plt.plot(test[1], solution(test[1]))
plt.show()