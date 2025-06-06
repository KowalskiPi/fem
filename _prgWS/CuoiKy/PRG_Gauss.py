"""
    By              :   Khanh Nguyen
    Email           :   nmkhanhfem@gmail.com
    Github          :   https://github.com/KowalskiPi
    Date            :   28/04/2025
    Description     :   2D Gause quadrature. 
"""

import numpy as np

def shape_functions(xi, eta):
    """
    Compute the bilinear shape functions and their derivatives
    with respect to the natural coordinates (xi, eta) for a 
    4-node quadrilateral element.

    4-------3
    |       |
    |  QX   |       
    1-------2

    Parameters:
      xi, eta: natural coordinates (each in [-1, 1])

    Returns:
      N       - Array of shape functions [N1, N2, N3, N4]
      dN_dxi  - Array containing the derivative of each shape function with respect to xi
      dN_deta - Array containing the derivative of each shape function with respect to eta
    """
    # Shape functions
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),  # N1: bottom left (xi=-1,eta=-1)
        0.25 * (1 + xi) * (1 - eta),  # N2: bottom right (xi=1,eta=-1)
        0.25 * (1 + xi) * (1 + eta),  # N3: top right (xi=1,eta=1)
        0.25 * (1 - xi) * (1 + eta)   # N4: top left (xi=-1,eta=1)
    ])
    # Derivatives of shape functions w.r.t xi
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    # Derivatives of shape functions w.r.t eta
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi)
    ])
    return N, dN_dxi, dN_deta

def mapping(xi, eta, nodes):
    """
    Map from natural coordinates (xi, eta) to physical coordinates (x1, x2)

    Parameters:
      xi, eta: natural coordinates
      nodes:   A (4 x 2) physical coordinates ([N1, N2, N3, N4])

    Returns:
      x    - The physical coordinates (x1, x2) @ (xi, eta)
      J    - The 2x2 Jacobian matrix d(x)/d(xi,eta)
      detJ - Det(J)
    """
    N, dN_dxi, dN_deta = shape_functions(xi, eta)
    # Physical coordinates
    x = np.dot(N, nodes)  # x1 and x2 = sum(N_i * (x_i, y_i))
    
    # Compute the derivatives of physical coordinates with respect to xi and eta:
    dx_dxi  = np.dot(dN_dxi, nodes)
    dx_deta = np.dot(dN_deta, nodes)
    
    # Jacobian matrix:
    J = np.array([dx_dxi, dx_deta]).T
    # Determinant of the Jacobian matrix
    detJ = np.linalg.det(J)
    
    return x, J, detJ

def compute_area(nodes):
    """ 
    Area is calculated as:
      A = int_{-1}^{1}int_{-1}^{1} |detJ(xi, eta)| dxi deta
    
    Parameters:
      nodes: A (4 x 2) physical coordinates.

    Returns:
      area: The computed area.
    """
    # For 2x2 Gauss integration, int. points in the natural coordinate system:
    a = 1 / np.sqrt(3)
    gauss_points = [-a, a]
    area = 0.0

    for xi in gauss_points:
        for eta in gauss_points:
            # Map (xi, eta) to physical
            _, _, detJ = mapping(xi, eta, nodes)
            # Each Gauss weight is 1 for 2-point integration in each direction.
            area += abs(detJ)
    
    return area

def compute_integral(nodes):
    """
    Compute the integral of f(xi, eta) = ξ^3+η^2-ξη over the quadrilateral element
    using 2x2 Gauss quadrature.
    
    Parameters:
      nodes: A (4 x 2) array of the element’s nodal coordinates.

    Returns:
      I: The computed value of the integral.
    """
    def f(xi):
        # Integrate: f(xi, eta) = ξ^3 + η^2 - ξη
        return xi[0]**3 + xi[1]**2 - xi[0]*xi[1]

    #2×2 Gauss quadrature (with Gauss points ±1/sqrt(3) and weights 1
    a = 1 / np.sqrt(3)
    gauss_points = [-a, a]
    I = 0.0

    for xi in gauss_points:
        for eta in gauss_points:
            # Get the Jacobian determinant
            _, _, detJ = mapping(xi, eta, nodes)
            I += f([xi, eta]) * abs(detJ)

    return I

def transform_to_natural_coordinates(sequence, nodes_input):
    """
    Transform the input coordinates of nodes in the order  1->2->3->4
    """
    if sorted(sequence) != [1, 2, 3, 4]:
        raise ValueError("Sequence must contain exactly [1, 2, 3, 4] in any order.")
    else:
        # Reorder the nodes to match the natural coordinate order
        nodes = np.array([
            nodes_input[sequence[0] - 1],  # Node 1: bottom left
            nodes_input[sequence[1] - 1],  # Node 2: bottom right
            nodes_input[sequence[2] - 1],  # Node 3: top right
            nodes_input[sequence[3] - 1]   # Node 4: top left
        ])

    return nodes

if __name__ == '__main__':
    # Define nodal coordinates for the quadrilateral element in counterclockwise order:

    """
       4-------3              4-------3
      /       /               |       |
     /       /     --->       |  QX   |       
    1-------2                 1-------2
    """ 
    
    nodes_input = np.array([
        [0.0,  0.0],  # Node 1: top right (3 - (xi,eta)=(1,1))
        [2.0, 0.0],  # Node 2: top left (4 - (xi,eta)=(-1,1))
        [1.0, 1.0],  # Node 3: bottom left (1 - (xi,eta)=(-1,-1))
        [0.0, 1.0]   # Node 4: bottom right (2- xi,eta)=(1,-1))
    ])

    nodes = transform_to_natural_coordinates([1, 2, 3, 4], nodes_input)

    # Compute area and the integral I over the element
    area = compute_area(nodes)
    integral_I = compute_integral(nodes)
    
    print(f"Area, A = {area:.4f}")
    print(f"Integral, I = {integral_I:.4f}")
