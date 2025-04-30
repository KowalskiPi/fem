import numpy as np

def shape_functions(xi, eta):
    """
    Compute the bilinear shape functions and their derivatives
    with respect to the natural coordinates (xi, eta) for a 
    4-node quadrilateral element.

    Parameters:
      xi, eta: natural coordinates (each in [-1, 1])

    Returns:
      N       - Array of shape functions [N1, N2, N3, N4]
      dN_dxi  - Array containing the derivative of each shape function with respect to xi
      dN_deta - Array containing the derivative of each shape function with respect to eta
    """
    # Shape functions
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),  # N1: corresponds to bottom left (xi=-1,eta=-1)
        0.25 * (1 + xi) * (1 - eta),  # N2: bottom right (xi=1,eta=-1)
        0.25 * (1 + xi) * (1 + eta),  # N3: top right (xi=1,eta=1)
        0.25 * (1 - xi) * (1 + eta)   # N4: top left (xi=-1,eta=1)
    ])
    # Derivatives of shape functions with respect to xi
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    # Derivatives of shape functions with respect to eta
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
    using the isoparametric formulation. Also computes the Jacobian matrix
    and its determinant.

    Parameters:
      xi, eta: natural coordinates
      nodes:   A (4 x 2) array of nodal coordinates in physical space (ordered as [N1, N2, N3, N4])

    Returns:
      x    - The physical coordinates (x1, x2) at (xi, eta)
      J    - The 2x2 Jacobian matrix d(x)/d(xi,eta)
      detJ - The determinant of the Jacobian (used for integration)
    """
    N, dN_dxi, dN_deta = shape_functions(xi, eta)
    # Compute physical coordinates using the shape functions
    x = np.dot(N, nodes)  # x1 and x2 calculated by summing N_i * (x_i, y_i)
    
    # Compute the derivatives of physical coordinates with respect to xi and eta:
    dx_dxi  = np.dot(dN_dxi, nodes)
    dx_deta = np.dot(dN_deta, nodes)
    
    # Forming the Jacobian matrix: each column corresponds to derivatives with respect to xi and eta
    J = np.array([dx_dxi, dx_deta]).T
    # Compute the determinant of the Jacobian matrix
    detJ = np.linalg.det(J)
    
    return x, J, detJ

def compute_area(nodes):
    """
    Compute the area of the quadrilateral element using 2x2 Gauss quadrature.
    
    The area is calculated as:
      A = ∫_{-1}^{1}∫_{-1}^{1} |detJ(xi, eta)| dxi deta
    
    Parameters:
      nodes: A (4 x 2) array containing nodal coordinates in physical space.

    Returns:
      area: The computed area of the quadrilateral element.
    """
    # For 2x2 Gauss integration, the quadrature points in the natural coordinate system:
    a = 1 / np.sqrt(3)
    gauss_points = [-a, a]
    area = 0.0

    # Loop over quadrature points in both xi and eta directions
    for xi in gauss_points:
        for eta in gauss_points:
            # Map (xi, eta) to physical coordinates and get Jacobian determinant
            _, _, detJ = mapping(xi, eta, nodes)
            # Each Gauss weight is 1 for 2-point integration in each direction.
            area += abs(detJ)
    
    return area

def compute_integral(nodes):
    """
    Compute the integral of f(x1, x2) = x1^3 + x1*x2 - x2 over the quadrilateral element
    using 2x2 Gauss quadrature. In the physical domain, the integral is given by:
    
      I = ∫_A f(x1, x2) dA = ∫_{-1}^{1}∫_{-1}^{1} f(x1(xi, eta), x2(xi, eta)) |detJ(xi, eta)| dxi deta
    
    Parameters:
      nodes: A (4 x 2) array of the element’s nodal coordinates.

    Returns:
      I: The computed value of the integral.
    """
    def f(x):
        # Function to integrate: f(x1, x2) = x1^3 + x1*x2 - x2
        return x[0]**3 + x[0]*x[1] - x[1]

    a = 1 / np.sqrt(3)
    gauss_points = [-a, a]
    I = 0.0

    for xi in gauss_points:
        for eta in gauss_points:
            # Map to physical space and get the Jacobian determinant
            x, _, detJ = mapping(xi, eta, nodes)
            I += f(x) * abs(detJ)
    
    return I

if __name__ == '__main__':
    # Define nodal coordinates for the quadrilateral element in counterclockwise order:
    # Note: the ordering is crucial for the isoparametric mapping.
    nodes = np.array([
        [0.5, -0.5],  # Node 1: bottom left (corresponds to (xi,eta)=(-1,-1))
        [2.0,  0.0],  # Node 2: bottom right (corresponds to (xi,eta)=(1,-1))
        [1.0,  1.5],  # Node 3: top right (corresponds to (xi,eta)=(1,1))
        [-1.0, 1.0]   # Node 4: top left (corresponds to (xi,eta)=(-1,1))
    ])

    # Compute area and the integral I over the element
    area = compute_area(nodes)
    integral_I = compute_integral(nodes)
    
    print("Computed area of the element: ", area)
    print("Computed integral I: ", integral_I)
