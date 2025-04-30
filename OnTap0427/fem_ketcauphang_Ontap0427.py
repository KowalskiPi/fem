import numpy as np

def shape_functions(xi, eta):
    """
    Compute the bilinear shape functions and their derivatives
    with respect to the natural coordinates (xi, eta) for a 
    4-node quadrilateral element.

    4 ------- 3
    |         |
    |         |
    1 ------- 2

    Parameters:
      xi, eta: natural coordinates (each in [-1, 1])

    Returns:
      N       - Array of shape functions [N1, N2, N3, N4]
      dN_dxi  - Array containing the derivative of each shape function with respect to xi
      dN_deta - Array containing the derivative of each shape function with respect to eta
    """
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),  # N1: bottom left
        0.25 * (1 + xi) * (1 - eta),  # N2: bottom right
        0.25 * (1 + xi) * (1 + eta),  # N3: top right
        0.25 * (1 - xi) * (1 + eta)   # N4: top left
    ])
    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi)
    ])
    return N, dN_dxi, dN_deta

def mapping(xi, eta, nodes):
    """
    Map from natural coordinates (xi, eta) to physical coordinates (x, y)
    
    Parameters:
      xi, eta: natural coordinates.
      nodes:   A (4 x 2) array of the element’s nodal coordinates.
    
    Returns:
      x    - The physical coordinates (x, y) corresponding to (xi, eta)
      J    - The 2x2 Jacobian matrix d(x)/d(xi,eta)
      detJ - Determinant of J
    """
    N, dN_dxi, dN_deta = shape_functions(xi, eta)
    # Compute physical coordinates: x = sum_i N_i * (x_i, y_i)
    x = np.dot(N, nodes)
    
    # Compute derivatives of x with respect to xi and eta
    dx_dxi = np.dot(dN_dxi, nodes)
    dx_deta = np.dot(dN_deta, nodes)
    
    # Form the Jacobian; each column corresponds to derivatives with respect to xi and eta.
    J = np.array([dx_dxi, dx_deta]).T   # 2x2
    detJ = np.linalg.det(J)
    return x, J, detJ

def transform_to_natural_coordinates(sequence, nodes_input):
    """
    Transform the input nodal coordinates arranged in any sequence 
    to the natural order 1->2->3->4.
    
    The expected natural order for a quadrilateral is:
      1: bottom left, 2: bottom right, 3: top right, 4: top left.

    Parameters:
      sequence: A list containing the node numbers in the input order.
      nodes_input: The original nodal coordinates in the order provided.
    
    Returns:
      nodes: Reordered nodal coordinates in the natural order.
    """
    if sorted(sequence) != [1, 2, 3, 4]:
        raise ValueError("Sequence must contain exactly [1, 2, 3, 4] in any order.")
    else:
        # Reorder the nodes to conform with the natural order.
        nodes = np.array([
            nodes_input[sequence[0] - 1],  # Node 1: bottom left
            nodes_input[sequence[1] - 1],  # Node 2: bottom right
            nodes_input[sequence[2] - 1],  # Node 3: top right
            nodes_input[sequence[3] - 1]   # Node 4: top left
        ])
    return nodes

def compute_quad_element_stiffness(E, nu, t, nodes):
    """
    Compute the stiffness matrix for a 4-node quadrilateral element
    under plane stress using 2x2 Gauss integration (using helper functions).

    Parameters:
      E     : Young's modulus (Pa)
      nu    : Poisson's ratio
      t     : Element thickness (m)
      nodes : A (4 x 2) array of the element's nodal coordinates, arranged in
              the natural order (bottom-left, bottom-right, top-right, top-left).

    Returns:
      Ke    : Element stiffness matrix (8 x 8)
    """
    a = 1/np.sqrt(3)
    gauss_points = [-a, a]
    # Plane stress D-matrix:
    D = (E / (1 - nu**2)) * np.array([
                                        [1 , nu,        0],
                                        [nu, 1 ,        0],
                                        [0 , 0 , (1-nu)/2]
                                    ])
    
    Ke = np.zeros((8, 8))
    
    for xi in gauss_points:
        for eta in gauss_points:
            # Use the shape_functions helper function.
            N, dN_dxi, dN_deta = shape_functions(xi, eta)
            # Map from natural (xi,eta) to physical coordinates.
            _, J, detJ = mapping(xi, eta, nodes)
            if detJ <= 0:
                raise ValueError("Det(J) < 0. Check nodal ordering!")
            invJ = np.linalg.inv(J)
            # Compute derivatives with respect to global coordinates.
            dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            
            # Build the strain-displacement matrix B (3 x 8)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i]   = dN_dx[i]
                B[1, 2*i+1] = dN_dy[i]
                B[2, 2*i]   = dN_dy[i]
                B[2, 2*i+1] = dN_dx[i]
            # Each Gauss point (with weight=1) contributes to the stiffness.
            Ke += (B.T @ D @ B) * detJ * t
    return Ke

# ---------------------------
# Main: Assemble a structure with 3 elements and 8 nodes
# ---------------------------

# Global material and geometric properties.
E = 210e9      # Young's modulus in Pa.
nu = 0.3       # Poisson's ratio.
t  = 0.01      # Thickness in meters.

# Global nodal coordinates for 8 nodes.
a = 1 #length of edge

nodes_global = np.array([
    [0.0,   a],   # Node 1
    [a  ,   a],   # Node 2
    [a  , 2*a],   # Node 3
    [0.0, 2*a],   # Node 4
    [2*a,   a],   # Node 5
    [2*a, 2*a],   # Node 6
    [a  , 0.0],   # Node 7
    [2*a, 0.0]    # Node 8
])

# Define connectivity array (using 0-indexing). These the node indices for each element.
# Element 1: Nodes 1, 2, 3, 4  → [0, 1, 2, 3]
# Element 2: Nodes 2, 5, 6, 3  → [1, 4, 5, 2]
# Element 3: Nodes 7, 8, 5, 2  → [6, 7, 4, 1]
connectivity = [
    [0, 1, 2, 3],
    [1, 4, 5, 2],
    [6, 7, 4, 1]
]

# Global stiffness matrix (each node has 2 DOF: x and y).
num_nodes = nodes_global.shape[0]
total_dof = num_nodes * 2
global_K = np.zeros((total_dof, total_dof))

print("\n--- Element Stiffness Matrices ---")
for e_idx, elem in enumerate(connectivity):
    # Extract the element nodal coordinates:
    elem_nodes = nodes_global[elem, :]  # (4x2)
    
    natural_order = [1, 2, 3, 4]
    elem_nodes_ordered = transform_to_natural_coordinates(natural_order, elem_nodes)
    
    Ke_local = compute_quad_element_stiffness(E, nu, t, elem_nodes_ordered)
    
    print(f"\nElement {e_idx + 1} stiffness matrix:\n", Ke_local)
    
    # Assemble into the global stiffness matrix.
    dof_indices = []
    for node in elem:
        dof_indices.extend([2*node, 2*node+1])
    dof_indices = np.array(dof_indices)
    
    for i in range(len(dof_indices)):
        for j in range(len(dof_indices)):
            global_K[dof_indices[i], dof_indices[j]] += Ke_local[i, j]

np.set_printoptions(precision=2, suppress=True)
print("\n--- Global Stiffness Matrix (size {} x {}) ---".format(total_dof, total_dof))
print(global_K)

# Optionally, compute and report the element area and an example integral.
for e_idx, elem in enumerate(connectivity):
    elem_nodes = nodes_global[elem, :]
    # Again reordering can be applied if needed:
    elem_nodes_ordered = transform_to_natural_coordinates([1, 2, 3, 4], elem_nodes)

# def compute_area(nodes):
#     """ 
#     Compute the area of a quadrilateral element using 2x2 Gauss integration.
    
#     Parameters:
#       nodes: A (4 x 2) array of nodal coordinates.
    
#     Returns:
#       area: The computed area.
#     """
#     a = 1/np.sqrt(3)
#     gauss_points = [-a, a]
#     area = 0.0

#     for xi in gauss_points:
#       return x[0]**3 + x[0]*x[1] - x[1]
    
#     a = 1/np.sqrt(3)
#     gauss_points = [-a, a]
#     I = 0.0

#     for xi in gauss_points:
#         for eta in gauss_points:
#             x, _, detJ = mapping(xi, eta, nodes)
#             I += f(x) * abs(detJ)
#     return I        for eta in gauss_points:
#             _, _, detJ = mapping(xi, eta, nodes)
#             area += abs(detJ)
#     return area

# def compute_integral(nodes):
#     """
#     Compute the integral of f(x, y) = x^3 + x*y - y over
#     the quadrilateral element using 2x2 Gauss quadrature.
    
#     Parameters:
#       nodes: A (4 x 2) nodal coordinates array.
    
#     Returns:
#       I: Value of the integral.
#     """
#     def f(x):
#   