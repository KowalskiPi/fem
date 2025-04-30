import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -------------------------------
# Geometry and Material Properties
# -------------------------------
L = 0.2
D = 0.1
E = 2e11
nu = 0.3
stress_state = 'PLANE_STRESS'
sigmato = -20e6

# Constitutive matrix
if stress_state == 'PLANE_STRESS':
    C = E/(1-nu**2) * np.array([[1, nu, 0],
                                [nu, 1, 0],
                                [0, 0, (1-nu)/2]])
else:
    C = E/((1+nu)*(1-2*nu)) * np.array([[1-nu, nu, 0],
                                        [nu, 1-nu, 0],
                                        [0, 0, (1-2*nu)/2]])

# -----------------
# Mesh Definition
# -----------------
node = np.array([[0, 0],    # Node 1
                 [L, 0],    # Node 2
                 [2*L, 0],  # Node 3
                 [2*L, D],  # Node 4
                 [L, D],    # Node 5
                 [0, D]])   # Node 6

element = np.array([[0, 1, 4, 5],  # Element 1 (MATLAB 1-based: [1 2 5 6])
                    [1, 2, 3, 4]]) # Element 2 (MATLAB 1-based: [2 3 4 5])

num_node = node.shape[0]
num_elem = element.shape[0]

# ----------------------
# Plot Original Mesh
# ----------------------
plt.figure(figsize=(8, 4))
plt.scatter(node[:, 0], node[:, 1], s=50)
left_nodes = [0, 5]  # MATLAB nodes 1,6
plt.scatter(node[left_nodes, 0], node[left_nodes, 1], c='k', marker='s')
for i in range(num_node):
    plt.text(node[i, 0]+L/20, node[i, 1]+L/20, str(i+1))
plt.title('Original Mesh')
plt.axis('equal')

# --------------------------
# FEM Analysis Setup
# --------------------------
t_dof = num_node * 2
K = np.zeros((t_dof, t_dof))
f = np.zeros(t_dof)

# Quadrature points (2x2 Gauss)
gp_coords = np.array([[1/np.sqrt(3), 1/np.sqrt(3)],
                      [1/np.sqrt(3), -1/np.sqrt(3)],
                      [-1/np.sqrt(3), 1/np.sqrt(3)],
                      [-1/np.sqrt(3), -1/np.sqrt(3)]])
gp_weights = np.ones(4)

# Shape functions for Q4 elements
def shape_func_Q4(xi, eta):
    N = 0.25 * np.array([(1-xi)*(1-eta),
                         (1+xi)*(1-eta),
                         (1+xi)*(1+eta),
                         (1-xi)*(1+eta)])
    dNdxi = 0.25 * np.array([[-(1-eta), -(1-xi)],
                             [ (1-eta), -(1+xi)],
                             [ (1+eta),  (1+xi)],
                             [-(1+eta),  (1-xi)]])
    return N, dNdxi

# ---------------------------
# Stiffness Matrix Assembly
# ---------------------------
for i_ele in range(num_elem):
    nodes = element[i_ele]
    elem_nodes = node[nodes]
    dofs = np.zeros(8, dtype=int)
    for i in range(4):
        dofs[2*i] = 2*nodes[i]
        dofs[2*i+1] = 2*nodes[i]+1
    
    ke = np.zeros((8, 8))
    for gp in range(4):
        xi, eta = gp_coords[gp]
        N, dNdxi = shape_func_Q4(xi, eta)
        
        # Jacobian matrix
        J = elem_nodes.T @ dNdxi
        inv_J = np.linalg.inv(J)
        dNdx = dNdxi @ inv_J
        
        # B matrix
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx[:, 0]
        B[1, 1::2] = dNdx[:, 1]
        B[2, 0::2] = dNdx[:, 1]
        B[2, 1::2] = dNdx[:, 0]
        
        # Element stiffness matrix
        ke += B.T @ C @ B * np.linalg.det(J) * gp_weights[gp]
    
    # Assemble to global matrix
    K[np.ix_(dofs, dofs)] += ke

# ---------------------
# Force Vector Assembly
# ---------------------
top_edges = [[4, 5]]  # MATLAB nodes 5-6 (0-based: 4-5)
q1D = np.array([1/np.sqrt(3), -1/np.sqrt(3)])
w1D = np.array([1.0, 1.0])

for edge in top_edges:
    n1, n2 = edge
    x1, y1 = node[n1]
    x2, y2 = node[n2]
    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    for gp in range(2):
        xi = q1D[gp]
        N = np.array([0.5*(1-xi), 0.5*(1+xi)])
        fy = N * sigmato * length/2 * w1D[gp]
        
        # Add to force vector
        f[2*n1+1] += fy[0]  # Y-dof of node n1
        f[2*n2+1] += fy[1]  # Y-dof of node n2

# -----------------------------
# Boundary Conditions (Left fixed)
# -----------------------------
clamped_dofs = [0, 1, 10, 11]  # MATLAB dofs 1,2,11,12
free_dofs = np.setdiff1d(np.arange(t_dof), clamped_dofs)

# Apply boundary conditions
K_red = K[np.ix_(free_dofs, free_dofs)]
f_red = f[free_dofs]

# -----------------
# Solve System
# -----------------
u_red = np.linalg.solve(K_red, f_red)

# Reconstruct full displacement vector
u = np.zeros(t_dof)
u[free_dofs] = u_red

# Extract displacements
ux = u[0::2]
uy = u[1::2]

# -----------------
# Post-processing
# -----------------
# Plot deformed mesh
plt.figure(figsize=(8, 4))
scale = 1000
deformed = node + scale * np.column_stack((ux, uy))

for i_ele in range(num_elem):
    nodes = element[i_ele]
    plt.fill(deformed[nodes, 0], deformed[nodes, 1], 
             edgecolor='k', alpha=0.8)
    
plt.scatter(deformed[:, 0], deformed[:, 1], s=30, c='r')
plt.title(f'Deformed Configuration (Scale: {scale}x)')
plt.axis('equal')
plt.show()