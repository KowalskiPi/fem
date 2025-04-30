import numpy as np
import matplotlib.pyplot as plt

def mesh_annularSector(fInnerR, fOuterR, fAngle, nNumNodesRadial, nNumNodesCir):
    """
    Generate a mesh for an annular sector.

    Parameters:
        fInner          : Inner radius.
        fOuter          : Outer radius.
        fAngle          : Angle of the sector in degrees.
        nNumNodesRadial : Number of nodes in the radial direction.
        nNumNodesCir    : Number of nodes in the circumferential direction.

    Returns:
        nodes    - 2D array (n x 2) - coordinates of n nodes [(x, y)]
        elements - 2D array (m x 4) - indices of nodes forming each element
                    (the node order is: bottom left, bottom right, top right, top left)
                        4-------3
                        |       |
                        |  QX   |       
                        1-------2
    """
    # Convert alpha to radians
    fAngleRadian = np.radians(fAngle)

    # Generate radial and angular coordinates
    r = np.linspace(fInnerR, fOuterR, nNumNodesRadial)
    theta = np.linspace(0, fAngleRadian, nNumNodesCir)
    R, Theta = np.meshgrid(r, theta)

    # Convert polar coordinates to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Stack the grid coordinates into a list of nodes
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # Generate elements
    elements = []
    for j in range(nNumNodesCir - 1):
        for i in range(nNumNodesRadial - 1):
            # Node numbering convention:
            # n1: bottom left, n2: bottom right, n3: top right, n4: top left
            n1 = j * nNumNodesRadial + i + 1  # Start 1
            n2 = n1 + 1
            n3 = n1 + nNumNodesRadial + 1
            n4 = n1 + nNumNodesRadial
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements)

    return nodes, elements

def plot_annular_sector_mesh(nodes, elements):
    """
    Plot the mesh for the annular sector.

    Parameters:
        nodes (ndarray): Node matrix (n x 2), containing the coordinates of the nodes.
        elements (ndarray): Element matrix (m x 4), containing the indices of nodes forming each element.
    """
    plt.figure(figsize=(8, 6))

    # Plot each element (cell) of the mesh
    for idx, elem in enumerate(elements):
        pts = nodes[elem - 1]  # Adjust for 1-based indexing

        # Close the polygon by appending the first node at the end
        pts = np.vstack([pts, pts[0]])
        plt.plot(pts[:, 0], pts[:, 1], 'b-o', linewidth=1, markersize=5)

        # Element number in the center of the polygon
        centroid = pts[:-1].mean(axis=0)
        plt.text(centroid[0], centroid[1], f'Q{idx + 1}', color='magenta', fontsize=8)

    # Label each node
    for idx, (xi, yi) in enumerate(nodes, start=1):  # Start 1
        plt.text(xi, yi, f'{idx}', color='red', fontsize=8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Annular Sector Mesh')
    plt.axis('equal')
    plt.show()

# -------------------------- MAIN FUNCTION --------------------------:
if __name__ == '__main__':
    # Input parameters
    fInnerR = 6     # Inner radius
    fOuterR = 10    # Outer radius
    fAngle  = 120.0 # Angle of the sector in degrees
    nNNR    = 5     # Number of nodes in the radial direction
    nNNCir  = 9     # Number of nodes in the circumferential direction

    # Generate the mesh
    nodes, elements = mesh_annularSector(fInnerR, fOuterR, fAngle, nNNR, nNNCir)

    # Debug print
    print("BTL20% - 2")
    print("Nodes array:")
    for idx, row in enumerate(nodes, start=1):
        print(f"\t{idx}: {row}")
    print("Elements array:")
    for idx, row in enumerate(elements, start=1):
        print(f"\t{idx}: {row}")

    # Plot the mesh
    plot_annular_sector_mesh(nodes, elements)