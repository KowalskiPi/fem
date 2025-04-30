import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def grid_rectangle(fLength, fWidth, nNumNodesX, nNumNodeY):
    """
    Create a grid for a rectangular domain.

    Parameters:
        fLength       - Length of the domain (x-axis)
        fWidth        - Width of the domain (y-axis)
        nNumNodesX    - Number of nodes in the x direction
        nNumNodeY     - Number of nodes in the y direction

    Returns:
        nodes    - 2D array (n x 2) - coordinates of n nodes [(x, y)]
        elements - 2D array (m x 4) - indices of nodes forming each element
                    (the node order is: bottom left, bottom right, top right, top left)
                    4-------3
                    |       |
                    |  QX   |       
                    1-------2
    """
    # Create arrays for node coordinates along x and y axes
    x = np.linspace(0, fLength, nNumNodesX)
    y = np.linspace(0, fWidth, nNumNodeY)
    X, Y = np.meshgrid(x, y)
    # Stack the grid coordinates into a list of nodes
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    
    elements = []
    # The total number of square elements is (nx - 1) * (ny - 1)
    for j in range(nNumNodeY - 1):
        for i in range(nNumNodesX - 1):
            # Node numbering convention:
            # n1: bottom left, n2: bottom right, n3: top right, n4: top left
            n1 = j * nNumNodesX + i + 1  # Start 1
            n2 = n1 + 1
            n3 = n1 + nNumNodesX + 1
            n4 = n1 + nNumNodesX
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements)
    
    return nodes, elements

def plot_grid(nodes, elements):
    """
    Plot the grid with node and element numbering.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot each element (cell) of the mesh
    for idx, elem in enumerate(elements):
        pts = nodes[elem - 1]  #1-based indexing

        # Close the polygon by appending the first node at the end
        pts = np.vstack([pts, pts[0]])
        plt.plot(pts[:, 0], pts[:, 1], 'b-o', linewidth=1, markersize=5)

        # Element number in the center of the polygon
        centroid = pts[:-1].mean(axis=0)
        plt.text(centroid[0], centroid[1], f'Q{idx + 1}', color='magenta', fontsize=10)
    
    # Label each node
    for idx, (xi, yi) in enumerate(nodes, start=1):  # Start 1
        plt.text(xi, yi, f'{idx}', color='red', fontsize=10)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rectangular Domain Grid')
    plt.axis('equal')
    plt.show()
    
# -------------------------- MAIN FUNCTION --------------------------:
# Using in BTL20% - 1
if __name__ == '__main__':
    userLength     = 8    # Domain length (x-axis)
    userWidth      = 4     # Domain width (y-axis)
    nUserNumNodeX  = 9     # Number of nodes in the x direction
    nUserNumNodeY  = 5     # Number of nodes in the y direction
    nodes, elements = grid_rectangle(userLength, userWidth, nUserNumNodeX, nUserNumNodeY)
    
    # Debug print
    print("BTL20% - 1")
    print("\tNodes array:")
    for idx, row in enumerate(nodes, start=1):  # Start node numbering from 1
        print(f"\t{idx}: {row}")
    print("\tElements array:")
    for idx, row in enumerate(elements, start=1):  # Start element numbering from 1
        print(f"\t{idx}: {row}")
    
    plot_grid(nodes, elements)
