import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

# a: Calculate natural freq. (no damping)
def cau_a(M, K):
    
    # Solve generalized eigenvalue: K·φ = λ·M·φ = omeaga**2·M·φ
    eigenvalues, _ = eigh(K, M)
    
    # Calculate natural freq. (rad/s)
    omegaRad = np.sqrt(eigenvalues)
    
    # Convert to Hz
    freq_hz = omegaRad / (2 * np.pi)
    
    return freq_hz

# b: Time inte. w/ Newmark-beta method
def cau_b(M, K):

    # M = np.diag([1.5, 3.0, 1.0])
    # K = np.array([
    #     [12.0, 5.0, 3.0],
    #     [5.0, 13.0, 1.0],
    #     [3.0, 1.0, 11.0]
    # ])
    
    # Damping matrix
    C = 0.1 * M + 0.05 * K
    
    # F(t)
    def F(t):
        return np.array([-5.0, 3.0, 1.0]) * (t**2 + 0.12*t)
    
    # Time parameters
    dt = 0.1
    t_end = 0.5
    t_steps = np.arange(0, t_end + dt, dt)
    n_steps = len(t_steps)
    
    # Initialize
    u = np.zeros((3, n_steps))      # Displacement
    u_dot = np.zeros((3, n_steps))  # Velocity
    u_ddot = np.zeros((3, n_steps)) # Acceleration
    
    # Assumed zero initial conditions
    u0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    
    u[:, 0] = u0
    u_dot[:, 0] = v0
    u_ddot[:, 0] = np.linalg.solve(M, F(0) - C@v0 - K@u0)
    
    # Newmark params (const average u_ddot)
    gamma = 0.5
    beta = 0.25
    
    # Precompute K_bar and LU factori.
    b1 = (1/(beta*dt**2))
    b2 = (1/(beta*dt))
    K_bar = b1*M + b2*C + K
    lu_piv = lu_factor(K_bar)
    
    # Time integration loop
    for i in range(n_steps - 1):
        t_next = t_steps[i+1]
        
        # Predictor step
        u_predictor = u[:, i] + dt*u_dot[:, i] + (0.5 - beta)*dt**2*u_ddot[:, i]
        v_predictor = u_dot[:, i] + (1 - gamma)*dt*u_ddot[:, i]

        # Effective force vector
        F_next = F(t_next)
        F_eff = F_next + b1*M@u_predictor + b2*C@v_predictor

        # Solve next displacement
        u_next = lu_solve(lu_piv, F_eff)
        
        # Update u_ddot and u_dot
        u_ddot_next = b1 * (u_next - u_predictor)
        u_dot_next = v_predictor + gamma*dt*u_ddot_next

        # Store results
        u[:, i+1] = u_next
        u_dot[:, i+1] = u_dot_next
        u_ddot[:, i+1] = u_ddot_next
    
    return u, u_dot, u_ddot, t_steps

# MAIN
if __name__ == "__main__":

    M = np.diag([1.5, 3.0, 1.0])
    K = np.array([
        [12.0, 5.0, 3.0],
        [5.0, 13.0, 1.0],
        [3.0, 1.0, 11.0]
    ])

    # Cau a: Natural frequencies
    frequencies = cau_a(M=M, K=K)
    print("Natural Frequencies (Hz):")
    for i, f in enumerate(frequencies):
        print(f"Mode {i+1}: {f:.4f} Hz")
    
    # Cau b: Time history results
    u, u_dot, u_ddot, t_steps = cau_b(M=M, K=K)

    print("\nTime History Results:")
    for i, t in enumerate(t_steps):
        print(f"\nt = {t:.1f} s:")
        print(f"u: {u[:, i]}")
        print(f"u_dot: {u_dot[:, i]}")
        print(f"u_ddot: {u_ddot[:, i]}")