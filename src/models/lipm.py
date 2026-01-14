import numpy as np 

def discretize_double_integrator(dt: float): 
    I = np.eye(2)
    Z = np.zeros((2, 2))
    Ad = np.block([[I, dt * I], [Z, I]])
    Bd = np.block([[0.5 * dt**2 * I], [dt * I]])
    return Ad, Bd

def step(x: np.ndarray, u: np.ndarray, Ad: np.ndarray, Bd: np.ndarray) -> np.ndarray: 
    return Ad @ x + Bd @ u 

def zmp_from(x: np.ndarray, u: np.ndarray, zc: float, g: float):
    # x = [rx, ry, vx, vy]
    r = x[0:2]
    p = r - (zc / g) * u
    return p 

