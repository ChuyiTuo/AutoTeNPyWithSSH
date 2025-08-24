import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d

def quspin_ising_ground_energy(L: int, J: float, h: float) -> float:
    """Compute the ground state energy of the 1D transverse field Ising model using QuSpin.
    
    Args:
        L (int): System size
        J (float): Coupling strength
        h (float): Transverse field strength
    
    Returns:
        float: Ground state energy
    """
    # 使用 pauli=True 来正确解释 "zz" 和 "x"
    basis = spin_basis_1d(L=L) 

    bc = "OBC"
    if bc == 'PBC':
        # 周期性边界条件 (Periodic Boundary Conditions)
        J_zz = [[J, i, (i + 1) % L] for i in range(L)]
    elif bc == 'OBC':
        # 开放边界条件 (Open Boundary Conditions)
        J_zz = [[J, i, i + 1] for i in range(L - 1)]
    else:
        raise ValueError("边界条件(bc)必须是 'PBC' 或 'OBC'")
        
    h_x = [[-h, i] for i in range(L)]
    
    static = [["zz", J_zz], ["x", h_x]]
    
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)
    
    # 精确对角化求解基态能量
    E = np.linalg.eigvalsh(H.toarray())[0]
    return E
