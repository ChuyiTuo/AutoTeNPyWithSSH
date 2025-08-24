from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinModel, SpinChain

def tenpy_ising_ground_energy(L: int, J: float, h: float) -> float:
    """Use TeNPy to compute the ground state energy of the 1D transverse field Ising model.
    
    Args:
        L (int): System size
        J (float): Coupling strength
        h (float): Transverse field strength
    
    Returns:
        float: Ground state energy
    """
    
    # TeNPy 的哈密顿量定义 H = -J sum(sigma_z sigma_z) - g sum(sigma_x)
    # 所以我们需要将输入的 J 取反
    
    model_params = {
        'L': L,
        'J': -J, 
        'g': h,
        'conserve': None,
        'bc_MPS': 'finite', # MPS 总是有限的
    }

    M = TFIChain(model_params)
    
    psi = MPS.from_product_state(M.lat.mps_sites(), [("up")] * L)
    
    dmrg_params = {
        'mixer': False, # 对 PBC 和小系统更稳定
        'max_E_err': 1.e-10,
        'max_sweeps': 100,
        'trunc_params': {
            'chi_max': 100,
            'svd_min': 1.e-10
        },
        'verbose': 0, # 关闭详细输出
    }
    
    eng = dmrg.run(psi, M, dmrg_params)
    return eng['E']