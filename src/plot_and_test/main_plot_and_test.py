from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging

from exact_ising import quspin_ising_ground_energy
from dmrg_ising import tenpy_ising_ground_energy

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()


@mcp.tool()
def quspin_ising_energy(L: int, J: float, h: float) -> float:
    """Compute the ground state energy of the 1D transverse field Ising model using QuSpin.
    
    Args:
        L (int): System size
        J (float): Coupling strength
        h (float): Transverse field strength
    
    Returns:
        float: Ground state energy
    """
    return quspin_ising_ground_energy(L, J, h)

@mcp.tool()
def tenpy_ising_energy(L: int, J: float, h: float) -> float:
    """Use TeNPy to compute the ground state energy of the 1D transverse field Ising model.
    
    Args:
        L (int): System size
        J (float): Coupling strength
        h (float): Transverse field strength
    
    Returns:
        float: Ground state energy
    """
    return tenpy_ising_ground_energy(L, J, h)


import numpy as np
import matplotlib.pyplot as plt
import h5py
from tenpy.tools import hdf5_io
from scipy.optimize import curve_fit
import base64
import io

@mcp.tool()
def plot_entanglement_entropy(filename: str) -> ImageContent:
    """Calculates and plots the entanglement entropy from DMRG data,
    fits it to the CFT scaling form, and returns the plot as a base64 encoded image.
    
    Args:
        filename (str): The path to the HDF5 file containing the DMRG results.
    Returns:
        ImageContent: An object containing the base64 encoded PNG image of the plot.
    """
    def cft_scaling(l, c, const):
        """CFT scaling form for entanglement entropy in a finite system with periodic BC."""
        L = 32  # System size is fixed in our data
        # The argument of the log is the chord length
        return (c / 6.) * np.log((L / np.pi) * np.sin(np.pi * l / L)) + const

    # Load the wavefunction at the critical point
    with h5py.File(filename, 'r') as f:
        psi_data = hdf5_io.load_from_hdf5(f)
        psi = psi_data['psi']

    L = len(psi.sites)

    # We calculate entropy for cuts from l=1 to l=L-1
    subsystem_sizes = np.arange(1, L)
    entanglement_entropy = [psi.entanglement_entropy(i) for i in subsystem_sizes]

    # Perform the curve fit
    # We exclude the edges (l=1 and l=L-1) from the fit
    popt, pcov = curve_fit(cft_scaling, subsystem_sizes[1:-1], entanglement_entropy[0][1:-1])
    fitted_c = popt[0]
    fitted_const = popt[1]
    c_error = np.sqrt(np.diag(pcov))[0]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(subsystem_sizes, entanglement_entropy[0], 'o', label='DMRG Data')
    plt.plot(subsystem_sizes, cft_scaling(subsystem_sizes, fitted_c, fitted_const), '-',
             label=f'CFT Fit (c = {fitted_c:.4f} \pm {c_error:.4f})')
    plt.xlabel('Subsystem size l')
    plt.ylabel('Entanglement Entropy S(l)')
    plt.title('Entanglement Entropy at the Critical Point (Jz=1.0, L=32)')
    plt.legend()
    plt.grid(True)

    # Save plot to a bytes buffer and encode in base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return ImageContent(data=image_base64, mimeType="image/png", type="image")



# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()
