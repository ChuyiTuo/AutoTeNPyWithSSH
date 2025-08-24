from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import logging

from pathlib import Path
import yaml
import inspect
import math
import numpy as np
import tenpy


# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()

# MCP tools
def load_yaml(path):
    path = Path(path)
    if path.exists():
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
        with path.open("w") as f:
            yaml.safe_dump(data, f)
    return data

def dump_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f)


_ALLOWED_LATTICES_1D = {"Chain"}
_ALLOWED_LATTICES_2D = {"Square", "Triangular", "Honeycomb",  "Kagome"}
_ALLOWED_BC = {"open", "periodic"}


@mcp.tool()
def spinhalf_lattice_config(lattice_type: str, Lx: int, Ly: int | None, 
                            bc_x: str = "open", bc_y: str | None = "periodic", 
                            path: str | Path | None = None) -> TextContent:
    """
    Build and write a tenpy lattice configuration for spin 1/2 models into a yaml file at the given path.

    Args:
        lattice_type: 
                The type of lattice. Must be one of the currently supported lattices:
                1D: Chain.
                2D: Square, Triangular, Honeycomb, Kagome.  
        Lx: 
            Lx of 2D lattice / L of 1D lattice.
        Ly: 
            Ly of 2D lattice / ignored for 1D lattice (better to set None).
        bc_x:
            Boundary condition along x direction of 2D lattice / Boundary condition of 1D lattice (better to set None). Default to be open.
        bc_y:
            Boundary condition along y direction of 2D lattice / ignored for 1D lattice. Default to be periodic.
        path:
            The path of the yaml file. If set to None, the default path is set to Path.cwd()/config.yml.
    
    Return:
        The source code of the init_terms of the model_class. The LLM should read the source code and tell the user what physical parameters are further required (with their default values). It is better providing a explanation of each terms.

    Note: 
        For 2D lattices with cylinder boundary condition (i.e. bc_x = open and bc_y = periodic), larger lattice extent should correspond to the open boundary direction (i.e. Lx > Ly).
    """
    model_class = "SpinModel"
    S = 0.5

    if lattice_type in _ALLOWED_LATTICES_1D:
        d = 1
        Ly = None
        bc_y = None
    elif lattice_type in _ALLOWED_LATTICES_2D:
        d = 2
    else:
        raise ValueError(f"Unsupported lattice_type: {lattice_type}")

    if bc_x not in _ALLOWED_BC:
        raise ValueError(f"Invalid bc_x: {bc_x}")
    if d == 2 and bc_y not in _ALLOWED_BC:
        raise ValueError(f"Invalid bc_y: {bc_y}")

    path = Path(path) if path is not None else (Path.cwd() / "config.yml")
    data = load_yaml(path)
    data["model_class"] = model_class
    model_params = data.setdefault("model_params", {})
    
    model_params["lattice"] = lattice_type
    model_params["S"] = S
    model_params["bc_MPS"] = "finite" 

    if d == 1:
        model_params["L"] = Lx
        model_params["bc_x"] = bc_x

        model_params.pop("Lx", None)
        model_params.pop("Ly", None)
        model_params.pop("bc_y", None)
    elif d == 2:
        model_params["Lx"] = Lx
        model_params["Ly"] = Ly
        model_params["bc_x"] = bc_x
        model_params["bc_y"] = bc_y

        model_params.pop("L", None)
    else:
        raise ValueError
    
    dump_yaml(data, path)
    return TextContent(type="text", text=inspect.getsource(eval(f"tenpy.models.{model_class}.init_terms")))



@mcp.tool()
def spinhalf_model_config(params_dict: dict[str, float],
                          path: str | Path | None = None) -> TextContent:
    """
    Build and write a tenpy model configuration for spin 1/2 models into a yaml file at the given path. The LLM should parse the model parameters into params_dict as a dictionary from natural language inputs (do not use the string representation of the dict). 

    Args:
        params_dict:
            A dictionary of model parameters. 
            key: The name of parameters, which is most likely provided by LLM's reading of the init_terms function of the model.
            value: The value of model parameters, provided by users. 
        path:
            The path of the yaml file. If set to None, the default path is set to Path.cwd()/config.yml.
    
    Return:
        A sentence indicating the conservation property of the model we defined. 
        If there is some conservation property, the LLM should require the user further providing the total symmetry charge of the problem they intersted, which will be further used to set the initial product state. 
        If there is no any conservation property, the user need not provide any data, the LLM should just ask the user whether generating random product state now.
    """

    path = Path(path) if path is not None else (Path.cwd() / "config.yml")
    data = load_yaml(path)

    model_params = data.setdefault("model_params", {})
    for name, val in params_dict.items():
        model_params[name] = val
    
    model_params["conserve"] = "best"
    dump_yaml(data, path)

    model_obj =  eval(f"tenpy.models.{data["model_class"]}(model_params)")
    is_conserve_str = model_obj.lat.site(0).conserve
    return TextContent(type="text", text=f"The conservation property of the model is: {is_conserve_str}")



@mcp.tool()
def spinhalf_initial_state(tot_charge: float | None, path: str | None = None) -> TextContent:
    """
    Build and write a tenpy initial state configuration for DMRG simulations of spin 1/2 models into a yaml file at the given path. 

    Args:
        tot_charge: 
            The total symmetry charge of the initial state. For Sz symmetry case, total_charge is sum over Sz for all sites, which should be some integer multiple of 1/2. For no symmetry case, set total_charge to None.
        path:
            The path of the yaml file. If set to None, the default path is set to Path.cwd()/config.yml.

    Return:
        A sentence of successful generation of initial state, and guide the LLM to further generate the DMRG algorithm parameters.
    """
    if tot_charge is not None and not math.isclose(2 * tot_charge, int(2 * tot_charge)):
        raise ValueError("tot_charge is not half integer! ")

    path = Path(path) if path is not None else (Path.cwd() / "config.yml")
    data = load_yaml(path)
    model_params = data.get("model_params", {}).copy()

    model_obj =  eval(f"tenpy.models.{data["model_class"]}(model_params)")
    lat = model_obj.lat
    is_conserve_str = lat.site(0).conserve

    if is_conserve_str not in ["Sz", None]:
        raise ValueError("Symmetry not yet supported! ")
    
    shape = lat.shape
    n_sites = lat.N_sites

    if tot_charge is None:
        n_up = np.random.randint(n_sites)
        n_dn = n_sites - n_up
    else:
        n_up = tot_charge + n_sites / 2
        n_dn = -tot_charge + n_sites / 2
        if not (math.isclose(n_up, int(n_up)) and math.isclose(n_dn, int(n_dn))):
            raise ValueError("Inconsistent tot_charge and system size! ")
        n_up = int(n_up)
        n_dn = int(n_dn)

    init = np.array(["up"] * n_up + ["down"] * n_dn, dtype=str)
    np.random.shuffle(init)
    init = init.reshape(shape).tolist()

    init_state_params = data.setdefault("initial_state_params", {})
    init_state_params["method"] = "lat_product_state"
    init_state_params["product_state"] = init

    dump_yaml(data, path) 
    return TextContent(type="text", text=f"The DMRG initial state is successfully generated! We should then set the DMRG algorithm parameters and measurement parameters. ")

@mcp.tool()
def dmrg_and_measu_config(chi_max: int, max_sweeps: int, path: str | None = None) -> TextContent:
    """
    Generate the DMRG algorithm parameters and measurement parameters. The inputs are some main DMRG algorithm parameters. The measurements are not inputted (some common measurements are performed internally).

    Args:
        chi_max: 
            the maximum DMRG bond dimension.
        max_sweeps: 
            the maximum number of sweeps.
        path:
            The path of the yaml file. If set to None, the default path is set to Path.cwd()/config.yml.

    Return:
        A prompt for successfully generating all DMRG configurations. Ready for computation.
    """
    path = Path(path) if path is not None else (Path.cwd() / "config.yml")
    data = load_yaml(path)  
    
    data["simulation_class"] = "GroundStateSearch"
    data["algorithm_params"] = {
        'algorithm': 'TwoSiteDMRGEngine',
        'max_sweeps': max_sweeps,
        'mixer': False,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': 1e-8
        }
    }

    data["measure_initial"] = False
    data["connect_measurements"] = [
            ["tenpy.simulations.measurement", "m_onsite_expectation_value",
                {"opname": "Sz"}],
            ["psi_method", "wrap correlation_function",
                {"results_key": "<Sz_i Sz_j>", "ops1": "Sz", "ops2": "Sz"}],
            ["psi_method", "wrap correlation_function",
                {"results_key": "<Sp_i Sm_j>", "ops1": "Sp", "ops2": "Sm"}],
        ]

    data["output_filename_params"] = {"prefix": "result", "suffix": ".h5"}

    dump_yaml(data, path)
    return TextContent(type="text", text=f"All required configurations are generated. Ready for performing computations! ")


@mcp.tool()
def dmrg_run(path: str | None = None):
    """
    Run the DMRG program with the configuration file at path.

    Args:
        path:
            The path of the yaml file. If set to None, the default path is set to Path.cwd()/config.yml.

    Return:
        A prompt indicating the path of DMRG result and the path of log.
    """
    path = Path(path) if path is not None else (Path.cwd() / "config.yml")
    simulation_params = tenpy.load_yaml_with_py_eval(path)
    tenpy.run_simulation(**simulation_params)
    result_path = simulation_params["output_filename_params"]["prefix"] + simulation_params["output_filename_params"]["suffix"]
    log_path = simulation_params["output_filename_params"]["prefix"] + ".log"
    return TextContent(type="text", text=f"The DMRG result is stored at {result_path}. The log is store at {log_path}.")




# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()
