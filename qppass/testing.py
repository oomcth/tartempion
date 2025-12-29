import numpy as np
import example_robot_data as erd
import pinocchio as pin
from pathlib import Path
import tartempion


src_path = Path("model/src")
files = [str(p) for p in src_path.rglob("*")]

robot = erd.load("ur5")
rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
    "model/mantis.urdf", package_dirs=files
)

init_pos = pin.neutral(rmodel)
init_pos[len(init_pos) - 5] = -np.pi / 2
init_pos[10] = -np.pi / 2
rmodel = pin.buildReducedModel(rmodel, list(range(7, len(init_pos) + 1)), init_pos)
tartempion.Test(rmodel, False)
