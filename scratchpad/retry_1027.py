import sys, json
from pathlib import Path
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia/scratchpad")
import boundary_diag as bd
r = bd.process("1027")
allres = json.loads((Path(bd.__file__).parent / "boundary_diag_results.json").read_text())
allres = [x for x in allres if x["sid"] != "bb_1027"] + [r]
(Path(bd.__file__).parent / "boundary_diag_results.json").write_text(json.dumps(allres, indent=2))
print("merged results saved")
