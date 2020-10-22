from rdkit import Chem
import numpy as np
from pathlib import Path

from gencond.properties import PROPERTIES

def process_properties(set):
    target = "data/%s_props.npz" % set
    if Path(target).exists():
        print("Skipping %s data; file %s already exists" % (set, target))
        return

    smiles = []
    props = []
    for row in open("data/Guacamol/guacamol_v1_%s.smiles" % set):
        row = row.strip()
        smiles.append(row)
        props.append([prop(Chem.MolFromSmiles(row)) for prop in PROPERTIES.values()])
        if len(smiles) % 10000 == 0:
            print("processed %d strings" % len(smiles))
    props = np.array(props)
    np.savez_compressed(target, smiles, props)


if __name__ == '__main__':
    process_properties("train")
    process_properties("valid")
    process_properties("test")
