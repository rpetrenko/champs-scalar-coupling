import numpy as np
import pandas as pd


class Atom(object):
    def __init__(self, atom_index, atom_name, xyz):
        self.idx = atom_index
        self.name = atom_name
        self.point = xyz

    def __str__(self):
        return "{} {} {}".format(self.name, self.idx, self.point)


class Structure(object):
    def __init__(self, name, atoms):
        self.name = name
        self._init_atoms(atoms)

    def _init_atoms(self, atoms):
        self.atoms = []
        names = atoms['atom'].values
        atom_index = atoms['atom_index'].values
        coordinates = np.asarray(atoms[['x', 'y', 'z']])
        for i in range(len(names)):
            self.atoms.append(Atom(atom_index[i], names[i], coordinates[i, :]))

    def __str__(self):
        s = self.name
        for atom in self.atoms:
            s = "{}\n  {}".format(s, atom)
        return s


def get_structures(fname, limit=None, seed=42):
    np.random.seed(seed)
    df_structures = pd.read_csv(fname)
    df_structures = df_structures.set_index('molecule_name')
    mol_names = df_structures.index
    if limit:
        mol_names = np.random.choice(mol_names, limit)
    structures = list()
    for mol_name in mol_names:
        # print(mol_name)
        df_s = df_structures.loc[mol_name]
        # print(df_s)
        structure = Structure(mol_name, df_s[['atom_index', 'atom', 'x', 'y', 'z']])
        print(structure)
        structures.append(structure)
    return structures


def main():
    # load structures
    structures_file = "data/structures.csv"
    structures = get_structures(structures_file, limit=1)

    # load train data
    # df_train = pd.read_csv("data/train.csv")
    # print(df_train.shape)

    # load test data
    # df_test = pd.read_csv("data/test.csv")
    # print(df_test.shape)


if __name__ == "__main__":
    main()
