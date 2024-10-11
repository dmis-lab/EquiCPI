import warnings
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable
from Bio.PDB import PDBParser
import torch.nn.functional as F


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def rec_residue_featurizer(struct):
    feature_list = []
    for residue in struct:
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.res_name)])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)

def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])
    return torch.tensor(atom_features_list)
def onek_encoding_unk(value: int, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def lig_atom_one_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_one_features = [
            onek_encoding_unk(safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),allowable_features['possible_atomic_num_list']),
            onek_encoding_unk(allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),allowable_features['possible_chirality_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),allowable_features['possible_degree_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),allowable_features['possible_formal_charge_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),allowable_features['possible_implicit_valence_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),allowable_features['possible_numH_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),allowable_features['possible_number_radical_e_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),allowable_features['possible_hybridization_list']),
            onek_encoding_unk(allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),allowable_features['possible_is_aromatic_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),allowable_features['possible_numring_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),allowable_features['possible_is_in_ring3_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),allowable_features['possible_is_in_ring4_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),allowable_features['possible_is_in_ring5_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),allowable_features['possible_is_in_ring6_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),allowable_features['possible_is_in_ring7_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),allowable_features['possible_is_in_ring8_list'])
        ]
        atom_f = []
        for one_feature in atom_one_features:
            atom_f.extend(one_feature)
        atom_features_list.append(atom_f)
    return torch.tensor(atom_features_list)


def lig_atom_one_exceptatomnums_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        # atom_one_features = []
        atom_one_features = [
            [safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum())],
            onek_encoding_unk(allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),allowable_features['possible_chirality_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),allowable_features['possible_degree_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),allowable_features['possible_formal_charge_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),allowable_features['possible_implicit_valence_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),allowable_features['possible_numH_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),allowable_features['possible_number_radical_e_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),allowable_features['possible_hybridization_list']),
            onek_encoding_unk(allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),allowable_features['possible_is_aromatic_list']),
            onek_encoding_unk(safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),allowable_features['possible_numring_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),allowable_features['possible_is_in_ring3_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),allowable_features['possible_is_in_ring4_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),allowable_features['possible_is_in_ring5_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),allowable_features['possible_is_in_ring6_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),allowable_features['possible_is_in_ring7_list']),
            onek_encoding_unk(allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),allowable_features['possible_is_in_ring8_list'])
        ]
        atom_f = []
        for one_feature in atom_one_features:
            atom_f.extend(one_feature)
        atom_features_list.append(atom_f)
    return torch.tensor(atom_features_list)

def rec_atom_featurizer(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


def read_molecule(molecule_file, sanitize=True, calc_charges=True, remove_hs=True):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=True, removeHs=True)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=True, removeHs=True)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol

def get_ligand_graph(mol, type_encode):
    # 3D position
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    # atom features
    if type_encode == 'one_hot_encoding':
        atom_feats = lig_atom_one_featurizer(mol)
    if type_encode == 'label_encoding': 
        atom_feats = lig_atom_featurizer(mol)
    if type_encode == 'one_hot_encoding_except':
        atom_feats = lig_atom_one_exceptatomnums_featurizer(mol)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    return lig_coords, atom_feats, edge_index, edge_attr

