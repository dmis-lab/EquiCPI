import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Dataset
from common.get_infor_sdf import rec_atom_featurizer
import biotite.structure.io as strucio
import scipy.spatial as spa


class Gen3DProteinGraph(Dataset):
    def __init__(self, filename, processed_dir, pt_filename, cutoff=30, max_neighbors=20, feature_size=200):
        super().__init__()
        self.data = pd.read_csv(filename)
        self.processed_dir = processed_dir
        self.pt_filename = pt_filename
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.feature_size = feature_size
        self.residue_names = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
            'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
            'TYR', 'VAL'
        ]
        self.atom_names = [
            'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG',
            'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE',
            'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2',
            'OG', 'OG1', 'OH', 'SD', 'SG'
        ]

        os.makedirs(self.processed_dir, exist_ok=True)
        pt_path = os.path.join(self.processed_dir, self.pt_filename)
        if os.path.isfile(pt_path):
            self.data_processed = torch.load(pt_path)
        else:
            self.data_processed = self.process_data()
            torch.save(self.data_processed, pt_path)

    def read_structure(self, pdb_path):
        return strucio.load_structure(pdb_path)

    def extract_features(self, struct):
        feature_res = {}
        res_ids = set(struct.res_id)

        for res_id in tqdm(res_ids, desc="Processing residues"):
            res_feat = []
            for atom in struct:
                if atom.res_id == res_id:
                    try:
                        res_index = self.residue_names.index(atom.res_name)
                        atom_index = self.atom_names.index(atom.atom_name)
                        element = Chem.MolFromSmiles(atom.element)
                        element_feat = rec_atom_featurizer(element.GetAtoms()[0]) if element else [0] * (self.feature_size - 2)
                        res_feat.append(atom_index)
                        res_feat.extend(element_feat)
                    except Exception as e:
                        print(f"Warning: Failed to extract features for atom: {e}")
                        continue

            if res_feat:
                full_feat = [res_index] + res_feat
                full_feat = full_feat[:self.feature_size] + [0] * (self.feature_size - len(full_feat))
                feature_res[res_id] = full_feat

        return np.array(list(feature_res.values()))

    def compute_knn_edges(self, coords):
        distances = spa.distance.cdist(coords, coords)
        src, dst = [], []

        for i in range(len(coords)):
            neighbors = list(np.where(distances[i] < self.cutoff)[0])
            if i in neighbors:
                neighbors.remove(i)

            if not neighbors:
                neighbors = list(np.argsort(distances[i]))[1:2]

            if self.max_neighbors and len(neighbors) > self.max_neighbors:
                neighbors = list(np.argsort(distances[i]))[1: self.max_neighbors + 1]

            src.extend([i] * len(neighbors))
            dst.extend(neighbors)

        return np.array([src, dst], dtype=np.int64)

    def process_data(self):
        container = {}

        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Processing proteins"):
            prot_name = os.path.splitext(os.path.basename(row['receptor']))[0]
            struct = self.read_structure(row['receptor'])
            coords = [atom.coord for atom in struct if atom.atom_name == 'CA']

            if len(coords) <= 1:
                print(f"Skipping {prot_name}: only 1 residue")
                continue

            features = self.extract_features(struct)
            edge_index = self.compute_knn_edges(coords)

            graph = {
                'receptor_x': torch.tensor(features, dtype=torch.float),
                'receptor_pos': torch.tensor(coords, dtype=torch.float),
                'receptor_edge_index': torch.tensor(edge_index, dtype=torch.long)
            }

            container[prot_name] = graph

        return container

    def len(self):
        return len(self.data_processed)

    def get(self, idx):
        return self.data_processed[list(self.data_processed.keys())[idx]]


def main(dataset_folder, csv_path, processed_dir, pt_filename):
    pdb_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.pdb')]
    df = pd.DataFrame({'receptor': pdb_files})
    df.to_csv(csv_path, index=False)
    Gen3DProteinGraph(filename=csv_path, processed_dir=processed_dir, pt_filename=pt_filename)


if __name__ == '__main__':
    dataset_folder = str(sys.argv[1])
    csv_path = str(sys.argv[2])
    processed_dir = str(sys.argv[3])
    pt_filename = str(sys.argv[4])
    main(dataset_folder, csv_path, processed_dir, pt_filename)
