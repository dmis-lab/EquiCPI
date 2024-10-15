import pandas as pd
from rdkit import Chem
import numpy as np 
import os
from tqdm import tqdm
import sys
import torch
import torch_geometric
from torch_geometric.data import Dataset
from common.get_infor_sdf import rec_atom_featurizer

import biotite.structure.io as strucio

import scipy.spatial as spa

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class Gen3Dprteingraph(Dataset):
    def __init__(self, filename, processed_dir_data, pt_file_name, test=False):

        super(Gen3Dprteingraph, self).__init__()
        self.data = pd.read_csv(filename)
        self.processed_dir_data = processed_dir_data

        self.pt_file_name = pt_file_name
        self.feature_size = 200
        self.cutoff = 30
        self.max_neighbor = 20
        if os.path.isfile(os.path.join(self.processed_dir_data, self.pt_file_name)):
            self.data_processed = torch.load(os.path.join(self.processed_dir_data, self.pt_file_name))
        else:
            os.makedirs(self.processed_dir_data, exist_ok=True)
            self.data_processed = self.process_data()

    def process_data(self):
        data_container = {}
        
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            prot_fold_name = row['receptor'].split('/')[-1].split('.')[0]
            
            def rec_all_features(struct):
                #residue
                res_ids = set(struct.res_id)
                feature_res = {}
                res_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'] 
                atom_names = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'SD', 'SG']
                for res_id in tqdm(res_ids):
                    res_features = []
                    dummy_array = [0]*self.feature_size
                    for res in struct:
                        if res.res_id == res_id:
                            resnames_tensor = res_names.index(res.res_name)
                            #atom
                            rec_feature1 = atom_names.index(res.atom_name)
                            # rec_feature1= F.one_hot(atomnames_tensor, num_classes = len(atom_names))
                            res_features.append(rec_feature1)
                            #element
                            rec_feature2 = rec_atom_featurizer(Chem.MolFromSmiles(res.element).GetAtoms()[0])
                            res_features.extend(rec_feature2)
                    res_features.insert(0, resnames_tensor)
                    res_features.extend(dummy_array)
                    feature_res[res_id] = res_features[:self.feature_size]
                res_features_f = np.array(list(feature_res.values()))
                
                return res_features_f

            def read_rec_pdb(pdb_path):
                struct = strucio.load_structure(pdb_path)
                c_alpha_coords = [list(atom.coord) for atom in struct if atom.atom_name == 'CA']
                rec_features = rec_all_features(struct)
                return c_alpha_coords, rec_features

            c_alpha_coords, rec_features = read_rec_pdb(row['receptor'])
            num_residues = len(c_alpha_coords)
            if num_residues <= 1:
                raise ValueError(f"rec contains only 1 residue!")

            # Build the k-NN graph
            distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
            src_list = []
            dst_list = []
            mean_norm_list = []
            data = {}
            for i in range(num_residues):
                dst = list(np.where(distances[i, :] < self.cutoff)[0])
                dst.remove(i)
                if self.max_neighbor != None and len(dst) > self.max_neighbor:
                    dst = list(np.argsort(distances[i, :]))[1: self.max_neighbor + 1]
                if len(dst) == 0:
                    dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
                    print(f'The c_alpha_cutoff {self.cutoff} was too small for one c_alpha such that it had no neighbors. '
                          f'So we connected it to the closest other c_alpha')
                assert i not in dst
                src = [i] * len(dst)
                src_list.extend(src)
                dst_list.extend(dst)

            assert len(src_list) == len(dst_list)
            data['receptor_x'] = torch.from_numpy(rec_features)
            data['receptor_pos'] = torch.tensor(np.array(c_alpha_coords)).float()
            data['receptor_edge_index'] = torch.tensor(torch.from_numpy(np.asarray([src_list, dst_list])), dtype=torch.long)        
            data_container.update({prot_fold_name:data})

        torch.save(data_container,os.path.join(self.processed_dir_data,self.pt_file_name))
        return torch.load(os.path.join(self.processed_dir_data, self.pt_file_name))

    def len(self):
        return len(self.data_processed)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        return self.data_processed[idx]

def main(dataset_folder, filename, processed_dir_data, pt_file_name):
    data_prots = []

    for prot_name in tqdm(os.listdir(dataset_folder)):
        if prot_name.endswith('.pdb'):
            data_prots.append(os.path.join(dataset_folder,prot_name))

    df = pd.DataFrame({'receptor':data_prots})
    df.to_csv(filename)

    Gen3Dprteingraph(filename = filename, processed_dir_data = processed_dir_data, pt_file_name = pt_file_name)

if __name__ == '__main__':
    # dataset_folder = '/ssd1/quang/moldock/Benchmark_data/for_equi/esm/esm1binddingdb_data'
    # filename = 'data_bindingDB_prot_classification.csv'
    # processed_dir_data = '/ssd1/quang/moldock/e3nn_cpi_project/processed/bindingDB_class_prot'
    # pt_file_name = 'bindingDB_prot_classification.pt'
    # main(dataset_folder, filename, processed_dir_data, pt_file_name)

    # ESM output
    dataset_folder = str(sys.argv[1])
    # protein file name
    filename = str(sys.argv[2])
    # processed_dir
    processed_dir_data = str(sys.argv[3])
    # processed_pt_name
    pt_file_name = str(sys.argv[4])
    main(dataset_folder, filename, processed_dir_data, pt_file_name)
