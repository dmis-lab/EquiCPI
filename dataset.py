from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np 
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Dataset, HeteroData
from common.get_infor_sdf import get_ligand_graph, read_molecule

import os

class CPI3DDataset(Dataset):
    def __init__(self, df = None, data_name = None, 
                 processed_dir_data = None, 
                 protein_pt='/ssd1/quang/moldock/e3nn_cpi_project/processed/bindingDB_prot/bindingDB_prot.pt',
                 test = False,
                 processed_data_pt = None):
        super(CPI3DDataset, self).__init__()
        if os.path.exists(processed_data_pt):
            self.data_processed = torch.load(processed_data_pt)
        else:
            self.data = df.reset_index(drop=True) 
            self.processed_dir_data = processed_dir_data
            self.test = test
            self.data_name = data_name
            self.pt_file_name = 'data_{}.pt'.format(self.data_name)
            self.pt_file_protein = torch.load(protein_pt)
            if os.path.isfile(os.path.join(self.processed_dir_data, self.pt_file_name)):
                self.data_processed = torch.load(os.path.join(self.processed_dir_data, self.pt_file_name))
            else:
                os.makedirs(self.processed_dir_data, exist_ok=True)
                self.data_processed = self.process_data()

    def morgan_fingerprint(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, nBits=2048)
        morgan_feature = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, morgan_feature)
        return morgan_feature
    
    def apply_noise(self, tensor, std):
        if std is None or std == 0.0:
            return tensor
        noise = torch.randn_like(tensor) * std
        return tensor + noise


    def process_data(self):
        
        data_container = {}
        count_index = 0
        list_complex = []
        for _, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = read_molecule(row['ligand'])
            try:
                # 'one_hot_encoding' 'label_encoding' 'one_hot_encoding_except'
                lig_coords, node_feats, edge_index, edge_feats = get_ligand_graph(mol_obj, type_encode='label_encoding') 
                data = HeteroData()

                data['ligand'].pos = self.apply_noise(torch.tensor(np.array(lig_coords.numpy())), self.noise)
                data['ligand'].x = node_feats
                data['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
                data['ligand', 'lig_bond', 'ligand'].edge_attr = edge_feats

                fold_name = row['receptor'].split('/')[-1].split('.')[0]

                data['receptor'].x = self.pt_file_protein[fold_name]['receptor_x']
                data['receptor'].pos = self.pt_file_protein[fold_name]['receptor_pos']
                data['receptor', 'rec_contact', 'receptor'].edge_index = self.pt_file_protein[fold_name]['receptor_edge_index']

                data['ligand'].morgan_fingerprint = torch.tensor([np.float32(self.morgan_fingerprint(mol_obj))])

                data.y = torch.tensor(row['label'])        
                
                data_container.update({count_index:data})
                count_index += 1
                list_complex.append(row['complex_name'])
            except:
                pass
        torch.save(data_container,os.path.join(self.processed_dir_data,self.pt_file_name))
        return torch.load(os.path.join(self.processed_dir_data, self.pt_file_name))


    def len(self):
        return len(self.data_processed)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        return self.data_processed[idx]