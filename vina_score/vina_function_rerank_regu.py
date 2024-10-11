

from vina import Vina
from tqdm import tqdm
from openbabel import pybel

import numpy as np
import os
import pandas as pd
import numpy as np
import shutil


def convert_sdf_to_pdbqt(input_sdf, output_pdbqt):
    # Create an input molecule object
    input_molecule = next(pybel.readfile("sdf", input_sdf))

    # Convert the molecule to PDBQT format
    pdbqt_molecule = pybel.Molecule(input_molecule)
    pdbqt_molecule.addh()
    pdbqt_molecule.write("pdbqt", output_pdbqt, overwrite=True)


def take_center(input_file):
    input_molecule = next(pybel.readfile("sdf", input_file))
    center = [atom.coords for atom in input_molecule.atoms]
    center = list(np.mean(center,0))
    return center
    

def calculate_vina_score(output_pdbqt_ligand, output_pdbqt_receptor,center):
    v = Vina(sf_name='vina')
    v.set_receptor(output_pdbqt_receptor)
    v.set_ligand_from_file(output_pdbqt_ligand)
    v.compute_vina_maps(center=center, box_size=[20, 20, 20])

    # Score the current pose
    # energy = v.optimize()
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])
    return energy[0]

if __name__ == "__main__":
    # ['split_2.csv', 'split_3.csv', 'split_4.csv', 'split_0.csv', 'split_1.csv', 'split_5.csv']
    # split_data_fold = '/ssd1/quang/moldock/Benchmark_data/for_equi/for_diff_encode/bindingdb_split'
    import sys
    diff_pre_folder = str(sys.argv[2])

    data_pur = 'regu_data'
    data_equi = str(sys.argv[1])   
    temp_dir = diff_pre_folder+'{}'.format(data_pur)+'/temp'
    new_rank_dataset_dir = diff_pre_folder+'{}'.format(data_pur)+'/rerank'
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(new_rank_dataset_dir, exist_ok=True)

    df = pd.read_csv(data_equi)
    complex_name = df['complex_name']
    receptor_path = df['protein_path']
    label = df['label']
    no_over = [i for i in tqdm(complex_name) if str(i) not in os.listdir(new_rank_dataset_dir)]
    for complex_n in tqdm(no_over):

        try:
            pre_poses_folder = os.path.join(diff_pre_folder,str(complex_n))
            poses = os.listdir(pre_poses_folder)
            re_pdb_path = list(df[df['complex_name']==complex_n]['protein_path'])[0]
            re_pdb_name = re_pdb_path.split('/')[-1][:-4]

            temp_dir_comp = os.path.join(temp_dir, str(complex_n))
            os.makedirs(temp_dir_comp, exist_ok=True)
            new_rank_dataset_dir_comp = os.path.join(new_rank_dataset_dir, str(complex_n))
            os.makedirs(new_rank_dataset_dir_comp,exist_ok=True)
            
            re_pdbqt_path = os.path.join(temp_dir_comp, re_pdb_name+'.pdbqt')
            os.system('/ssd1/quang/moldock/dock_vina/build/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor -r {} -o {}'.format(re_pdb_path,re_pdbqt_path))
            rank_dict = {}
            for pose in poses:
                if pose.split('_')[0] in ['rank0','rank1','rank2','rank3','rank4','rank5']:
                    pose_path_sdf = os.path.join(pre_poses_folder,pose)
                    if pose.split('-')[0]== 'rank1_confidence':
                        shutil.copy(os.path.join(pre_poses_folder,pose),os.path.join(new_rank_dataset_dir_comp,pose))
                    center = take_center(pose_path_sdf)
                    pose_path_pdbqt = os.path.join(temp_dir_comp,pose[:-4]+'.pdbqt')
                    convert_sdf_to_pdbqt(pose_path_sdf, pose_path_pdbqt)
                    energy = calculate_vina_score(pose_path_pdbqt,re_pdbqt_path,center)
                    rank_dict[pose] = energy
                else:
                    pass
            
            re_rank_dict = dict(sorted(rank_dict.items(), key=lambda item: item[1]))
            list_re_rank_pose = list(re_rank_dict.keys())
            shutil.copy(os.path.join(pre_poses_folder,list_re_rank_pose[0]),os.path.join(new_rank_dataset_dir_comp,'{}{}.sdf'.format(list_re_rank_pose[0],'rank0')))
            shutil.copy(os.path.join(pre_poses_folder,list_re_rank_pose[1]),os.path.join(new_rank_dataset_dir_comp,'{}{}.sdf'.format(list_re_rank_pose[1],'rank1')))
            shutil.copy(re_pdb_path,os.path.join(new_rank_dataset_dir_comp,re_pdb_name+'.pdb'))
            df_pre = pd.DataFrame([re_rank_dict[list_re_rank_pose[0]]],columns=['affinity_pre'])
            df_pre.to_csv(os.path.join(new_rank_dataset_dir_comp,'pre_vina.csv'))
            df[df['complex_name']==complex_n]['label'].to_csv(os.path.join(new_rank_dataset_dir_comp,'label.csv'))
        except:
            pass
    import pdb
    pdb.set_trace() 
