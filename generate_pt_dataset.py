import pandas as pd
import os
from tqdm import tqdm
from common.ultils import *
from dataset import CPI3DDataset
import os

def data_making_path(data_name= None, dataset_folder = None):
    data_complex_name=[]
    data_comps = []
    data_prots = []
    data_labels = []
    for complex_name in tqdm(os.listdir(dataset_folder)):
        complex_dir = os.path.join(dataset_folder,complex_name)
        if len(os.listdir(complex_dir))>=3:
            index = 0
            for file in os.listdir(complex_dir):

                if file.endswith('rank0.sdf'):
                    data_comps.append(os.path.join(complex_dir,file))
                    index+=1
                if file.endswith('fold.pdb'):
                    data_prots.append(os.path.join(complex_dir,file))
                    index+=1
                if file.endswith('label.csv'):
                    label = pd.read_csv(os.path.join(complex_dir,file))
                    data_labels.append(label['label'].iloc[0])
                    index+=1
            if index ==3:
                data_complex_name.append(complex_name)

    df = pd.DataFrame({'complex_name':data_complex_name,'ligand':data_comps,'receptor':data_prots,'label':data_labels})
    df.to_csv(data_name)
    return data_name

def data_making_path_or_diffdock(data_name= None, dataset_folder = None):
    data_complex_name=[]
    data_comps = []
    data_prots = []
    data_labels = []

    or_diff_pre_fold = #predictions_diffdock_folder 

    for complex_name in tqdm(os.listdir(dataset_folder)):
        complex_dir = os.path.join(dataset_folder,complex_name)

        or_diff_pre_fold_complex = os.path.join(or_diff_pre_fold,complex_name)
        if len(os.listdir(complex_dir))>=3:
            index = 0
            for file in os.listdir(complex_dir):
                if file.endswith('rank0.sdf'):
                    data_comps.append(os.path.join(or_diff_pre_fold_complex,'rank1.sdf'))
                    index+=1
                if file.endswith('fold.pdb'):
                    data_prots.append(os.path.join(complex_dir,file))
                    index+=1
                if file.endswith('label.csv'):
                    label = pd.read_csv(os.path.join(complex_dir,file))
                    data_labels.append(label['label'].iloc[0])
                    index+=1
            if index ==3:
                data_complex_name.append(complex_name)

    df = pd.DataFrame({'complex_name':data_complex_name,'ligand':data_comps,'receptor':data_prots,'label':data_labels})
    df.to_csv(data_name)
    return data_name

def data_making_path_or_diffdock_classification(data_name, dataset_folder):
    csv_dataset = #Datasets.csv
    df_csv_dataset = pd.read_csv(csv_dataset)
    data_complex_name=[]
    data_comps = []
    data_prots = []
    data_labels = []

    for complex_name in tqdm(os.listdir(dataset_folder)):
        complex_dir = os.path.join(dataset_folder,complex_name)
        if len(os.listdir(complex_dir))>=3:

            data_comps.append(os.path.join(complex_dir,'rank1.sdf'))
            data_prots.append(df_csv_dataset[df_csv_dataset['complex_name']==int(complex_name)]['protein_path'].to_list()[0])
            data_labels.append(df_csv_dataset[df_csv_dataset['complex_name']==int(complex_name)]['label'].to_list()[0])
            data_complex_name.append(complex_name)
    df = pd.DataFrame({'complex_name':data_complex_name,
                       'ligand':data_comps,
                       'receptor':data_prots,
                       'label':data_labels})
    df.to_csv(data_name)
    return data_name

def main(task, dataset_name, data_name_csv):
    if task == 'classification':
        CPI3DDataset(df = pd.read_csv(data_name_csv), 
                        data_name = 'data_bindingDB_train_{}label_encode111'.format(dataset_name), 
                        processed_dir_data = './processed_data/{}diff_classification'.format(dataset_name),
                        protein_pt = '/dude_prot_classification.pt')
    else:
        df = pd.read_csv(data_name_csv)
        task = 'novel_comp' # novel_prot/ novel_comp/ novel_pair 
        tasks = ['novel_prot', 'novel_comp', 'novel_pair']
        for task in tasks:
            for fold in range(5):

                train_csv_path = pd.read_csv('/{}/{}_{}_train.csv'.format(task,task,fold))
                val_csv_path = pd.read_csv('/{}/{}_{}_val.csv'.format(task,task,fold))
                test_csv_path = pd.read_csv('/{}/{}_{}_test.csv'.format(task,task,fold))

                df_train = df_for_ex(df, train_csv_path)
                df_val = df_for_ex(df, val_csv_path)
                df_test = df_for_ex(df, test_csv_path)

                CPI3DDataset(df = df_train, 
                            data_name = 'data_bindingDB_train_{}{}label_encode'.format(task,fold), 
                            processed_dir_data = './processed_data/')
                CPI3DDataset(df = df_val, 
                            data_name = 'data_bindingDB_val_{}{}label_encode'.format(task,fold), 
                            processed_dir_data = './processed_data/')
                CPI3DDataset(df = df_test, 
                            data_name = 'data_bindingDB_test_{}{}label_encode'.format(task,fold), 
                            processed_dir_data = './processed_data/')    


if __name__ == '__main__':
    # ML task
    task = str(sys.argv[1])
    # dataset name 
    dataset_name = str(sys.argv[2])
    # csv file name (to compare overlap, some interactions can not be predicted)
    data_name_csv = str(sys.argv[3])

    # task = 'classification'
    # dataset_name = 'dude_classification'
    # data_name_csv = 'dude_classification_or1.csv'
    main(task, dataset_name, data_name_csv)
