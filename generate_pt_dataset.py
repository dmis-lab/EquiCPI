import os
import sys
import pandas as pd
from tqdm import tqdm
from common.ultils import df_for_ex  # Ensure correct import
from dataset import CPI3DDataset


def read_label_from_csv(csv_path: str) -> float:
    """Read the label from a CSV file."""
    try:
        label_df = pd.read_csv(csv_path)
        return label_df['label'].iloc[0]
    except Exception as e:
        print(f"[Warning] Failed to read label from {csv_path}: {e}")
        return None


def collect_data_entries(dataset_folder: str,
                         ligand_ext: str,
                         receptor_ext: str,
                         label_ext: str,
                         ligand_override_folder: str = None) -> pd.DataFrame:
    """
    Traverse the dataset directory to collect ligand, receptor, and label paths.
    """
    complex_names, ligands, receptors, labels = [], [], [], []

    for complex_name in tqdm(os.listdir(dataset_folder), desc="Processing complexes"):
        complex_dir = os.path.join(dataset_folder, complex_name)
        if not os.path.isdir(complex_dir) or len(os.listdir(complex_dir)) < 3:
            continue

        ligand_path = receptor_path = label = None

        for file in os.listdir(complex_dir):
            file_path = os.path.join(complex_dir, file)

            if file.endswith(ligand_ext):
                ligand_path = os.path.join(ligand_override_folder or complex_dir, 'rank1.sdf')
            elif file.endswith(receptor_ext):
                receptor_path = file_path
            elif file.endswith(label_ext):
                label = read_label_from_csv(file_path)

        if all([ligand_path, receptor_path, label is not None]):
            complex_names.append(complex_name)
            ligands.append(ligand_path)
            receptors.append(receptor_path)
            labels.append(label)

    return pd.DataFrame({
        'complex_name': complex_names,
        'ligand': ligands,
        'receptor': receptors,
        'label': labels
    })


def create_dataset_csv(output_csv_path: str,
                       dataset_folder: str,
                       ligand_ext: str,
                       receptor_ext: str,
                       label_ext: str,
                       ligand_override_folder: str = None) -> str:
    """
    Create a CSV file from the dataset directory structure.
    """
    df = collect_data_entries(dataset_folder, ligand_ext, receptor_ext, label_ext, ligand_override_folder)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def create_dataset_from_predefined_csv(output_csv_path: str,
                                       dataset_folder: str,
                                       metadata_csv_path: str) -> str:
    """
    Create a dataset CSV using an external metadata CSV.
    """
    metadata_df = pd.read_csv(metadata_csv_path)
    complex_names, ligands, receptors, labels = [], [], [], []

    for complex_name in tqdm(os.listdir(dataset_folder), desc="Reading metadata"):
        try:
            complex_id = int(complex_name)
            ligand_path = os.path.join(dataset_folder, complex_name, 'rank1.sdf')
            matched_row = metadata_df[metadata_df['complex_name'] == complex_id]

            if matched_row.empty:
                print(f"[Warning] Complex {complex_id} not found in metadata.")
                continue

            receptor_path = matched_row['protein_path'].values[0]
            label = matched_row['label'].values[0]

            complex_names.append(complex_name)
            ligands.append(ligand_path)
            receptors.append(receptor_path)
            labels.append(label)

        except Exception as e:
            print(f"[Error] Skipping complex {complex_name}: {e}")

    df = pd.DataFrame({
        'complex_name': complex_names,
        'ligand': ligands,
        'receptor': receptors,
        'label': labels
    })
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def process_classification_task(df: pd.DataFrame, dataset_name: str):
    """
    Process dataset for classification tasks.
    """
    CPI3DDataset(
        df=df,
        data_name=f'data_bindingDB_train_{dataset_name}label_encode111',
        processed_dir_data=f'./processed_data/{dataset_name}diff_classification',
        protein_pt='/dude_prot_classification.pt'
    )


def process_regression_task(df: pd.DataFrame):
    """
    Process dataset for regression tasks across different split strategies and folds.
    """
    split_tasks = ['novel_prot', 'novel_comp', 'novel_pair']

    for task in split_tasks:
        for fold in range(5):
            try:
                train_df = pd.read_csv(f'/{task}/{task}_{fold}_train.csv')
                val_df = pd.read_csv(f'/{task}/{task}_{fold}_val.csv')
                test_df = pd.read_csv(f'/{task}/{task}_{fold}_test.csv')

                df_train = df_for_ex(df, train_df)
                df_val = df_for_ex(df, val_df)
                df_test = df_for_ex(df, test_df)

                for split, df_split in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):
                    CPI3DDataset(
                        df=df_split,
                        data_name=f'data_bindingDB_{split}_{task}{fold}label_encode',
                        processed_dir_data='./processed_data/'
                    )
            except Exception as e:
                print(f"[Error] Failed on task={task} fold={fold}: {e}")


def main(task: str, dataset_name: str, data_name_csv: str):
    df = pd.read_csv(data_name_csv)

    if task.lower() == 'classification':
        process_classification_task(df, dataset_name)
    else:
        process_regression_task(df)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <task> <dataset_name> <data_name_csv>")
        sys.exit(1)

    task = sys.argv[1]
    dataset_name = sys.argv[2]
    data_name_csv = sys.argv[3]

    main(task, dataset_name, data_name_csv)
