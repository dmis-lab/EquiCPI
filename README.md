# EquiCPI

A PyTorch implementation of:
**[EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions](https://arxiv.org/abs/2504.04654)**

## 🔬 Overview
EquiCPI is a novel model designed to leverage the full SE(3) Euclidean group by incorporating multiple **e3nn neural networks** to predict **binding affinity free energy**. These networks apply principles of **equivariance and invariance** to process 3D molecular structures, ensuring robustness against transformations such as **rotations, translations, and reflections**. Here we used the predicted 3D structure of compounds by adopting [Diffdock-L](https://github.com/gcorso/DiffDock) and the predicted 3D fold of protein sequence by using [ESMFold](https://github.com/facebookresearch/esm). 
Traditional sequence-based models for compound-protein interaction (CPI) prediction often  rely on molecular fingerprints, descriptors, or graph representations. These approaches tend to overlook  the significant information of three-dimensional (3D) structures. To address this limitation, we developed  a novel model, EquiCPI, based on Euclidean neural networks (e3nns), which leverage the SE(3)  (Euclidean group) group to predict binding affinity. The model leverages principles of equivariance and  invariance, enabling it to extract 3D information while maintaining consistency across transformations such as rotations, translations, and reflections. We utilized predicted 3D structures from sequence data of compounds from state-of-the-art DiffDock-L and 3D protein folds from ESMFold to train and validate the proposed model.

To achieve this, we use:
- **[DiffDock-L](https://github.com/gcorso/DiffDock)** for predicting the **3D structures of compounds**.
- **[ESMFold](https://github.com/facebookresearch/esm)** for predicting **protein 3D folds** from sequences.

### 🚀 Key Advantages Over Traditional Models
Traditional **sequence-based** CPI prediction models rely on molecular fingerprints, descriptors, or graphs, often **overlooking** critical 3D structural information. **EquiCPI**, built on **Euclidean neural networks (e3nn)**, fully utilizes the **SE(3) group** to process 3D structures, providing more accurate and structure-aware CPI predictions.

![Model Architecture](https://github.com/user-attachments/assets/8ab233e5-d264-4bdf-b4a2-b3fa5a584c24)

---

## ⚙️ Setup & Installation
### Prerequisites
- Python 3.9
- PyTorch 2.1.2 + CUDA 11.8

### Installation
```bash
# Clone the repository
git clone https://github.com/dmis-lab/EquiCPI.git

# Create and activate the environment
conda env create -f environment.yml
```

---

## 🔄 Data Preprocessing & Graph Generation
### 🏗️ Generating a 3D Graph for a Protein from a .pdb File
To convert a **.pdb file** into a **.pt file** containing a 3D protein graph:
```bash
python generate_graph_for_protein.py #output_ESM #file_protein_name.csv #processed_dir #name_of_file.pt
```

### 📊 Re-ranking Complexes with Vina Docking Score
Our workflow starts with:
1. **SMILES strings** representing compounds.
2. **Amino acid sequences** defining proteins.
3. **DiffDock-L & ESMFold** generating **3D structures** of compounds and proteins.
4. **AutoDock Vina** predicting **binding affinities** and identifying optimal docking poses.

To re-rank **predicted complexes** based on **Vina docking scores**, run:
```bash
python ./vina_score/vina_function_rerank_regu.py #prediction_output_diffdock #dataset.csv(with compound.sdf, protein.pdb)
```
**Note:** `dataset.csv` must contain `compound.sdf` and `protein.pdb` files.

### 🏗️ Converting Compounds & Proteins into 3D Graph Representations
```bash
python generate_pt_dataset.py #machine_learning_task #data_name #data_csv_file.csv
```

---

## 🎯 Training & Evaluation
### 🏋️ Training the Model
```bash
bash run_class.sh
```

---

## 📂 Datasets
We utilize several datasets for training and evaluation:
- **[BindingDB curated](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)**
- **[DUD-E Diverse](http://dude.docking.org/subsets/diverse)**
- **[BindingDB_class](https://github.com/IBM/InterpretableDTIP)**

---

## 📌 Dependencies & Related Work
EquiCPI builds upon the source code and data from the following projects:
- **[DiffDock-L](https://github.com/gcorso/DiffDock)** – Deep confident steps to new pockets.
- **[ESMFold](https://github.com/facebookresearch/esm)** – Atomic-level protein structure prediction.
- **[AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina)** – Molecular docking software.

We sincerely appreciate all contributors and maintainers for their efforts! 🙌

---

## 📜 License
This repository follows the license terms of the **EquiCPI project**. ## License
[MIT](https://choosealicense.com/licenses/mit/).

---
## ✅ TO DO

| Task | Status       | Notes                                             |
|------|--------------|---------------------------------------------------|
| Improve CLI usability                  | ✅ Done | Add YAML/argparse defaults                      |
| Add structured W&B logging             | ✅ Done        | Already integrated                              |
| Clean folder structure                 | ✅ Done | Organize into `scripts/`, `models/`, `data/`    |
| Add Jupyter notebooks                  | 🔧 In Progress | Demo for training, testing, visualizing graphs  |
| Detailed API documentation             | 🔧 In Progress | Add docstrings + auto doc                       |
| How to train your dataset              | 🔧 In Progress | supporting custom datasets                      |

---
## 📖 Citation
If you use this code or dataset in your research, please cite:
```bibtex
@misc{nguyen2025equicpise3equivariantgeometricdeep,
      title={EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions},
      author={Ngoc-Quang Nguyen},
      year={2025},
      eprint={2504.04654},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04654},
}
```
