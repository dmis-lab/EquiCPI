# EquiCPI

A PyTorch implementation of:
**[EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions](https://arxiv.org/abs/2504.04654)**

In this project, we proposed a model to fully use SE(3) group (Euclidian group) by using multiple e3nn neural networks to predict the binding affinity free energy. These networks leverage the principles of equivariance and invariance to process 3D structures, ensuring that the extracted information remains consistent regardless of transformations such as rotations, translations, and reflections.
Here we used the predicted 3D structure of compounds by adopting [Diffdock-L](https://github.com/gcorso/DiffDock) and the predicted 3D fold of protein sequence by using [ESMFold](https://github.com/facebookresearch/esm).

Traditional sequence-based models for compound-protein interaction (CPI) prediction often  rely on molecular fingerprints, descriptors, or graph representations. These approaches tend to overlook  the significant information of three-dimensional (3D) structures. To address this limitation, we developed  a novel model, EquiCPI, based on Euclidean neural networks (e3nns), which leverage the SE(3)  (Euclidean group) group to predict binding affinity. The model leverages principles of equivariance and  invariance, enabling it to extract 3D information while maintaining consistency across transformations such as rotations, translations, and reflections. We utilized predicted 3D structures from sequence data of compounds from state-of-the-art DiffDock-L and 3D protein folds from ESMFold to train and validate the proposed model.

![model_arch](https://github.com/user-attachments/assets/8ab233e5-d264-4bdf-b4a2-b3fa5a584c24)

Set up the environment:

In our experiment we use, Python 3.9 with PyTorch 2.1.2 + CUDA 11.8.

```bash
git clone https://github.com/dmis-lab/EquiCPI.git
conda env create -f environment.yml
```


 **Generate a .pt file containing a 3D graph of a protein from a .pdb input file.**<br />
To generate a .pt file containing a 3D graph representation of a protein, run:
~~~
python generate_graph_for_protein.py #output_ESM #file_protein_name.csv #processed_dir #name_of_file.pt
~~~
**Re_ranking the complexes with Vina_docking score**<br />
![image](https://github.com/user-attachments/assets/c32a437d-01c3-4e88-a83b-716d2150b5f8)

Our workflow starts by utilizing SMILES strings to represent the chemical structures of various compounds and amino acid sequences to define protein structures. These inputs are then processed through DiffDock-L and ESMFold, two advanced computational tools that predict the three-dimensional (3D) conformations of small molecules and proteins, respectively.

Once the 3D structures are generated, we proceed to docking simulations to assess potential interactions between the compounds and target proteins. Specifically, we employ AutoDock Vina, a widely used molecular docking software, to predict binding affinities and identify optimal docking poses. Among the generated poses, we systematically evaluate and select the most favorable binding conformation based on the AutoDock Vina score, ensuring that the predicted interaction is both energetically stable and biologically relevant.

This workflow enables us to efficiently model molecular interactions and screen for promising compound-protein binding events, facilitating drug discovery and molecular design processes.
To re-rank predicted complexes based on Vina docking scores, execute:
~~~
python ./vina_score/vina_function_rerank_regu.py #prediction_output_diffdock #dataset.csv(with compound.sdf, protein.pdb)
~~~
Ensure that dataset.csv contains compound.sdf and protein.pdb files.
**Run generate data for turning compounds and proteins into 3D graphs.** <br /> 
To convert compounds and proteins into 3D graph representations, run:
~~~
python generate_pt_dataset.py #machine_learning_task #data_name #data_csv_file.csv
~~~
**Run training the model.** <br /> 
To start model training, execute:
~~~
bash run_class.sh
~~~

**Datasets** <br /> 
The related Datasets are as follows: <br /> 
[BindingDB curated from articles](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp): https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp <br /> 
[DUD-E Diverse](http://dude.docking.org/subsets/diverse): http://dude.docking.org/subsets/diverse <br /> 
[BindingDB_class](https://github.com/IBM/InterpretableDTIP): https://github.com/IBM/InterpretableDTIP <br /> 

EquiDPI builds upon the source code and data from the following projects: <br /> 
[DiffDock-L](https://github.com/gcorso/DiffDock): Deep Confident Steps to New Pockets: Strategies for Docking Generalization <br /> 
[ESMFold](https://github.com/facebookresearch/esm): Evolutionary-scale prediction of atomic-level protein structure with a language model <br /> 
[AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina): AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings <br /> 

**License** <br /> 
This repository follows the license terms of the original EquiCPI project. Refer to LICENSE for details.
We thank all their contributors and maintainers!

**Citing this work**
If you use the code or data associated with this package or otherwise find this work useful, please cite:
~~~
@misc{nguyen2025equicpise3equivariantgeometricdeep,
      title={EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions}, 
      author={Ngoc-Quang Nguyen},
      year={2025},
      eprint={2504.04654},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.04654}, 
}
~~~
