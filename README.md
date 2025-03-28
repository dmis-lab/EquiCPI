# EquiCPI

A PyTorch implementation of:
**EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions**

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


# 1 **Generate a .pt file containing a 3D graph of a protein from a .pdb input file.**<br />
~~~
python generate_graph_for_protein.py #output_ESM #file_protein_name.csv #processed_dir #name_of_file.pt
~~~
**Re_ranking the complexes with Vina_docking score**<br />
~~~
python ./vina_score/vina_function_rerank_regu.py #prediction_output_diffdock #dataset.csv(with compound.sdf, protein.pdb)
~~~
**Run generate data for turning compounds and proteins into 3D graphs.** <br /> 
~~~
python generate_pt_dataset.py #machine_learning_task #data_name #data_csv_file.csv
~~~
**Run training the model.** <br /> 
~~~
bash run_class.sh
~~~

**Datasets** <br /> 
The related Datasets are as follows: <br /> 
[BindingDB curated from articles](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp): https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp <br /> 
[DUD-E Diverse](http://dude.docking.org/subsets/diverse): http://dude.docking.org/subsets/diverse <br /> 
[BindingDB_class](https://github.com/IBM/InterpretableDTIP): https://github.com/IBM/InterpretableDTIP <br /> 

