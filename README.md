# EquiCPI

A PyTorch implementation of:
**EquiCPI: A equivariant neural network for compound protein interaction prediction**

In this project, we proposed a model to fully use SE(3) group (Euclidian group) by using multiple e3nn neural networks to predict the binding affinity free energy. These networks leverage the principles of equivariance and invariance to process 3D structures, ensuring that the extracted information remains consistent regardless of transformations such as rotations, translations, and reflections.
Here we used the predicted 3D structure of compounds by adopting [Diffdock](https://github.com/gcorso/DiffDock) and the predicted 3D fold of protein sequence by using [ESMFold](https://github.com/facebookresearch/esm).


![model_arch](https://github.com/user-attachments/assets/8ab233e5-d264-4bdf-b4a2-b3fa5a584c24)


**Generate a .pt file containing a 3D graph of a protein from a .pdb input file.**<br />
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

