# EquiCPI

A PyTorch implementation of:
**EquiCPI: A equivariant neural network for compound protein interaction prediction**

In this project, we proposed a model to fully use SE(3) group (Euclidian group) by using multiple e3nn neural networks to predict the binding affinity free energy. These networks leverage the principles of equivariance and invariance to process 3D structures, ensuring that the extracted information remains consistent regardless of transformations such as rotations, translations, and reflections.
Here we used the predicted 3D structure of compounds by adopting [Diffdock](https://github.com/gcorso/DiffDock) and the predicted 3D fold of protein sequence by using [ESMFold](https://github.com/facebookresearch/esm).



**
Generate a .pt file containing a 3D graph of a protein from a .pdb input file.**<br />
~~~
python generate_graph_for_protein.py  
~~~
**Run generate data for turning compounds and proteins into 3D graph.** <br /> 
~~~
python generate_pt_dataset.py 
~~~
Run training the model. <br /> 
~~~
CUDA_VISIBLE_DEVICES=4,5,6,7 train.py 
~~~
