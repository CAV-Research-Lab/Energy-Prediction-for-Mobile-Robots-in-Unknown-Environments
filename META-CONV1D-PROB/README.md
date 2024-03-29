# Probabilistic Meta-Conv1D Driving Energy Prediction for Mobile Robots in Unstructured Terrains
This is the implementation of the paper: "Probabilistic Meta-Conv1D Driving Energy Prediction for Mobile Robots in Unstructured Terrains", Visca et al., 2022, DOI: [xxx](https://doi.org/10.36227/techrxiv.19538575.v1xx).

<img src="https://github.com/picchius94/META-CONV1D/blob/main/Images/transition_conv_1.gif" width="270"> <img src="https://github.com/picchius94/META-CONV1D/blob/main/Images/transition_conv_2.gif" width="270"> <img src="https://github.com/picchius94/META-CONV1D/blob/main/Images/transition_conv_3.gif" width="270">

## Dataset Collection
- Run `collect_dataset_handler.py` to collect multiple geometry-energy pairs datasets.
- Run `merge_dataset.py` to merge the datasets
- Run `generate_sum_indices.py` (needed for the models training)

### Note!
For visualisation, Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.

### Terrain Types and SCM Parameters
Deformable terrains are modelled using the Project Chrono [[1]](#1) implementation of the Soil Contact Model (SCM) [[2]](#2). The complete list of implemented terrain types and respective terramechanical parameters is given in the image below and at `terrain_list.py`.

<p align="center">
<img src="https://github.com/picchius94/META-CONV1D/blob/main/Images/SCM_Params.png" width="700">
</p>

### Geometry Generator
The geometry of the environments is generated using a Perline Noise algorithm described in [[3]](#3).
For more info check `terrain_generator.py`.

## Training
Our neural network models and the original model this work is based upon [[4]](#4) have already been trained and the model weights are available at `./Training/Exp00/log*`.

Modify `models.py` for creating new models.

Modify and run `train_*.py` for training new models.

Run `evaluate_models.py` for evaluating the models on the validation datasets.

## Path Planning
Run `path_planning_experiment_quantitative.py` to test the performance of different energy predictors, integrated into the path planner, in randomly generated unstructured environments.

All the entries of the dictionary `params` can be changed to modify map size, initial vehicle position, etc..


## Dependencies
The following dependencies are required:
- numpy
- matplotlib
- pandas
- pychrono
- tensorflow
- tensorflow-addons
- tensorflow_probability
- opensimplex




## References
<a id="1">[1]</a> 
A. Tasora, R. Serban, H. Mazhar, A. Pazouki, D. Melanz, J. Fleischmann, M. Taylor, H. Sugiyama, and D. Negrut, “Chrono: An open source multi-physics dynamics engine” in International Conference on High Performance Computing in Science and Engineering. Springer, 2015, pp. 19–49.

<a id="2">[2]</a>
F. Buse, R. Lichtenheldt, and R. Krenn, “Scm-a novel approach for soil deformation in a modular soil contact model for multibody simulation”, IMSD2016 e-Proceedings, 2016.

<a id="3">[3]</a>
Visca, M., Kuutti, S., Powell, R., Gao, Y., & Fallah, S. (2021). Deep Learning Traversability Estimator for Mobile Robots in Unstructured Environments. arXiv preprint [arXiv:2105.10937](https://arxiv.org/abs/2105.10937).

<a id="4">[4]</a>
Visca, M., Powell, R., Gao, Y., & Fallah, S. (2022). Meta-Conv1D Energy-Aware Path Planner for Mobile Robots in Unstructured Terrains. DOI: [10.36227/techrxiv.19538575.v1](https://doi.org/10.36227/techrxiv.19538575.v1).

