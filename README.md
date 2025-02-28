# Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods

This repository contains the code for the paper: 
[Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods](https://openreview.net/forum?id=Iht4NNVqk0&noteId=Iht4NNVqk0)

**Authors:** [Buelent Uendes](https://buelentuendes.github.io/), [Shujian Yu](https://sjyucnel.github.io/), and Mark Hoogendoorn

## Abstract
Mask-based explanation methods offer a powerful framework for interpreting deep learning model predictions across diverse data modalities, such as images and time series, in which the central idea is to identify an instance-dependent mask that minimizes the performance drop from the resulting masked input. Different objectives for learning such masks have been proposed, all of which, in our view, can be unified under an information-theoretic framework that balances performance degradation of the masked input with the complexity of the resulting masked representation. Typically, these methods initialize the masks either uniformly or as all-ones.
In this paper, we argue that an effective mask initialization strategy is as important as the development of novel learning objectives, particularly in light of the significant computational costs associated with existing mask-based explanation methods. To this end, we introduce a new gradient-based initialization technique called StartGrad, which is the first initialization method specifically designed for mask-based post-hoc explainability methods. Compared to commonly used strategies, StartGrad is provably superior at initialization in striking the aforementioned trade-off. Despite its simplicity, our experiments demonstrate that StartGrad enhances the optimization process of various state-of-the-art mask-explanation methods by reaching target metrics faster and, in some cases, boosting their overall performance.

<p align="center">
  <img src="pics/startgrad_pseudocode.png" width="450">
</p>

## Setup and Usage

### Environment Setup & Installation of requirements
1. Please first create and activate a virtualenv via:
```bash
virtualenv <NAME_OF_YOUR_VIRTUALENV> 
source <NAME_OF_YOUR_VIRTUALENV>/bin/activate
```
2. Install the pytorch wavelets 
```bash
cd ./src
git clone https://github.com/fbcotter/pytorch_wavelets
cd pytorch_wavelets
pip install .
```
3. Now you can run and install the required packages via:
```bash
cd ../../
pip3 install -r requirements.txt
pip3 install -e .
```
4. Lastly, you need to install pyShearLab via:
```bash
pip3 install https://github.com/stefanloock/pyshearlab/archive/master.zip
```

#### Tested environment:
The project was run and tested on both MacOS Sonoma and Linux Ubuntu with Python 3.9 installed.

### Repository structure
The repository is structured as follows. 

- `data/` contains the source code as well as the datasets used in this study
- `utils/` contains helper functions for the project.
- `src/` contains the main source code for the project for both vision and time-series experiments.
- `config_environment/` contains configuration files for experiments
- `experiments/` contains experiment scripts and notebooks
- `pics/` contains images used in documentation

### Weights and Biases
This project uses [W&B](https://wandb.ai) for experiment tracking and model management.

### Running Experiments
Experiment configurations are in the `config_environment` folder. The config files are configured to correpond to the default settings used in the paper. Please read the paper for more information on the hyperparameters.

#### Vision
To work with ImageNet data, one needs to download the ImageNet validation set and put them in 
`data/ImageNet/validation_set/<INSERT_HERE_THE_FOLDERS>`. This repository comes with custom images that are there to run the methods
quickly on some exemplary images. 

There are three main baseline mask-based XAI methods implemented that work on the vision domain which
can be found in the `src/vision` folder.
- `pixel_RDE.py` -> PixelMask 
- `cartoonX.py` -> WaveletX 
- `shearletX.py` -> ShearletX 
- `saliency_methods.py` -> Additional gradient-based methods such Integrated Gradients, SmoothGrad or GradCam

The corresponding config files for each of the methods are in the `config_environment/vision/<MASK_NETHOD>` folder.

**Important** The startgrad initialization is named 'saliency' throughout. For instance, the config file for the ShearletX model (StartGrad)
can be found in the '`hparams_shearletX_saliency.yaml` file.

To run the ShearletX model with all three initialization on the kobe.jpg custom image across
25 iterations on a pretrained ResNet18 model, you can run the command below:
```bash
cd ./experiments/vision
python3 python3 main_vision.py --method shearletX,shearletX_saliency,shearletX_uniform --folder Custom --input kobe.jpg --iterations 5 --pretrained_model resnet18 --seed 123

```
The command above will plot the original image alongside each of the masked compressed images. The figure is 
automatically saved under `figures/shearletX/123/resnet18/Comparison_explainers_<LAST_MASK_METHOD>_<SEED_NUMBER>.png`

In our example above, the figure with the name `Comparison_explainers_shearletX_uniform_5_123.png`.

Specifications on the parameters specified via the argsparser can be found in the 
corresponding `main_vision.py` file. 

**ToDo**:
Add how to get the visualizations obtained in the paper.

#### Time-Series

Run experiment with state dataset:
```bash
cd ./experiments/time_series
python run_experiment.py experiment=mnist/dvp_vae
```

## Citation
If you found this work useful in your research, please consider citing:
```bibtex
@article{buendes2025startgrad,
  title={Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods},
  author={Uendes, Buelent  and Yu, Shujian, and Hoogendoorn, Mark},
  journal={Proceedings of the 13th International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgements
This work is funded by [Stress in Action]( www.stress-in-action.nl). The research project [Stress in Action]( www.stress-in-action.nl) is financially supported by the Dutch Research Council and the Dutch Ministry of Education, Culture and Science (NWO gravitation grant number 024.005.010).

Part of the code relies on repository [ShearletX](https://github.com/skmda37/ShearletX) for the associated paper [Explaining Image Classifiers with Multiscale Directional Image Representation](https://arxiv.org/pdf/2211.12857) for the vision experiments. For the time series experiments, we also relied on the repository [time interpret](https://github.com/josephenguehard/time_interpret) for the implementation of the ExtremalMask model and generation of the synthetic datasets.
