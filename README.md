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

- `data/` contains the source code as well as the datasets used in this study.
- `utils/` contains helper functions for the project.
- `src/` contains the main source code for the project for both vision and time-series experiments.
- `config_environment/` contains configuration files for experiments.
- `experiments/` contains experiment scripts and bash scripts.
- `pics/` contains images used in documentation.

### Weights and Biases
This project uses [W&B](https://wandb.ai) for experiment tracking and model management.

### Running Experiments
Experiment configurations are in the `config_environment` folder. The config files are configured to correpond to the default settings used in the paper. Please read the paper for more information on the hyperparameters.

### Vision Experiments

#### Data Setup
- For ImageNet experiments: Download the validation set and place in `data/ImageNet/validation_set/`
- For quick testing: Use provided custom images in the repository

#### Available Methods
All implementations are in `src/vision/`:
- PixelMask (`pixel_RDE.py`)
- WaveletX (`waveletX.py`)
- ShearletX (`shearletX.py`)
- Gradient-based methods (`saliency_methods.py`): Integrated Gradients, SmoothGrad, GradCam

#### Running Experiments
1. Configuration files are in `config_environment/vision/<METHOD>/`
   - Note: StartGrad initialization is labeled as 'saliency' in configs
   - Example: ShearletX with StartGrad uses `hparams_shearletX_saliency.yaml`

2. Example command:
```bash
cd ./experiments/vision
python3 main_vision.py --method shearletX,shearletX_saliency,shearletX_uniform --folder Custom --input kobe.jpg --iterations 5 --pretrained_model resnet18 --seed 123
```
The command above stores a visual comparison between the original image and the masked image to: `figures/shearletX/<seed>/<model>/Comparison_explainers_<method>_<seed>.png`

For detailed parameter descriptions, see `main_vision.py`.

3. Results & Figures
The results folder contains important results file stored as csv files which can be used to generate plots of the paper.
Important: Some of the csv files in the corresponding resnet folders are in a zip format, so one needs to unzip these first.

To generate the main figures, please have a look at the bash scripts of the `experiments/vision` folder.

### Time Series
#### Data Setup
There are two datasets implemented, state and switch-feature dataset. The corresponding scripts to generate the 
datasets can be found in `utils/time_series/`.

#### Available Methods
##### Explainer
In this repo, we use the ExtremalMask method introduced [in this paper](https://openreview.net/forum?id=WpeZu6WzTB) 
as a time-series mask-based explainer. The implementation can be found in `src/time_series/timeseries_mask_explainer.py`.

##### Classifiers
There are two classifiers implemented, a LSTM and a GRU deep learning architecture. The corresponding code can be found
`src/time_series/XAI_classifier.py`.

#### Running Experiments
1. Configuration files are in `config_environment/time_series/<OBJECTIVE_FORMULATION>/`
   - Note: StartGrad initialization is labeled as 'gradient' in configs
   - Example: `hparams_extrema_gradient.yaml`

2. Example command:
```bash
cd ./experiments/time_series
python3 main_time_series.py --iterations 500 --epochs 50 --mode preservation_game --model_type GRU --dataset state --plot_average
```

If the dataset has not yet been generated for the seed, it will first create it before training the time-series classifier
and fitting the mask-based XAI method.

3. Results
- Performance results are then saved to `results/time_series/extremal/<seed>/<model_type>/<mode>/<fold_number>`
- Average performance plots are then generated and saved to `figures/time_series/<dataset>/extremal/<seed_number>/<model_type>/<mode>`

For detailed parameter descriptions, see `main_time_series.py`.

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
