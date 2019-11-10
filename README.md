# Project Title

Mushroom classifier using Tensorflow 2.0 and Keras

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please install Tensorflow 2.0 (obviously) and Numpy.

## Dataset
The Mushroom dataset is collection of mushroom species useful to classify if mushroom are edible or poisonous, the dataset is public and can be downloaded from [here](https://www.mldata.io/dataset-details/mushroom/).

## Training

For training the model just run:
```
python train.py
```
And for visualizing the training you could use tensorboard by executing:
```
tensorboard --logdir=./logs/scalars
```
### Convergence


### Predicting

After the model finished training, take the saved model folder name, and modify the variable 'export_path_keras' in predict.py. Then run:

```
python predict.py
```
 

## Built With

* [Tensorflow 2.0](https://www.tensorflow.org/) - Machine learning library 
* [Numpy 1.16.5](https://numpy.org/) - Math library
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds


## Authors

* **Miguel Angel Campos** - *Software Engineer* - [Portfolio](http://mcampos.herokuapp.com/)
 
## Acknowledgments

* mldata.io
