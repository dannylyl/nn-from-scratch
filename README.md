# Neural Network Using only NumPy

This repository is a personal project I challenged myself to do to implement a
simple Multilayer Perceptron / Fully Connected Feedforward Neural Network / Dense
Network (and other similar terms) using **only NumPy**. 

#### The Neural Network implemented has the following architecture:
1. 4 Fully Connected Hidden Layers
2. Residual connection between the 1st Hidden Layer and the 3rd Hidden Layer

#### Other details about the Neural Network:
I have written some utility functions and included them in the `nn_utils.py` script.
Some utilities include:
1. Dropout Implementation
2. Momentum Implementation
3. ReLU, Sigmoid, and tanh Activation Functions
4. Binary Cross Entropy Loss
5. Derivatives of the activations functions for backpropagation

### Folder Structure for Project:
```
NN-FROM-SCRATCH
├── conf
│   └── nn_config.yaml
├── data
│   └── diabetes.csv
├── src
│   ├── test
│   │   ├── test_nn_clf.py
│   │   └── test_nn_utils.py
│   ├── base.py
│   ├── nn_clf.py
│   └── nn_utils.py
├── neuralnetwork_test.ipynb
├── README.md
└── requirements.txt
```
* `nn_config.yaml` - Configuration YAML file which is used to instantiate the
  `NeuralNetworkClassifier` class
* `src/base.py` - Added the abstract base class `Layer`, which is inherited by the
  `FullyConnected` class
* `src/nn_clf.py` - Python script where I wrote the `NeuralNetworkClassifier` class. The
  class implemented is meant to meet the specifications in page 3 of the assessment PDF
  file, and makes use of all the given hyper-parameters in the abstract base class
  `BaseMLP`
* `src/nn_utils.py` - Script containing the `FullyConnected` class which is instantiated
  for every hidden layer in the `NeuralNetworkClassifier`. Also contains functions for
  dropout implementation, activation functions, their derivatives, and binary cross
  entropy loss.
* `src/test` - Pytests for the classes and functions written. You can run the pytests by
  going into the `Section A` directory and running `pytest src` with pytest installed.
  

## Jupyter Notebook Going Through the Usage of the Neural Network
A notebook  `neuralnetwork_test.ipynb` has been written where I go through a quick
Exploratory Data Analysis, Data Preprocessing and Model Training as well as evaluation
and insights on the model's performance. 

This personal project has been a fun undertaking to challenge my own understanding and
deepen my knowledge in implementing more features like **Dropout**, **Momentum** and
**Residual Connections**.

Thanks for taking the time to look at my work!
