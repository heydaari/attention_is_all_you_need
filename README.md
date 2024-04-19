# Attention Is All You Need

## Implementation of the paper "Attention Is All You Need" with Tensorflow-Keras

![alt text](https://th.bing.com/th/id/R.a21eb096373b5a85bc1bed0a34d049c3?rik=covqvwKF9f95lw&pid=ImgRaw&r=0)



## Introduction

This repository contains the implementation of the paper "Attention Is All You Need" using TensorFlow and Keras. The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output, is implemented here.

## Dataset

The model was trained on an English to Spanish dataset. The dataset comprises of numerous English sentences and their Spanish translations, which were used to train the model to understand the context and semantics of the sentences and translate them accurately.

## Model Architecture and Training

The Transformer model architecture was used for this task. The model was trained for 30 epochs. The RMSProp optimizer was used for training the model. RMSProp is an optimizer that's reliable and fast, and it's often a good choice for recurrent neural networks.

After training for 30 epochs, the model achieved an accuracy of 93.54% on the training set. This indicates that the model was able to correctly translate English sentences to Spanish with a high degree of accuracy most of the time.

The model also achieved a validation accuracy of 88.90%. This is the accuracy of the model on a validation set that was not used during training. This high validation accuracy indicates that the model generalizes well to new, unseen data.

## Training

The Model is trained on google colab T4 GPU 

## Repository Contents

The repository contains two main files:

1. `transformers.py`: This is the main Python script that contains the implementation of the Transformer model. It includes the model architecture, the training loop, and code for evaluating the model.

2. `transformers.ipynb`: This is a Jupyter notebook that contains the same code as `transformers.py`. The notebook was used for interactive development and training of the model on Google Colab. After training, the notebook was downloaded from Colab and added to this repository.

## Usage

To use the model for translating English sentences to Spanish, you can clone this repository and run the `transformers.py` script. Alternatively, you can open the `transformers.ipynb` notebook in Jupyter and run the cells interactively.

## Contributions

Contributions to this project are welcome. If you have suggestions for improving the model or the code, please open an issue to discuss your ideas. If you wish to contribute code, please open a pull request.

