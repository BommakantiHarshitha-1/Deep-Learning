## Table of Contents
- [Introduction](#introduction)
- [HW1 MLP | Phoneme Recognition](#hw1-mlp--phoneme-recognition)
- [HW2 CNN | Face Recognition and Verification](#hw2-cnn--face-recognition-and-verification)
- [HW3 RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification](#hw3-rnn---forwardbackwordctc-beamsearch--connectionist-temporal-classification)

## Introduction
This repo contains course project of [11785 Deep Learning](http://deeplearning.cs.cmu.edu) at turnkey Learning. The projects starts off with MLPs and progresses into more complicated concepts like attention and seq2seq models. Each homework assignment consists of two parts. 
Part 1 is the Autolab software engineering component that involves engineering my own version of pytorch libraries, implementing important algorithms, and developing optimization methods from scratch. 
Part 2 is the Kaggle data science component which work on project on hot AI topics, like speech recognition, face recognition.


## HW1 MLP | Phoneme Recognition

- <b>HW1P1</b>
Implement simple MLP activations, loss, batch normalization. 

- <b>HW1P2</b>
Kaggle challenge: [Frame level classification of speech](https://www.kaggle.com/c/11-785-s20-hw1p2). <br>Using knowledge of feedforward neural networks and apply it to speech recognition task. The provided dataset consists of audio recordings (utterances) and their phoneme state (subphoneme) lables. The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text.
The job is to identify the phoneme state label for each frame in the test dataset. It is important to note that utterances are of variable length.

## HW2 CNN | Face Recognition and Verification
- <b>HW2P1</b>
Implement NumPy-based Convolutional Neural Networks libraries.

- <b>HW2P2</b>
Kaggle challebge: [Face Classification](https://www.kaggle.com/c/11-785-s20-hw2p2-classification) & [Verification](https://www.kaggle.com/c/11-785-s20-hw2p2-verification) using Convolutional Neural Networks.<br>Given an image of a person’s face, the task of classifying the ID of the face is known as face classification. The input to the system will be a face image and the system will have to predict the ID of the face. The ground truth will be present in the training data and the network will be doing an
N-way classification to get the prediction. The system is provided with a validation set for fine-tuning the model.
## HW3 RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification
- <b>HW3P1</b>
Implement RNNs and GRUs deep learning library like PyTorch.




