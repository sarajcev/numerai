# Numerai
Machine Learning Hedge Fond

![Numerai](https://numer.ai/homepage/img/Numerai-Logo-Side-Black.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

Numerai is a new kind of hedge fond, which synthesizes many different and uncorrelated models with many different characteristics, provided by a large group of anonymous people from all over the world. It is free to participate. Data offered weekly is encripted in order to protect the proprietary financial information of the hedge fond. The challenge is posed as a classification problem, where users train their models on the train dataset and upload their predictions for the test dataset. More info can be found at www.numeria.ai

This Jupyter Notebook features a "sandbox" for exploring different classification strategies on the numerai data. It employs scikit-learn Python library and comprises three stages:

1. feature extraction and feature engineering; following methods are explored:
    - princilpal component analysis (PCA)
    - linear discriminant analysis (LDA)
    - selecting best features (KBest)
    - t-SNE method for feature engineering
    - feature interactions using PolynomialFeatures

2. training multiple individual classifiers; these include:
    - Keras neural networks
    - Logistic regression
    - Support vector machine
    - Gaussian naive Bayes
    - Random forrest classifier
    - Extra trees classifier
    - Gradient boost classifier
    - AdaBoost classifier
    - Bagging classifier
    - Stochastic gradient descent
    - K-Nearest neighbors

Grid search and cross validation are used with some of the classifiers in order to fine tune their hyperparameters. Pipelines are used for automating tasks when needed. Keras neural network can be easily reconfigured using different number of hidden layers and/or neurons per layer, along with different training algorithms.

3. aggregating individual classifiers using ensambling by soft voting, blending and stacking; following methods are explored:
    - blending with logistic regression
    - blending with linear regression
    - blending with Extremly randomised trees
    - blending with Keras neural network classifier
    - stacking with TensorFlow DNN classifier
    - stacking with Extremly randomised trees
    - stacking with Keras neural network classifier with Merged branches
    - simple averageing of classifiers using different weights

Keras and TensorFlow neural networks can be easily reconfigured using different number of hidden layers and/or neurons per layer, along with different training algorithms.

Predictions on test dataset are carried out using ensambles of indivual classifiers (blending, stacking, averageing).
