Classifying Hate Speech on Twitter
------------------------------------------

Creating a model to recognize hate speech in tweets 

Table of Contents
#################

.. contents::

Description
###########

The goal of this project is to identify tweets containing sexism and racism as two
prominent aspects of hate speech. The dataset used is available on `Hugging
Face <https://huggingface.co/datasets/tweets_hate_speech_detection>`__ and includes 31,962 labled tweets.
To do so several classification models were implemented:

*  `Decision Tree <https://github.com/Aaron9812/Data_mining/blob/main/src/models/final_decision_tree.ipynb>`__
*  `Logistic Regression <https://github.com/Aaron9812/Data_mining/blob/main/src/models/regression.ipynb>`__
*  `Naive Bayes <https://github.com/Aaron9812/Data_mining/blob/main/src/models/Naive_Bayes.ipynb>`__
*  `Support Vector Machine <https://github.com/Aaron9812/Data_mining/blob/main/src/models/SVM-final.ipynb>`__
*  `Neural Network <https://github.com/Aaron9812/Data_mining/blob/main/src/models/NN_with_CV.ipynb>`__
*  `K-Nearest Neighbor <https://github.com/Aaron9812/Data_mining/blob/main/src/models/KNN_latest_v2.ipynb>`__

How to setup up
################

Install dependencies

How to use
##########

Use the Notebooks of the different models to assess. 

Best model
########
The best model's configuration and state are saved in the folder `src/models/ <https://github.com/Aaron9812/Data_mining/tree/main/src/models>`__ under names "nn.config" and "nn.model". 

Credits
#######

This project started in April 2022 at the University of Mannheim.
The team consists of:

* `Munir <https://github.com/MunirAbobaker/>`__
* `Jonas <https://github.com/jodi106/>`__
* `Aaron <https://github.com/Aaron9812/>`__
* `Mayte <https://github.com/misssophieexplores/>`__
* `Anna <https://github.com/annadymanus/>`__
