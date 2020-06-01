# Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values

This repository is the official implementation of [Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values](https://arxiv.org/abs/2030.12345). 


## Requirements

* To install Python 3, follow these [instructions](https://realpython.com/installing-python/). 
* To install Pip, follow these [instructions](https://pip.pypa.io/en/stable/installing/).
* To install Jupyter Lab/Notebook, follow these [instructions](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). To run Jupyter Lab/Notebook, follow these [instructions](https://jupyter.readthedocs.io/en/latest/running.html). 

* To install requirements:

```setup
pip install -r requirements.txt
```

* To obtain the ECG ViEW II dataset, please use this [form](http://ecgview.org/ECG_ViEW.asp). After recieving the unprocessed files, follow the [data processing](https://github.com/lujainibrahim/ecg-view-machine-learning#data-processing) steps below. 


## Data Processing

To process the ECG-ViEW II dataset as it is done in the paper (with robust scaling and SMOTE), run this [notebook]().

This notebook will produce two csv files, test.csv and train.csv, that you can then train/evaluate models with. If you would like to use the train and test files we used to obtain/reproduce our results, those can be found [here](https://drive.google.com/drive/folders/1-WcMjYm-jhuvE1vDpW76HkYW-xrOuPQ6?usp=sharing).

## Training

* To train the CNN model in the paper, run this [notebook]().
* To train the RNN model in the paper, run this [notebook]().
* To train the XGBoost model in the paper, run this [notebook]().

These notebooks will train the model and save it in a file that can be imported for evaluation later (described in the next section). 

## Evaluation

* To evaluate the CNN on the processed ECG-ViEW II data, run this [notebook]().
* To evaluate the RNN on the processed ECG-ViEW II data, run this [notebook]().
* To evaluate the XGBoost on the processed ECG-ViEW II data, run this [notebook]().

To reproduce the results in the paper, use the [pretrained models](https://github.com/lujainibrahim/ecg-view-machine-learning#pre-trained-models) and this [data](https://drive.google.com/drive/folders/1-WcMjYm-jhuvE1vDpW76HkYW-xrOuPQ6?usp=sharing). 

## Pre-trained Models

You can download pretrained models here:
With age and sex: 
- [CNN](https://github.com/lujainibrahim/ecg-view-machine-learning/blob/master/CNN/cnn_ecgview.h5) trained on ECG-ViEW II
- [RNN]() trained on ECG-ViEW II
- [XGBoost]() trained on ECG-ViEW II

Witout age and sex: 
- [CNN](https://github.com/lujainibrahim/ecg-view-machine-learning/blob/master/CNN/CNN_noagesex.ipynb) trained on ECG-ViEW II
- [RNN]() trained on ECG-ViEW II
- [XGBoost]() trained on ECG-ViEW II

## Results

Our models achieve the following performances:

### [AMI Prediction Using Processed ECG-ViEW II](http://ecgview.org/default.asp)

| Model      | Accuracy  | F1 Score | AUROC | Sensitivity | Specificity |
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- |
|CNN   |    89.9 %         |     89.0 %       |90.7 %|88.1 %|93.2%|
|RNN   |    84.6 %         |     82.2 %       |82.9 %|78.0 %|87.8 %|
|XGBoost   |    97.5 %         |     97.2 %       |96.6 %|93.2 %|99.3 %|

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

### Shapley Analysis 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
