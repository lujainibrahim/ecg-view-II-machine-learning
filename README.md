# Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values

This repository is the official implementation of [Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values](https://arxiv.org/abs/2030.12345). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Data Processing

To process the ECG-ViEW dataset as it is done in the paper, run this command:

```train
python processing.py --input-data <path_to_data> --alpha 10 --beta 20
```


## Training

To train the CNN model in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

To train the RNN model in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

To train the XGBoost model in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate the CNN on the processed ECG-ViEW II data, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
To evaluate the RNN on the processed ECG-ViEW II data, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```
To evaluate the XGBoost on the processed ECG-ViEW II data, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

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

The ECG-ViEW II data was processed with robust scaling and SMOTE. Follow this [data processing notebook]() to obtain the processed data from the original csv files.

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
