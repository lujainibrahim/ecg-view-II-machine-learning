# Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values

This repository is the official implementation of [Explainable Prediction of Acute Myocardial Infarction using Machine Learning and Shapley Values](https://arxiv.org/abs/2030.12345). 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:
With age and sex: 
- [CNN](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II
- [RNN](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II
- [XGBoost](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II

Witout age and sex: 
- [CNN](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II
- [RNN](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II
- [XGBoost](https://drive.google.com/mymodel.pth) trained on ECG-ViEW II

The ECG-ViEW II data was processed with robust scaling and SMOTE. Follow this [data processing notebook]() to obtain the processed data from the original csv files.

## Results

Our model achieves the following performance on :

### [AMI Prediction Using Processed ECG-ViEW II](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model      | Accuracy  | F1 Score | AUROC | Sensitivity | Specificity |
| ------------------ |---------------- | -------------- | -------------- | -------------- | -------------- |
|CNN   |     %         |      %       |%|%|%|
|RNN   |     %         |      %       |%|%|%|
|XGBoost   |     %         |      %       |%|%|%|

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

### Shapley Analysis 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
