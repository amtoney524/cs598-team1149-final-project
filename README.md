# cs598-team1149-final-project
This repository documents the code used to train and evaluate several pretrained CNNs for classification of COVID-19 from publically available chest X-ray images.
The CNNs supported are ResNet18, ResNet50, SqueezeNet 1.1, DenseNet-121, Inception V3, COVID-Net, and COVID-Net combined with LSTM.

It builds on previous work from the [Deep-COVID](https://github.com/shervinmin/DeepCovid) and [COVID-Net](https://github.com/iliasprc/COVIDNet) papers.



## COVID-19 Radiography Database
The dataset used is a subset of the COVID-19 Radiography database, which may be downloaded from [Kaggle](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

The number of chest X-ray images per category is as follows:
| Split           | COVID                         | Non-COVID  |
| :-------------- |:------------------------------| :----------|
| Train           | 1440 (2880 with augmentation) | 4800       |
| Validation      | 580                           | 1199       |
| Test            | 1200                          | 3999       |



## Training the Models
To train the desired model, edit the first block in train.ipynb to specify which of the supported models you wish to train.

For example, if you wish to train DenseNet-121, you would update the value of model_name to "DenseNet121"
```
model_name = "DenseNet121"
```
