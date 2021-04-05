# bms-molecular-translation
Kaggle competition

## Introduction

![image](https://user-images.githubusercontent.com/33998401/112753716-78adc600-8fe1-11eb-95e1-b1eb764caf6e.png)

## Installation

```bash
conda create -n bms python=3.7.10
conda activate bms
conda install pytorch=1.3.1 torchvision=0.4.2
pip install -r requirements.txt
```


## Running the Code

First preprocess the data:

```bash
python preprocess_data.py
```

Then you can train and test:

```bash
python train.py <experiment_name>
```

```bash
python test.py
```

## Submit only CSV file

```bash
kaggle competitions submit -c bms-molecular-translation -f submission.csv -m "Message"
```
