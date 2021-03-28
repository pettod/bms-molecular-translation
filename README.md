# bms-molecular-translation
Kaggle competition

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
python train.py
```

```bash
python test.py
```
