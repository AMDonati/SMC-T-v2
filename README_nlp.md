## Instructions for the "nlp22" branch

### Requirements
The libraries needed to run the code are provided in the file requirements.txt.
* To run all the scripts from the origin repo (SMC-T-v2), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

### downloading the cache files for GPT2

`python src/scripts/save_datasets_models.py`

### Preprocess the datasets 

#### Dummy NLP dataset 

`python src/preprocessing/create_dummy_dataset_nlp.py`

#### ROC Stories Dataset 

`python src/preprocessing/preprocess_ROC.py`


