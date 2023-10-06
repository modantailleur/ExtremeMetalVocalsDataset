# EMVD dataset: a dataset of extreme vocal distortion techniques used in heavy metal

This repo contains code for the paper: **EMVD dataset: a dataset of extreme vocal distortion techniques used in heavy metal** [1]. It gives instruction to reproduce the baseline experiments shown in the paper (sections 2 and 3), as well as example codes for using the [EMVD dataset](https://zenodo.org/record/8406322) metadata correctly (section 4). 

## 1 - Setup Instructions

The codebase is developed with Python 3.9.15. To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

Run the following command to download the [EMVD dataset](https://zenodo.org/record/8406322) from zenodo:

```
python3 download_dataset.py
```

## 2 - Paper results replication

First, create a Mel dataset stored in a h5 file using this command:

```
python3 create_dataset.py -dataset kfold
```

Run the following commands to train the model, store the predictions, and calculate the metrics that are used in the paper:

```
python3 experiment.py -exp t_multiclass -step train
```
```
python3 experiment.py -exp t_multiclass -step eval
```
```
python3 experiment.py -exp t_multiclass -step metric
```

You can run the previous lines with ''' -exp t_binary ''' to train on binary classification instead of multi-class classification.

After running the metric step, results are plotted in the terminal, and the confusion matrix for the multi-class classification is saved in the 'results' folder.

Note that if you want to do the experiment again using another seed for the kfold cross-validation or another number of folds, you can run the following command that will save the split as 'new_split_kfolds.csv' in the main directory:

```
python3 create_split_k_fold.py -seed 0 -n_splits 4
```

You'll then just have to replace the old file with the new one in the folder 'EMVD'.

## 3 - Paper figures replication

To replicate the figure that shows the total duration and the number of singers for each technique in the dataset, run:

```
python3 plot_dataset_analysis.py
```

To replicate the spectrograms shown in the paper, run:

```
python3 plot_spectrograms.py
```

## 4 - Routine to merge metadata files

There are two different metadata files: 'metadata_files.csv' that has metadata for each audio file recorded, and 'metadata_singers.csv' that has metadata for each singer. The two files can be joined on the column 'singer_id'. An example of how to join the two metadata files with pandas can be found in the jupyter notebook 'load_metadata.ipynb'.

## REFERENCES

[1] link to our paper