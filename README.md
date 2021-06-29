# 2nd CSEDM Data Challenge Sample Code

This repository contains sample code for the [2nd CSEDM Data Challenge](go.ncsu.edu/csedm-dc). It contains:
* [naive_model.ipynb](naive_model.ipynb): Basic sample code for Track 1 and 2 of the competition, using simple features.
* [code_feature_model.ipynb](code_feature_model.ipynb): A more advanced example of a predictive model using code features for Track 1.
* [progsnap2.py](progsnap2.py): A basic library for loading data from [ProgSnap2](bit.ly/ProgSnap2) datasets.
* [CodaLabPackages](CodaLabPackages): The CodaLab packages (minus data and solutions) used to create the competition.
* [preprocess.ipynb](preprocess.ipynb) and [compare_semesters.ipynb](compare_semesters.ipynb): The original code used for preprocessing and data labeling and splitting (provided for transparency). **Note**: This code will not run, since you do not have access to the full datasets used to generate them.

# Setup

To run the code in the .ipynb files, you will first need data! Links to the data can be found on the [2nd CSEDM Data Challenge](go.ncsu.edu/csedm-dc) description page.

You should create the following folder structure:

```
data/Release
    S19: This folder comes from the S19_Release.zip file you downloaded.
        Test
        Train
    F19
        Test: This folder comes from the F19_Release_Test.zip file you downloaded.
        Train: This folder comes from the F19_Release_Train.zip file you downloaded*.
```
 *Note: You may not yet have access to the F19/Train dataset yet. It will be released at the start of the Within-Semester phase of the competition.

 Each model is written as a jupyter notebook. To run them, make sure you have [Jupyter installed](https://jupyter.org/install).

 You may need to `pip install` some required packages:
 * pandas
 * numpy
 * matplotlib
 * sklearn
