# AMLS Assignment 2023 - 2024

The goal of this project is to classify data from two medical images dataset leveraging advanced machine learning and deep learning models. Task A involves binary classification of the PneumoniaMNIST and Task B involves multiclassification of the PathMNIST dataset.

This project folder `AMLS_23-24_SN20099306` you will find:
- A folder `A`: this folder contains contains the model file codes for Task A:
  - `SVM_model.py` which contains the SVM model Class and corresponding functions to code and execute the SVM models
  - `CNN_model.py` which contains the CNN model Class and corresponding functions to code and execute the CNN model
  - `taskA_pretained_CNN.h5` The pretained deep learning model 
- A folder `B`: this folder contains contains the model file codes for Task B:
  - `CNN_model_B.py` which contains the CNN model Class and corresponding functions to code and execute the CNN model
  - `taskB_pretained_CNN_2.h5` The pretained deep learning model for this task
- `Datasets`: This folder is empty and the datasets files PneumoniaMNIST.npz and PathMNIST.npz should go here.
- `main.py`: this file runs the MENU to select which tasks to run
- `README.md`: This file describing the assignment and guidance to run the code

# Packages required 
- `numpy`: For calculations and numerical functions
- `matplotlib`: For data visualisation
- `seaborn`: For advanced data visualisation (eg: confusion matrix)
- `sklearn`: For machine learning models (SVM), data preprocessing tasks and evaluation metrics of our models
- `tensorflow` and `keras`: For designing and training deep learning models

# How to Run 
The entire assignment can be tested from main.py but make sure to do the following:
1. Check the packages required are installed
2. Download the PneumoniaMNIST.npz and PathMNIST.npz from the website https://doi.org/10.5281/zenodo.6496656 and load them in the folder `Datasets`
3. Run `main.py` to initiate the program. The following menu will appear:
   MACHINE LEARNING ASSIGNMENT:
   "WELCOME"
    Please Select an option
    0. Visualise Datasets
    1. Run Task A - Train SVM without PCA
    2. Run Task A - Train SVM with PCA
    3. Run Task A - Train CNN
    4. Run Task A - Use Pre-trained CNN (Recommended)
    5. Run Task B - Train CNN (Recommended)
    6. Run Task B - Use Pre-trained CNN (Recommended)
    7. Exit

   Input a number between 0-7 to run the corresponding Task and associated model.

