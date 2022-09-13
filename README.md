# Heart Disease Classification
Created a binary classification model with **87% accuracy** on heart disease data to help hospitals determine if a given patient has heart disease. 

# Data & Problem
* **Problem**: The goal of this problem is to explore **binary classification** (a sample can only be one of two things) on heart disease data.
This is because we're going to be using a number of different **features** about a person to predict whether they have heart disease or not.
In a statement,
> Given clinical parameters about a patient, can we predict whether or not they have heart disease?
* **Data Acquisition**: [UCI Machine Learning Repository: Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
* **Sucess Metrics**: If we can reach **90% accuracy** at predicting whether or not a patient has heart disease during the proof of concept, we'll pursuse this project.

# Code and Resources Used
* **Python Version**: 3.8
* **Environment**: Miniconda, Jupyter Notebook
* **Packages**: Pandas, Scikit-Learn, NumPy, Matplotlib, Seaborn, Joblib

# EDA & Feature Engineering
After loading the data and inspecting the dataset's features (patient attributes) and target variable (heart disease), I needed to clean the data and visualize it to better understand the relationship between the features, and the target. I did the following steps with the dataset:
* Found the number of unique values in the dataset through creating a dictionary
* Determined a normalized count of postive/negative heart disease patients (target variable, where 0 implies no heart disease and 1 implies heart disease) to conclude that the target column being dealt with is **balanced**.
* Seperated continuous and categorical columns to acquire summary statistics on the continuous features. 
* Checked for total number of missing values in each column and found that there are **no missing values** in this dataset.
* Checked for duplicate values in the dataset and found **1 duplicate value**, which was removed by keeping only the **first** duplicate value.
* Visualized heart disease patients in the dataset based on gender and found the following:
  - There are a total of 96 female patients in this dataset, and 72 of them are positive for heart disease. There is approximately a 75% chance that a female patient that participates in this dataset has heart disease.
  - As for males, there are 206 male patients in this datasst, and 114 of them have heart disease, which implies that there is less than a 50% chance that a male patient partaking in this study will have heart disease.
  - There is roughly a 54% chance that a patient partaking in this dataset has heart disease based on no other parameters. This will be used as a **basic baseline** to surpass with machine learning modelling. 
* Visualized Age vs. Max Heart Rate by Heart Disease patients and found that younger individuals tend to have a higher max heart rate (more dots on the left side of the graph), and the older someone is, the lower their max heart rate is. There also seems to be less heart disease patients in the older age range (around 55+). This is likely because a large portion of the data is collected on older patients (50+ years of age). 

![ageVSheartrateGraph](https://user-images.githubusercontent.com/46492654/161370926-abf9f1da-94e1-4f65-88fd-826acbf7a1ab.png)
![densityplots](https://user-images.githubusercontent.com/46492654/161370969-8dc079f1-358a-422c-bc58-9d433c032791.png)
* Created a heatmap of all the features as well as a correlation matrix for the continuous features, and I found that none of the features are strongly correlated with each other in the negative or positive direction. All features have correlation coefficients between -0.6 or 0.6 (not inclusive) indicating that none of the features or targets have strong correlations; rather relatively weak correlations between each other. 
![heatmap](https://user-images.githubusercontent.com/46492654/161370980-e2dd716f-ed4a-4063-802b-0f285c29cc7f.png)

# Model Building
First, I split the data into X (features) and y (target). Then, I split the X and y data into training and test sets with test set size of 20%. Based on the [Scikit-Learn Algorithm Selection Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html), I trained 7 classification algorithms and evaluated them using **accuracy** as the primary evaluation metric. I chose accuracy for checking baseline models because it is to interpret how well the models prediction heart disease (0 - Negative or 1 - Positive). 

The 7 models I selected for the classification:
1. Logistic Regression 
2. K-Nearest Neighbors
3. Support Vector Classification 
4. Decision Tree Classifier 
5. RandomForest
6. AdaBoost Classifier
7. Gradient Boost Classifier

# Model Performance
From the baseline model, the Random Forest Classifier provided the best results on the test sets. However, Logistic Regression and Gradient Boosting Classifier also showed promising results with high accuracies. 
![basemodelsperformance](https://user-images.githubusercontent.com/46492654/161371018-ea000394-5cd5-45e6-bc70-d41abd6570ed.png)

The Random Forest Classifier, Logistic Regression and Gradient Boosting Classifier models had their hyperparameters tuned with **RandomizedSearchCV**, and the **Random Forest Classifier** model provided the best accuracy. 
![confusionmatrix](https://user-images.githubusercontent.com/46492654/161371037-0a90b729-894e-4ae9-859b-27d6376b148f.png)

To improve the **Random Forest Classifier** model even more, additional hyperparameters were added to the initial hyperparameter grid and **GridSearchCV** was leveraged to achieve the best possible model. The best hyperparameters were obtained, and a classificaiton report as well as a confusion matrix was created on the results of the predictions with these hyperparameters.
![rfcvmetrics](https://user-images.githubusercontent.com/46492654/161371045-d86fc45a-3174-4e89-a522-c3075f2b5230.png)

# Feature Importance
Which features contribute most to a model predicting whether someone has heart disease or not?
* `ca`, `thal`, `oldpeak`, `cp`, `thalach` are the most important features for determining/predicting if a patient has heart disease.
* `fbs` and `restecg` are the least important features for determing/predicting if a patient has heart disease.
![featureimportance](https://user-images.githubusercontent.com/46492654/161371061-8c4b5e0d-9d66-45cf-a90b-913135496978.png)

# Conclusions
Did the final and best model achieve the desired accuracy from the problem statement?
> If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue this project.

Since the hgihest accuracy our model achieved was slightly below 90%, the target was not achieved.

Further experimentation will be required, such as testing different models (CatBoost? XGBoost?), trying to tune different hyperparameters, and selecting the most important features for the prediction process. Through these steps, achieving an accuracy closer to or beyond 95% is certainly possible.
