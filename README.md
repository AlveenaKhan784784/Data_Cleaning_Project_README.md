# Data_Cleaning_Project_README.md

## Dataset
The dataset used is the Titanic dataset from Kaggle, which includes features like `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`. 

## Data Preprocessing
- Handled missing values in the `Age` and `Fare` columns by using median imputation.
- One-hot encoded categorical variables `Embarked` and `Pclass`.
- Detected and treated outliers in the `Age` and `Fare` columns.
