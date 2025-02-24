# 📊 ML Competition 01 - Loan Approval Prediction
## 🚀 Competition Overview

This repository contains the solution for the [Hugging Face ML Competition 0120251](https://huggingface.co/spaces/MLEAFIT/MLCompetition0120251). The challenge focuses on developing a machine learning pipeline to predict loan approvals based on applicant information.

## 📂 Dataset

This dataset contains various attributes related to loan applicants, including personal information, financial status, and credit history. The goal is to use these features to predict whether a loan application will be approved, represented by the **`Loan_Status`** column.


### 🏷 Variables

- **`Loan_ID`**: Unique identifier for each loan application.
- **`Gender`**: Applicant's gender (**Male**/**Female**).
- **`Married`**: Whether the applicant is married (**Yes**/**No**).
- **`Dependents`**: Number of dependents the applicant has.
- **`Education`**: Applicant's education level (**Graduate**/**Not Graduate**).
- **`Self_Employed`**: Whether the applicant is self-employed (**Yes**/**No**).
- **`ApplicantIncome`**: Applicant's monthly income.
- **`CoapplicantIncome`**: Co-applicant's monthly income (if any).
- **`LoanAmount`**: The loan amount applied for.
- **`Loan_Amount_Term`**: Duration of the loan in months.
- **`Credit_History`**: Whether the applicant has a good credit history (**1** for good, **0** for bad).
- **`Property_Area`**: The area where the property is located (**Urban**/**Semiurban**/**Rural**).
- **`Loan_Status`**: **(Target Variable)** Loan approval status (**1** for Approved, **0** for Not Approved).


## 🏗 Project Structure

├── a_data_prep.ipynb # Data Preparation

├── b_eda.ipynb # Exploratory Data Analysis

├── c_training.ipynb # Model Training

├── d_test.ipynb # Model Testing & Predictions

├── loanpred_test.csv # Test Data 

└── README.md # This file

## ⚙️ Technical Approach

### 1️⃣ **Data Preparation (`a_data_prep.ipynb`)**
- **Loading Data**: Imported the training dataset.
- **Data Cleaning**: Handled missing values using imputation techniques:
  - Median imputation for numerical features.
  - Most frequent imputation for categorical features.
- **Feature Engineering**:
  - Categorical features were encoded using OneHotEncoder.
  - Numerical features were standardized using StandardScaler.
- **Preprocessing Pipeline**: Built using `ColumnTransformer` and `Pipeline` for streamlined preprocessing.

### 2️⃣ **Exploratory Data Analysis (`b_eda.ipynb`)**
- **Univariate Analysis**: Analyzed distributions of numerical features (e.g., `ApplicantIncome`, `LoanAmount`).
- **Categorical Feature Exploration**: Count plots for features like `Gender`, `Married`, and `Property_Area`.
- **Correlation Analysis**: Heatmap to examine correlations between numerical variables.
- **Outlier Detection**: Identified and visualized outliers using boxplots.

### 3️⃣ **Model Training (`c_training.ipynb`)**
- **Data Splitting**: Split the training data into train/validation sets (80/20).
- **Model Selection**:
  - Trained a **Random Forest Classifier** as the base model.
  - Implemented **Bagging** using `BaggingClassifier` with Random Forest as the base estimator.
  - Implemented **Boosting** using `GradientBoostingClassifier`.
- **Hyperparameter Tuning**:
  - Optimized Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).
- **Evaluation**:
  - Used **accuracy score** and **classification reports** for validation.
  - Selected the best-performing model (Gradient Boosting) for final testing.

### 4️⃣ **Model Testing & Submission (`d_test.ipynb`)**
- **Preprocessed Test Data**: Applied the same preprocessing pipeline.
- **Predictions**: Generated predictions using the best model.
- **Submission File**: Created `loan_predictions.csv` with predicted loan statuses.

📊 **Results**
Best Model: Gradient Boosting Classifier
Validation Accuracy: ~87.5%

🙌 **Acknowledgments**
Thanks to Santiago Gonzalez, who has been a huge guidance and help in the development of this challenge.
