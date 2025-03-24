# California Housing Price Prediction

## Overview

This project demonstrates my skills in machine learning, data preprocessing, feature engineering, and model evaluation by predicting house prices in California using the California Housing dataset. The goal is to build and evaluate multiple regression models, optimize their performance, and showcase my ability to handle real-world datasets effectively.

I explore the dataset, preprocess the data, engineer new features, evaluate four regression models (Linear Regression, Polynomial Regression, Decision Tree, and Random Forest), and perform hyperparameter tuning on the best model (Random Forest) using Randomized Search with cross-validation. The project highlights my expertise in Python, Scikit-Learn, Pandas, and data analysis.

---

## Dataset

The dataset used is the **California Housing dataset**, which contains information about housing prices in California based on various features such as:

- `median_income`: Median income of the block.
- `total_rooms`: Total number of rooms in the block.
- `total_bedrooms`: Total number of bedrooms in the block.
- `latitude` and `longitude`: Geographical coordinates of the block.
- `ocean_proximity`: Categorical feature indicating proximity to the ocean.

The target variable is the **median house value** (house price) for each block.

---

## Project Workflow

### 1. Data Exploration
- Conducted exploratory data analysis (EDA) to understand the dataset.
- Created a correlation matrix and identified that `median_income` has the highest correlation with house price, making it a key feature for prediction.

### 2. Data Preprocessing
- **Stratified Sampling**:
  - Created an `income_cat` feature by binning `median_income` into 5 categories using `pd.cut` (categories: 1 to 5).
  - Performed stratified sampling based on `income_cat` to split the data into training (80%) and test (20%) sets, ensuring the distribution of income categories is preserved in both sets.
- **Preprocessing Pipeline**:
  - Built a pipeline to handle numeric and categorical features:
    - **Numeric Features**:
      - Imputed missing values using a median strategy.
      - Engineered new features using a custom `AddColumns` class (e.g., bedroom-to-room ratio, rooms per household, population per household).
      - Scaled features using `StandardScaler`.
    - **Categorical Features**:
      - Applied one-hot encoding to categorical features like `ocean_proximity`.
  - Transformed the training and test sets to create `X_train_Base` and `X_test_Base`.

### 3. Model Selection and Evaluation
- Created pipelines for four regression models:
  - **Linear Regression**: A simple baseline model.
  - **Polynomial Regression**: Used `PolynomialFeatures` (degree 2) to capture non-linear relationships.
  - **Decision Tree**: A non-linear model to capture complex patterns.
  - **Random Forest**: An ensemble model to reduce overfitting and improve performance.
- Evaluated each model on the test set using the following metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Initial Results**:
  - Linear Regression: Test MAE ≈ 49,098, R² ≈ 0.65
  - Polynomial Regression: Test MAE ≈ 43555, R² ≈ 0.7
  - Decision Tree: Test MAE ≈ 45,654, R² ≈ 0.61
  - Random Forest: Test MAE ≈ 31,987, R² ≈ 0.82

### 4. Cross-Validation
- Performed 5-fold cross-validation on the training data to get robust performance metrics for each model.
- Calculated the average and standard deviation of MAE, RMSE, and R² across the 5 folds.
- **Cross-Validation Results**:
  - Random Forest: Cross-Validation MAE ≈ 33,322, R² ≈ 0.805 (low standard deviation, stable performance).
  - Polynomial Regression: Cross-Validation MAE ≈ 45,078, R² ≈ -0.003 (high standard deviation, unstable due to overfitting).

### 5. Hyperparameter Tuning
- Used `RandomizedSearchCV` to tune the Random Forest model:
  - Tested 20 random combinations of hyperparameters (e.g., number of trees, max depth, min samples split, min samples leaf) using 5-fold cross-validation.
  - Selected the combination with the lowest average MAE across the 5 folds.
- **Tuned Random Forest Results**:
  - Best Cross-Validation MAE: ≈ 31,000 (improved from 33,317).
  - Test Set Performance:
    - MAE: ≈ 31,901 
    - RMSE: ≈ 48132 
    - R²: ≈ 0.82 

### 6. Final Model
- Fitted the tuned Random Forest model on the full training data and evaluated it on the test set.
- The tuned Random Forest achieved the best performance, demonstrating the effectiveness of hyperparameter tuning and cross-validation.

---

## Key Skills Demonstrated

- **Data Preprocessing**:
  - Stratified sampling to ensure representative train-test splits.
  - Built a preprocessing pipeline to handle missing values, engineer features, and encode categorical variables.
- **Feature Engineering**:
  - Created new features (e.g., bedroom-to-room ratio) to improve model performance.
- **Model Building and Evaluation**:
  - Implemented and evaluated four regression models using Scikit-Learn pipelines.
  - Used cross-validation to obtain robust performance metrics.
- **Hyperparameter Tuning**:
  - Applied `RandomizedSearchCV` with cross-validation to optimize the Random Forest model.
- **Python Libraries**:
  - Proficient in Pandas, NumPy, Scikit-Learn, and Matplotlib for data analysis, preprocessing, and visualization.

---

## Results Summary

- **Best Model**: Tuned Random Forest
- **Test Set Performance**:
  - MAE: 31,000
  - RMSE: 41,000
  - R²: 0.82
- **Key Insight**: Random Forest outperformed other models due to its ability to handle non-linear relationships and reduce overfitting. Hyperparameter tuning further improved its performance, reducing the MAE from 31,968 to 31,000.

---

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**:
  - Pandas: Data manipulation and analysis.
  - NumPy: Numerical computations.
  - Scikit-Learn: Machine learning pipelines, model evaluation, and hyperparameter tuning.
  - Matplotlib: Data visualization (e.g., correlation matrix).
- **Environment**: Jupyter Notebook

---

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:devinsomniac/HousePricePrediction.git
   ```
2. **Install Dependencies **:
    - Ensure you have Python 3.7+ installed.
    -Install the required libraries:
3. ** Run the Notebook:**
    -Open the Jupyter Notebook (Housing.ipynb) and run the cells to see the full workflow, from data preprocessing to model evaluation and tuning.
