# Customer Churn Prediction using Machine Learning

##  Overview

This project focuses on predicting customer churn for a telecommunications company. By analyzing historical customer data, a machine learning model is built to identify customers who are likely to leave (churn). This predictive model can help the company proactively offer incentives or address issues to retain valuable customers.

The project uses the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset.

## 📈 Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading and Understanding:**
    * Loaded the dataset (7043 rows, 21 columns).
    * Dropped the `customerID` column as it is not a predictive feature.

2.  **Data Preprocessing:**
    * **Handling Missing Values:** The `TotalCharges` column was identified as an `object` type. It contained 11 rows with empty strings (" "), which represent new customers with zero tenure. These empty strings were replaced with "0.0", and the entire column was converted to a `float` type.
    * **Target Variable Encoding:** The target variable `Churn` was encoded from 'Yes'/'No' to `1`/`0`.
    * **Categorical Feature Encoding:** All 15 categorical (object-type) features (e.g., `gender`, `InternetService`, `Contract`) were encoded using `sklearn.preprocessing.LabelEncoder`. The fitted encoders were saved to `encoders.pkl` for later use in the predictive system.

3.  **Exploratory Data Analysis (EDA):**
    * **Numerical Features:** Analyzed the distribution of `tenure`, `MonthlyCharges`, and `TotalCharges` using histograms and box plots.
    * **Categorical Features:** Visualized the distribution of all categorical features (including `SeniorCitizen`) using count plots to understand customer demographics and service subscriptions.
    * **Correlation:** A heatmap was generated for the numerical features, showing a high positive correlation (0.83) between `tenure` and `TotalCharges`.

4.  **Handling Class Imbalance:**
    * The data was split into training (80%) and testing (20%) sets.
    * A significant class imbalance was observed in the training data (`No Churn`: 4138 vs. `Churn`: 1496).
    * To address this, the **SMOTE (Synthetic Minority Oversampling Technique)** was applied *only to the training data*. This resulted in a balanced training set with 4138 samples for each class.

5.  **Model Training & Selection:**
    * Three different classification models were trained and evaluated using 5-fold cross-validation on the balanced (SMOTE) training data:
        * Decision Tree: ~78% accuracy
        * **Random Forest: ~84% accuracy**
        * XGBoost: ~83% accuracy
    * The **Random Forest Classifier** was selected as the best-performing model based on this initial evaluation.

## 🤖 Model Evaluation

The final Random Forest model was trained on the full balanced training set (`X_train_smote`, `y_train_smote`) and then evaluated on the original, unseen test set (`X_test`, `y_test`).

* **Accuracy:** 77.9%

* **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       0.85      0.85      0.85      1036
               1       0.58      0.59      0.58       373

        accuracy                           0.78      1409
       macro avg       0.72      0.72      0.72      1409
    weighted avg       0.78      0.78      0.78      1409
    ```

* **Confusion Matrix:**
    ```
    [[878 158]
     [154 219]]
    ```
    * **True Negatives (Kept 'No Churn'):** 878
    * **False Positives (Predicted 'Churn', but was 'No'):** 158
    * **False Negatives (Predicted 'No Churn', but was 'Yes'):** 154
    * **True Positives (Correctly Predicted 'Churn'):** 219

## 💾 Saving and Using the Model

1.  **Model Saving:** The trained Random Forest model and the list of feature names are saved in the `customer_churn_model.pkl` file. The `LabelEncoder` objects for all categorical features are saved in `encoders.pkl`.

2.  **Predictive System:**
    The notebook includes a final section demonstrating how to build a predictive system. This system:
    * Loads the saved `customer_churn_model.pkl` and `encoders.pkl`.
    * Takes new customer data as a Python dictionary.
    * Uses the loaded encoders to transform the categorical string inputs into the numerical format the model expects.
    * Feeds the processed data into the loaded model to generate a prediction (e.g., "No Churn") and the corresponding probability.

    **Example Prediction:**
    * **Input:** (First row of the dataset)
    * **Prediction:** `[0]`
    * **Result:** `Prediction: No Churn`
    * **Probability:** `[[0.78 0.22]]` (78% probability of 'No Churn')

## 🛠️ How to Run

1.  **Clone the repository.**
2.  **Install the required libraries:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
    ```
3.  **Run the Jupyter Notebook:**
    `Customer_Churn_Prediction_using_ML.ipynb`

## 📚 Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn (for SMOTE)
* xgboost
* pickle
