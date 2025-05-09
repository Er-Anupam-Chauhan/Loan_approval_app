# Loan Approval Prediction
![image](https://github.com/user-attachments/assets/4815e4d3-5e67-477d-ad7f-aa59ea422b4d)

**Step 1:** Created a conda environment using python 3.11.3 --> conda create --name loan_pred_app python=3.11.1

**Step 2:** Installed all the dependencies mentioned in the requirements.txt file --> pip install -r requirements.txt

**Step 3:** Load the data

**Step 4:** 
## Data Loading: The dataset is loaded using pandas. The raw CSV file is read into a DataFrame.
Handling Missing Values: Missing values in the dataset are handled using forward fill and mode imputation.

## Feature Engineering:
Categorical features like Gender, Married, and Education are encoded using Label Encoding.
Continuous variables like ApplicantIncome and LoanAmount are scaled down using log transformation.
New features like loan term categories (Short, Medium, Long) are engineered based on the Loan_Amount_Term.

**Step 5:**
## Model Selection: 
We used Logistic Regression for the task. Other classification models were tried like Decision tree, random forest, SVC 

## Training: 
The model is trained on the preprocessed data, and evaluation metrics (accuracy, precision, recall, F1-score) are computed to measure performance.

Precision	--> Helps minimize false positives (avoid giving loans to risky applicants). -- FP Costly
Recall    -->	Useful if you care about not missing good applicants. -- FN Costly
F1 Score	--> Balances precision and recall — good if both errors matter. -- Both taken care

## Model Saving: 
The trained model is saved using pickle for future use.

## Streamlit Web Application
The web application is built using Streamlit and provides an easy interface for users to enter their details and get a loan approval prediction.

##  How to Run the Streamlit App
To run the Streamlit app locally, follow these steps:
Make sure the model has been trained and saved as trained_model.sav.
Run the following command in the terminal to launch the app : streamlit run app.py

