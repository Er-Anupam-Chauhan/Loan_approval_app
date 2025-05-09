from config import * 

def model_training(model,df):

    """
    Train a machine learning model and evaluate its performance on the test set.

    This function splits the preprocessed dataset into training and test sets,
    fits the provided model, evaluates it using classification metrics, and
    saves the trained model to a `.sav` file using pickle.

    Parameters:
    model: An instantiated scikit-learn classifier (e.g., LogisticRegression, RandomForestClassifier).
    df (pd.DataFrame): Preprocessed dataframe with feature columns and 'Loan_Status' as the target.

    Outputs:
    - Prints accuracy, recall, precision, and F1 score.
    - Saves the trained model to 'trained_model.sav'.
    """
    
    logger.info(" Started model_training ")

    x = df.drop(["Loan_Status"],axis=1)
    y = df["Loan_Status"]
    
    sm = SMOTE(random_state=42) 
    X_resampled, y_resampled = sm.fit_resample(x, y)  # Apply SMOTE to balance the classes
    print(y_resampled.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=7, stratify=y_resampled)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=7)

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    recall = recall_score(y_pred,y_test)
    precision = precision_score(y_pred,y_test)
    f1_score_value = f1_score(y_pred,y_test)
    print("Accuracy : ",accuracy)
    print("recall : ",recall)
    print("precision : ",precision)
    print("f1_score : ",f1_score_value)

    logger.info(" Finished model_training ")

    filename = "trained_model.sav"
    pickle.dump(model,open(filename,'wb'))

    logger.info(" Model Saved as .sav file ")