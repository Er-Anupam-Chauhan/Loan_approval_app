from config import * 

def term_group(term):
    """
    Categorize loan term into groups.

    Parameters:
    term (int or float): The loan term in months.

    Returns:
    str: Category of loan term - 'Short', 'Medium', or 'Long'.
    """

    if term <= 60:
        return 'Short'
    elif term <= 180:
        return 'Medium'
    else:
        return 'Long'
    

def scale_down(df):
    """
    Apply transformations to normalize and scale data distributions.

    Transforms 'LoanAmount' and 'ApplicantIncome' to reduce skewness
    and applies standard scaling.

    Parameters:
    df (pd.DataFrame): Input dataframe with raw values.

    Returns:
    pd.DataFrame: Transformed dataframe with scaled features.
    """
        
    logger.info(" Fixing distributions")
    # Fixing the data distributions
    df['LoanAmount'] = np.log1p(df['LoanAmount'])
    
    # Engineering new scaled down feature to prevent hallucination
    scaler = StandardScaler()
    logger.info(" Scaling values")
    df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
    df['ApplicantIncome'] = scaler.fit_transform(df[['ApplicantIncome']])
    
    return df


def processing_null_data(df):
    """
    Handle missing values in the dataset.

    Fills missing values using appropriate strategies:
    - Forward fill for 'Loan_Amount_Term' and 'Credit_History'
    - Mode for categorical features
    - Mean for 'LoanAmount'

    Parameters:
    df (pd.DataFrame): Dataframe with missing values.

    Returns:
    pd.DataFrame: Cleaned dataframe with no missing values.
    """

    logger.info(" Handling missing values ")
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(method='ffill')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype(int)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].apply(term_group)

    # replacing categorical variables with mode of the column and numerical with the mean
    gender_mode = df['Gender'].mode()[0]
    Married_mode = df['Married'].mode()[0]
    Dependents_mode = df['Dependents'].mode()[0]
    Self_Employed_mode = df['Self_Employed'].mode()[0]
    LoanAmount_mean = df['LoanAmount'].mean().round(2)
    Credit_History_mean = df['Credit_History'].mean().round(2)
    
    df['Gender'] = df['Gender'].fillna(gender_mode)
    df['Married'] = df['Married'].fillna(Married_mode)
    df['Dependents'] = df['Dependents'].fillna(Dependents_mode)
    df['Self_Employed'] = df['Self_Employed'].fillna(Self_Employed_mode)
    df['LoanAmount'] = df['LoanAmount'].fillna(LoanAmount_mean)
    df['Credit_History'] = df['Credit_History'].fillna(method='ffill')
    
    return df


def data_encoding(df):
    """
    Perform label encoding and one-hot encoding on categorical features.

    - Label encoding: Gender, Married, Dependents, Self_Employed, Education, Loan_Status
    - One-hot encoding: Property_Area, Loan_Amount_Term

    Parameters:
    df (pd.DataFrame): Cleaned dataframe with raw categorical values.

    Returns:
    pd.DataFrame: Encoded dataframe suitable for ML modeling.
    """
    
    print("------- Unique values before encoding -------")
    print("Gender unique values : ",df['Gender'].unique())
    print("Married unique values : ",df['Married'].unique())
    print("Dependents unique values : ",df['Dependents'].unique())
    print("Self_Employed unique values : ",df['Self_Employed'].unique())
    # print("LoanAmount unique values : ",df['LoanAmount'].unique())
    print("Loan_Amount_Term unique values : ",df['Loan_Amount_Term'].unique())
    print("Credit_History unique values : ",df['Credit_History'].unique())
    print("Education unique values : ",df['Education'].unique())
    print("Property_Area unique values : ",df['Property_Area'].unique())

    # label encoding
    logger.info(" Performing Label encoding ")
    df['Loan_Status'] = df['Loan_Status'].replace({"N":0,"Y":1})
    df['Dependents'] = df['Dependents'].replace({"0":0,"1":1,"2":2,"3+":3})
    df['Married'] = df['Married'].replace({"No":0,"Yes":1})
    df['Gender'] = df['Gender'].replace({"Female":0,"Male":1})
    df['Self_Employed'] = df['Self_Employed'].replace({"No":0,"Yes":1})
    df['Education'] = df['Education'].replace({"Not Graduate":0,"Graduate":1})
    
    df['Credit_History'] = df['Credit_History'].astype(int)

    # One hot encoding
    logger.info(" Performing One Hot encoding ")
    df = pd.get_dummies(df, columns=['Property_Area'])
    df = pd.get_dummies(df, columns=['Loan_Amount_Term'])
    
    print("------- Unique values Post encoding -------")
    print("Gender unique values : ",df['Gender'].unique())
    print("Married unique values : ",df['Married'].unique())
    print("Dependents unique values : ",df['Dependents'].unique())
    print("Self_Employed unique values : ",df['Self_Employed'].unique())
    print("Credit_History unique values : ",df['Credit_History'].unique())
    print("Education unique values : ",df['Education'].unique())

    df = df.drop(["Loan_ID","CoapplicantIncome"],axis=1)
    
    return df

def preprocess_main(df):

    logger.info(" Started preprocess_main ")

    df = scale_down(df)

    df = processing_null_data(df)

    df = data_encoding(df)

    logger.info(" End preprocess_main ")


    return df



