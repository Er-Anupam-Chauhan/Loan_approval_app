from config import *

model = pickle.load(open("trained_model.sav", "rb")) # Load the saved model

st.title("Loan Approval Prediction") # Title 

st.sidebar.header("Applicant Details") # side bar

# take input
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1)
loan_term = st.sidebar.selectbox("Loan Amount Term (in months)", [12, 36, 60, 84, 120, 180, 240, 300, 342, 360, 480])
credit_history = st.sidebar.selectbox("Credit History", ["Yes","No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


def preprocess_inputs(): # Preprocess the inputs
    data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": np.log1p(applicant_income),
        "LoanAmount": np.log1p(loan_amount),
        "Credit_History":1 if credit_history == "Yes" else 0,
        "Property_Area_Rural": 1 if property_area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        "Loan_Amount_Term_Long": 1 if loan_term > 180 else 0,
        "Loan_Amount_Term_Medium": 1 if 60 < loan_term <= 180 else 0,
        "Loan_Amount_Term_Short": 1 if loan_term <= 60 else 0
    }

    return pd.DataFrame([data])


if st.sidebar.button("Predict Loan Status"):# Predict button
    input_df = preprocess_inputs()
    prediction = model.predict(input_df)
    # result = "Approved" if prediction[0] == 1 else "Rejected"
    # st.success(f"Loan Status: {result}")
    if prediction[0] == 1:
        st.markdown("<h3 style='color: green;'>✅ Loan Status: Approved</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: red;'>❌ Loan Status: Rejected</h3>", unsafe_allow_html=True)

    logger.info(f"Prediction made: {prediction} with inputs: {input_df.to_dict(orient='records')[0]}")
