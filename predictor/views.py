from django.shortcuts import render
import joblib # Change this to 'import joblib' if you saved your model with joblib!

def home_page(request):
    result = None # This will hold our final answer

    # Check if the user clicked the "Predict Approval" button (a POST request)
    if request.method == 'POST':
        try:
            # 1. Grab the raw numbers from the HTML form boxes
            applicant_income = float(request.POST.get('ApplicantIncome'))
            coapplicant_income = float(request.POST.get('CoapplicantIncome'))
            loan_amount = float(request.POST.get('LoanAmount'))
            loan_amount_term = float(request.POST.get('Loan_Amount_Term'))
            credit_history = float(request.POST.get('Credit_History'))
            dependents = float(request.POST.get('Dependents'))

            # 2. Grab the text dropdowns and translate them to numbers 
            # (This does the exact same job as your LabelEncoder!)
            gender_dict = {'Female': 0, 'Male': 1}
            married_dict = {'No': 0, 'Yes': 1}
            education_dict = {'Graduate': 0, 'Not Graduate': 1}
            self_employed_dict = {'No': 0, 'Yes': 1}
            property_area_dict = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

            gender = gender_dict[request.POST.get('Gender')]
            married = married_dict[request.POST.get('Married')]
            education = education_dict[request.POST.get('Education')]
            self_employed = self_employed_dict[request.POST.get('Self_Employed')]
            property_area = property_area_dict[request.POST.get('Property_Area')]

            # 3. Group them into a list in the EXACT order your model was trained on
            user_data = [[gender, married, dependents, education, self_employed, 
                          applicant_income, coapplicant_income, loan_amount, 
                          loan_amount_term, credit_history, property_area]]
            # 4. Load your trained model AND your scaler
            model = joblib.load('loan_model.pkl')
            scaler = joblib.load('scaler.pkl') # <--- Load the squisher!

            # 4.5. Squish the user's data so the model understands it
            # (We use scaler.transform to apply the exact same math used during training)
            scaled_user_data = scaler.transform(user_data)

            # 5. Make the prediction using the SCALED data!
            prediction = model.predict(scaled_user_data)
            
            # 6. Translate the math answer (1 or 0) back into English for the user
            if prediction[0] == 1:
                result = "APPROVED! 🎉"
            else:
                result = "REJECTED. ❌"

        except Exception as e:
            # If they leave a box blank or a file is missing, show an error
            result = f"Error processing data: {e}"

    # Send the result back to the HTML page to be displayed
    return render(request, 'form.html', {'result': result})