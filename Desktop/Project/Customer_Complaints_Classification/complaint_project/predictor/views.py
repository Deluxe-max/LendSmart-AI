from django.shortcuts import render
import joblib
import os
from django.conf import settings

# 1. Build the exact file paths to where we stored the AI's brain
vectorizer_path = os.path.join(settings.BASE_DIR, 'predictor', 'tfidf_vectorizer.pkl')
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'complaint_classifier_model.pkl')

# 2. Load the Dictionary and the Brain into the server's memory
tfidf = joblib.load(vectorizer_path)
model = joblib.load(model_path)

def home_page(request):
    predicted_department = None
    user_complaint = ""

    # If the user clicks the "Submit" button on the website...
    if request.method == 'POST':
        # Grab the paragraph they typed in
        user_complaint = request.POST.get('complaint_text')

        if user_complaint:
            # 1. Translate the English text into the TF-IDF math matrix
            vectorized_text = tfidf.transform([user_complaint])
            
            # 2. Ask the AI to predict the department!
            prediction = model.predict(vectorized_text)
            predicted_department = prediction[0]

    # Send the result back to the HTML webpage
    return render(request, 'form.html', {
        'prediction': predicted_department,
        'user_text': user_complaint
    })
# Create your views here.
