from django.shortcuts import render
from .models import Product



from .ml_model import load_model, load_vectorizer

def index(request):
    return render(request, 'index.html')

# load the ML model
def predict(request):
    if request.method == 'POST':
        review = request.POST.get('review', '')
        # load the vectorizer
        vectorizer = load_vectorizer()
        # transform the input so that it fits the model
        review_vectorized = vectorizer.transform([review])
        # load the model
        model = load_model()
        # make the prediction
        prediction = model.predict(review_vectorized)[0]
        if prediction == 'Positive':
            prediction = 1
        else:
            prediction = 0

        # create a stars based on the prediction
        if prediction == 1:
            stars = '⭐⭐⭐⭐⭐'
        else:
            stars = '⭐'

        # render the result
        return render(request, 'index.html', {'review': review, 'prediction': prediction, 'stars': stars})
    else:
        return render(request, 'index.html', {'prediction': 'No prediction has been made yet.'})
    
def products(request):
    products = Product.objects.all()
    return render(request, 'products.html', {'products': products})