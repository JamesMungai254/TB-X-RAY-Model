from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
import os
from .form import PatientsModelForm


# Assuming `predict` is a function imported from your model file
from .models import predict


def prediction(request):
    return render(request, 'prediction.html')


def welcome(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def team(request):
    return render(request, 'team.html')

# def patient_view(request):
#     if request.method == 'POST':
#         form = PatientsModelForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('success')
#     else:
#         form = PatientsModelForm()
#     return render(request, 'prediction.html', {'form': form})



def upload_and_predict(request):
    form = PatientsModelForm();
    if request.method == 'POST' and 'image' in request.FILES:
        # Save the uploaded image
        uploaded_image = request.FILES['image']
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Construct the full image path
        image_path = os.path.join(upload_dir, uploaded_image.name)

        # Save the image to the filesystem
        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Predict using the model
        prediction_result = predict(image_path)
        result = 'Tuberculosis' if prediction_result == 1 else 'Normal'

        # Relative path for serving the image
        relative_image_path = os.path.join('uploads', uploaded_image.name)
        
        # form
        form = PatientsModelForm(request.POST)
        if form.is_valid():
            form.save()

        # Render the result
        return render(
            request,
            'prediction.html',
            {
                'form': form,
                'result': result,
                'image_path': relative_image_path,
                'MEDIA_URL': settings.MEDIA_URL,
            }
        )

    return JsonResponse({'error': 'No image uploaded'}, status=400)
