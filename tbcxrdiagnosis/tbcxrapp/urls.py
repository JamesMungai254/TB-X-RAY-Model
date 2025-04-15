from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('test/', views.prediction, name='prediction'),
    path('about/', views.about, name='about'),
    path('upload/', views.upload_and_predict, name='upload_and_predict'),
    path('team/', views.team, name='team'),
   
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
