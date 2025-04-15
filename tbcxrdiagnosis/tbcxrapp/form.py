from django import forms
from .form_model import PatientsModel

class PatientsModelForm(forms.ModelForm):
    class Meta:
        model = PatientsModel
        fields = ['p_name', 'p_status','p_address']