from django import forms
from .models import LoanApplication,UserProfile
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from django.contrib.auth.forms import PasswordChangeForm


GENDER_CHOICES = [
    ('Male', 'Male'),
    ('Female', 'Female'),
]

YES_NO_CHOICES = [
    ('Yes', 'Yes'),
    ('No', 'No'),
]

DEPENDENTS_CHOICES = [
    ('0', '0'),
    ('1', '1'),
    ('2', '2'),
    ('3+', '3+'),
]

EDUCATION_CHOICES = [
    ('Graduate', 'Graduate'),
    ('Not Graduate', 'Not Graduate'),
]

CREDIT_HISTORY_CHOICES = [
    (1, 'Good (1)'),
    (0, 'Bad (0)'),
]

PROPERTY_AREA_CHOICES = [
    ('Urban', 'Urban'),
    ('Semiurban', 'Semiurban'),
    ('Rural', 'Rural'),
]


class LoanApplicationForm(forms.ModelForm):
    class Meta:
        model = LoanApplication
        exclude = ['user']
        widgets = {
            'gender': forms.Select(choices=GENDER_CHOICES, attrs={'class': 'form-select'}),
            'married': forms.Select(choices=YES_NO_CHOICES, attrs={'class': 'form-select'}),
            'dependents': forms.Select(choices=DEPENDENTS_CHOICES, attrs={'class': 'form-select'}),
            'education': forms.Select(choices=EDUCATION_CHOICES, attrs={'class': 'form-select'}),
            'self_employed': forms.Select(choices=YES_NO_CHOICES, attrs={'class': 'form-select'}),
            'applicant_income': forms.NumberInput(attrs={'placeholder': 'e.g. 5000', 'class': 'form-control'}),
            'coapplicant_income': forms.NumberInput(attrs={'placeholder': 'e.g. 2000', 'class': 'form-control'}),
            'loan_amount': forms.NumberInput(attrs={'placeholder': 'e.g. 120', 'class': 'form-control'}),
            'loan_amount_term': forms.NumberInput(attrs={'placeholder': 'e.g. 360', 'class': 'form-control'}),
            'credit_history': forms.Select(choices=CREDIT_HISTORY_CHOICES, attrs={'class': 'form-select'}),
            'property_area': forms.Select(choices=PROPERTY_AREA_CHOICES, attrs={'class': 'form-select'}),
        }



class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=50)
    last_name = forms.CharField(max_length=50)
    email = forms.EmailField()
    mobile_number = forms.CharField(max_length=15)
    address = forms.CharField(widget=forms.Textarea)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'mobile_number', 'address', 'password1', 'password2']

class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class UserProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['mobile_number', 'address']

# Change Password Form (already provided by Django, we just use it)
class CustomPasswordChangeForm(PasswordChangeForm):
    pass