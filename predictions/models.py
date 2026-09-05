"""
Prediction Models Module

Defines database models for loan applications, predictions, and user profiles.
These models store borrower information, prediction results, and configuration.

Author: Loan Predictor Team
Version: 1.0
"""

from django.db import models
from django.contrib.auth.models import User


class LoanApplication(models.Model):
    """
    Stores initial loan application data from borrowers.
    
    This model captures borrower demographics, financial information, and
    credit history which are used as input to the ML prediction model.
    
    Attributes:
        user (ForeignKey): Reference to the User who submitted the application.
        gender (str): Borrower gender (Male/Female).
        married (str): Marital status (Yes/No/Unknown).
        dependents (str): Number of dependents.
        education (str): Education level (Graduate/Undergraduate).
        self_employed (str): Self-employment status (Yes/No).
        applicant_income (float): Applicant's monthly income.
        coapplicant_income (float): Co-applicant's monthly income (if any).
        loan_amount (float): Requested loan amount.
        loan_amount_term (float): Loan term in months.
        credit_history (int): Credit history availability (0/1).
        property_area (str): Property location (Urban/Rural/Semiurban).
        created_at (datetime): Application submission timestamp.
    """
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    gender = models.CharField(max_length=20)
    married = models.CharField(max_length=10)
    dependents = models.CharField(max_length=10)
    education = models.CharField(max_length=20)
    self_employed = models.CharField(max_length=10)
    applicant_income = models.FloatField()
    coapplicant_income = models.FloatField()
    loan_amount = models.FloatField()
    loan_amount_term = models.FloatField(null=True, blank=True)
    credit_history = models.IntegerField(null=True, blank=True)
    property_area = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Loan Applications"

    def __str__(self) -> str:
        return f"Application {self.id} - {self.user} - Rs{self.loan_amount}"


class PredictionResult(models.Model):
    """
    Stores ML model prediction results for loan applications.
    
    Links a prediction (default probability) to its corresponding loan application,
    enabling historical tracking and audit trails.
    
    Attributes:
        application (OneToOneField): Link to LoanApplication being predicted.
        predicted_default (bool): Whether applicant predicted to default (True/False).
        probability (float): Probability of default (0.0 to 1.0).
        model_version (str): Version of ML model used for prediction.
        created_at (datetime): Prediction generation timestamp.
    """
    application = models.OneToOneField(LoanApplication, on_delete=models.CASCADE, related_name='prediction')
    predicted_default = models.BooleanField()
    probability = models.FloatField()
    model_version = models.CharField(max_length=20, default='v1')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        status = "Will Default" if self.predicted_default else "Will Repay"
        return f"Prediction {self.id} - {status} ({self.probability:.2%})"


class LoanApplicationFormData(models.Model):
    """
    Tracks formal loan application requests with admin review workflow.
    
    After ML prediction, users can submit formal loan applications that go
    through admin approval/rejection process with remarks.
    
    Attributes:
        user (ForeignKey): User submitting the formal application.
        prediction (ForeignKey): Associated prediction result.
        loan_amount (float): Amount of loan requested.
        loan_tenure (int): Requested loan term in months.
        loan_purpose (str): Stated purpose of the loan.
        applied_at (datetime): Application submission time.
        status (str): Admin review status (Pending/Approved/Rejected).
        remark (str): Admin notes or rejection reasons.
    """
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]

    user = models.ForeignKey('auth.User', on_delete=models.SET_NULL, null=True, blank=True)
    prediction = models.ForeignKey('PredictionResult', on_delete=models.SET_NULL, null=True, blank=True)
    loan_amount = models.FloatField()
    loan_tenure = models.IntegerField(help_text="Tenure in months")
    loan_purpose = models.CharField(max_length=200)
    applied_at = models.DateTimeField(auto_now_add=True)

    # Admin fields
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    remark = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-applied_at']
        verbose_name_plural = "Loan Application Form Data"

    def __str__(self) -> str:
        return f"{self.user} - Rs{self.loan_amount} - {self.status}"


class PredictionConfig(models.Model):
    """
    Configuration settings for ML prediction threshold.
    
    Stores the probability threshold used to classify predictions as
    default/non-default. Admins can adjust this without retraining.
    
    Attributes:
        threshold (float): Probability threshold for default classification.
                         Applicants with prob > threshold are marked as defaulters.
    """
    threshold = models.FloatField(
        default=0.65, 
        help_text="Probability above which applicant is classified as defaulter"
    )

    class Meta:
        verbose_name_plural = "Prediction Config"

    def __str__(self) -> str:
        return f"Prediction Threshold: {self.threshold:.2%}"


class UserProfile(models.Model):
    """
    Extended user profile with additional contact and location information.
    
    Extends Django's built-in User model with domain-specific fields
    for loan application context.
    
    Attributes:
        user (OneToOneField): Link to Django User account.
        mobile_number (str): Contact phone number.
        address (str): Physical address for correspondence.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    mobile_number = models.CharField(max_length=15)
    address = models.TextField()

    class Meta:
        verbose_name_plural = "User Profiles"

    def __str__(self) -> str:
        return f"Profile of {self.user.username}"
