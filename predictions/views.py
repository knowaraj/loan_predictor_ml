"""
Predictions App Views

Handles all user-facing views for loan prediction, application, authentication,
and profile management.

Author: Loan Predictor Team
Version: 1.0
"""

import logging
from typing import Dict, Any
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
import pandas as pd

from .forms import (
    LoanApplicationForm, UserRegisterForm, LoginForm, 
    UserUpdateForm, CustomPasswordChangeForm, UserProfileUpdateForm
)
from .models import (
    PredictionResult, PredictionConfig, UserProfile, 
    LoanApplicationFormData, LoanApplication
)
from .predictor import get_model

logger = logging.getLogger(__name__)


def home(request: HttpRequest) -> HttpResponse:
    """
    Render the application homepage.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Rendered home page template.
    """
    return render(request, "predictions/home.html")


@login_required(login_url='login')
def predict_view(request: HttpRequest) -> HttpResponse:
    """
    Handle loan application and prediction flow.
    
    POST: Accepts loan application form, runs ML prediction, saves results.
    GET: Displays loan application form.
    
    The workflow:
    1. User fills loan application form
    2. Data saved to LoanApplication model
    3. Features extracted and sent to ML pipeline
    4. Prediction and probability stored in PredictionResult
    5. Result page displayed with prediction outcome
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Form page on GET, result page on successful POST.
        
    Note:
        - Requires user authentication
        - Uses configurable threshold from PredictionConfig
        - Default threshold: 0.65
    """
    if request.method == 'POST':
        form = LoanApplicationForm(request.POST)
        if form.is_valid():
            try:
                # Save loan application
                app = form.save(commit=False)
                app.user = request.user
                app.save()
                logger.info(f"Loan application saved: {app.id} by user {request.user.username}")
                
                # Prepare features for ML model
                input_data = {
                    'Gender': app.gender,
                    'Married': app.married,
                    'Dependents': app.dependents,
                    'Education': app.education,
                    'Self_Employed': app.self_employed,
                    'ApplicantIncome': app.applicant_income,
                    'CoapplicantIncome': app.coapplicant_income,
                    'LoanAmount': app.loan_amount,
                    'Loan_Amount_Term': app.loan_amount_term,
                    'Credit_History': app.credit_history,
                    'Property_Area': app.property_area
                }
                
                # Get prediction threshold from config
                config = PredictionConfig.objects.first()
                threshold = config.threshold if config else 0.65
                
                # Run ML prediction
                model = get_model()
                proba = model.predict_proba(pd.DataFrame([input_data]))[:, 1][0]
                predicted = bool(proba > threshold)
                
                # Save prediction result
                pred = PredictionResult.objects.create(
                    application=app,
                    predicted_default=predicted,
                    probability=proba
                )
                logger.info(f"Prediction created: {pred.id} - Probability: {proba:.4f}")
                
                return render(request, 'predictions/result.html', {
                    'prediction': pred,
                    'application': app
                })
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}", exc_info=True)
                messages.error(request, "An error occurred during prediction. Please try again.")
    else:
        form = LoanApplicationForm()
    
    return render(request, 'predictions/form.html', {'form': form})


@login_required(login_url='login')
def apply_loan_view(request: HttpRequest) -> HttpResponse:
    """
    Handle formal loan application submission.
    
    POST: Saves formal loan application with purpose and tenure.
    GET: Displays loan application form.
    
    Workflow:
    1. User selects a prediction result from history
    2. Fills in loan amount, tenure, and purpose
    3. Application saved for admin review
    4. Admin can approve/reject with remarks
    
    Args:
        request (HttpRequest): Django request object.
                              GET param 'prediction' = prediction_id to use.
        
    Returns:
        HttpResponse: Loan application form or success page.
        
    Security:
        - Only the loan owner can apply
        - Prevents duplicate applications for same prediction
        - Requires active prediction to exist
    """
    prediction_id = request.GET.get('prediction')
    loan_app = get_object_or_404(LoanApplication, id=prediction_id, user=request.user)
    
    # Verify prediction exists
    if not hasattr(loan_app, 'prediction'):
        logger.warning(f"No prediction found for application {loan_app.id}")
        messages.error(request, "Prediction not found for this application.")
        return redirect('prediction_history')
    
    prediction = loan_app.prediction
    
    # Check for duplicate applications
    if LoanApplicationFormData.objects.filter(user=request.user, prediction=prediction).exists():
        logger.info(f"User {request.user.username} already applied for prediction {prediction.id}")
        return render(request, "predictions/apply_loan.html", {
            "already_applied": True,
            "prediction": prediction,
            "original_loan_amount": loan_app.loan_amount
        })
    
    if request.method == 'POST':
        loan_amount = request.POST.get('loan_amount')
        loan_tenure = request.POST.get('loan_tenure')
        loan_purpose = request.POST.get('loan_purpose')
        
        LoanApplicationFormData.objects.create(
            user=request.user,
            prediction=prediction,
            loan_amount=loan_amount,
            loan_tenure=loan_tenure,
            loan_purpose=loan_purpose
        )
        logger.info(f"Formal loan application created by {request.user.username}")
        
        return render(request, 'predictions/apply_loan.html', {
            'success': True,
            'loan_amount': loan_amount,
            'prediction': prediction,
            'original_loan_amount': loan_app.loan_amount
        })
    
    return render(request, 'predictions/apply_loan.html', {
        "prediction": prediction,
        "loan_app": loan_app,
        "original_loan_amount": loan_app.loan_amount
    })


def register_view(request: HttpRequest) -> HttpResponse:
    """
    Handle user registration.
    
    POST: Creates new user account and profile.
    GET: Displays registration form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Registration form or redirect to login on success.
    """
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                mobile = form.cleaned_data['mobile_number']
                address = form.cleaned_data['address']
                UserProfile.objects.create(user=user, mobile_number=mobile, address=address)
                logger.info(f"New user registered: {user.username}")
                messages.success(request, "Account created successfully! Please log in.")
                return redirect('login')
            except Exception as e:
                logger.error(f"Registration error: {str(e)}")
                messages.error(request, "Registration failed. Please try again.")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserRegisterForm()
    
    return render(request, 'predictions/register.html', {'form': form})


def login_view(request: HttpRequest) -> HttpResponse:
    """
    Handle user login.
    
    POST: Authenticates user and creates session.
    GET: Displays login form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Login form or redirect to home on success.
    """
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            logger.info(f"User logged in: {user.username}")
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('home')
        else:
            logger.warning("Failed login attempt")
            messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm(request)
    
    return render(request, 'predictions/login.html', {'form': form})


def logout_view(request: HttpRequest) -> HttpResponse:
    """
    Handle user logout and session termination.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Redirect to login page.
    """
    logout(request)
    logger.info(f"User logged out")
    return redirect('login')


@login_required
def user_profile(request: HttpRequest) -> HttpResponse:
    """
    Display and handle user profile updates.
    
    POST: Updates user account and profile information.
    GET: Displays user profile edit form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Profile form page.
    """
    user = request.user
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    if request.method == "POST":
        user_form = UserUpdateForm(request.POST, instance=user)
        profile_form = UserProfileUpdateForm(request.POST, instance=profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            logger.info(f"Profile updated: {user.username}")
            messages.success(request, "Your profile has been updated successfully.")
            return redirect("user_profile")
    else:
        user_form = UserUpdateForm(instance=user)
        profile_form = UserProfileUpdateForm(instance=profile)
    
    return render(request, "predictions/profile.html", {
        "user_form": user_form,
        "profile_form": profile_form,
    })


@login_required
def change_password(request: HttpRequest) -> HttpResponse:
    """
    Handle user password change.
    
    POST: Changes user password with validation.
    GET: Displays password change form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Password change form page.
        
    Note:
        Keeps user logged in after successful password change.
    """
    if request.method == "POST":
        form = CustomPasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            logger.info(f"Password changed: {user.username}")
            messages.success(request, "Your password has been changed successfully.")
            return redirect("change_password")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = CustomPasswordChangeForm(user=request.user)
    
    return render(request, "predictions/change_password.html", {"form": form})


@login_required
def prediction_history(request: HttpRequest) -> HttpResponse:
    """
    Display user's prediction history with dynamic status.
    
    Shows all loan applications submitted by the user with their prediction results.
    Dynamically computes defaulter/non-defaulter status based on current threshold.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Prediction history page.
    """
    predictions = LoanApplication.objects.filter(user=request.user)
    config = PredictionConfig.objects.first()
    threshold = config.threshold if config else 0.65
    
    # Annotate predictions with readable status
    for p in predictions:
        if hasattr(p, 'prediction') and p.prediction is not None:
            p.predicted_status = "Likely Default" if p.prediction.probability > threshold else "Likely Repay"
        else:
            p.predicted_status = "No Prediction"
    
    logger.info(f"Retrieved prediction history for {request.user.username}: {len(predictions)} records")
    return render(request, "predictions/prediction_history.html", {"predictions": predictions})


@login_required
def loan_history(request: HttpRequest) -> HttpResponse:
    """
    Display user's formal loan application history.
    
    Shows all formal loan applications (those that went through admin review).
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Loan history page.
    """
    loans = LoanApplicationFormData.objects.filter(user=request.user)
    logger.info(f"Retrieved loan history for {request.user.username}: {len(loans)} applications")
    return render(request, "predictions/loan_history.html", {"loans": loans})


def user_reset_password(request: HttpRequest) -> HttpResponse:
    """
    Handle password reset for users who forgot their password.
    
    POST: Resets password using email verification.
    GET: Displays password reset form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Password reset form or redirect to login on success.
        
    Note:
        - Only works for non-staff (regular) users
        - Requires valid email in the system
        - New password must be confirmed
    """
    if request.method == "POST":
        email = request.POST.get('email')
        new_password = request.POST.get('newpassword')
        confirm_password = request.POST.get('confirmpassword')
        
        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('user_reset_password')
        
        try:
            user = User.objects.get(email=email, is_staff=False)
            from django.contrib.auth.hashers import make_password
            user.password = make_password(new_password)
            user.save()
            logger.info(f"Password reset for user: {user.username}")
            messages.success(request, "Password changed successfully. Please log in.")
            return redirect('login')
        except User.DoesNotExist:
            logger.warning(f"Password reset attempted with invalid email: {email}")
            messages.error(request, "Invalid email address.")
    
    return render(request, "predictions/user_reset_password.html")