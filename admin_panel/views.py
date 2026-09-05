"""
Admin Panel Views

Handles administrative functions for loan review, user management, and system monitoring.
Only accessible to staff users (is_staff=True).

Author: Loan Predictor Team
Version: 1.0
"""

import logging
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib import messages
from django.contrib.auth.models import User
from django import forms
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator

from predictions.models import (
    LoanApplicationFormData, UserProfile, LoanApplication, PredictionConfig
)

logger = logging.getLogger(__name__)


def admin_required(user) -> bool:
    """
    Check if user is authenticated and has staff privileges.
    
    Args:
        user (User): Django user object to check.
        
    Returns:
        bool: True if user is authenticated staff, False otherwise.
    """
    return user.is_authenticated and user.is_staff


def admin_panel_root(request: HttpRequest) -> HttpResponse:
    """
    Root redirect for admin panel.
    
    Redirects authenticated staff users to dashboard, others to admin login.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Redirect to dashboard or login page.
    """
    if request.user.is_authenticated and request.user.is_staff:
        logger.info(f"Admin access granted to {request.user.username}")
        return redirect('admin_dashboard')
    return redirect('admin_login')


def admin_login_view(request: HttpRequest) -> HttpResponse:
    """
    Handle admin login.
    
    POST: Authenticates admin user and creates session.
    GET: Displays admin login form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Login form on GET, redirect to dashboard on successful POST.
        
    Security:
        - Only staff users can login
        - Failed attempts are logged
    """
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:
            login(request, user)
            logger.info(f"Admin user logged in: {user.username}")
            return redirect('admin_dashboard')
        else:
            logger.warning(f"Failed admin login attempt for: {username}")
            messages.error(request, "Invalid credentials or you are not an admin!")
    
    return render(request, 'login.html')


def admin_logout_view(request: HttpRequest) -> HttpResponse:
    """
    Handle admin logout and session termination.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Redirect to admin login page.
    """
    logger.info(f"Admin user logged out: {request.user.username}")
    logout(request)
    return redirect('admin_login')


@user_passes_test(admin_required, login_url='admin_login')
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """
    Display admin dashboard with loan statistics.
    
    Shows key metrics:
    - Total loan applications
    - Pending applications awaiting review
    - Approved applications
    - Rejected applications
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Dashboard page with statistics.
        
    Security:
        Requires admin authentication.
    """
    total_loans = LoanApplicationFormData.objects.count()
    pending = LoanApplicationFormData.objects.filter(status='Pending').count()
    approved = LoanApplicationFormData.objects.filter(status='Approved').count()
    rejected = LoanApplicationFormData.objects.filter(status='Rejected').count()
    
    logger.info(f"Dashboard accessed by {request.user.username}")
    
    context = {
        'total_loans': total_loans,
        'pending': pending,
        'approved': approved,
        'rejected': rejected
    }
    return render(request, 'dashboard.html', context)


@user_passes_test(admin_required, login_url='admin_login')
def loan_applications_view(request: HttpRequest) -> HttpResponse:
    """
    List all loan applications with optional status filtering.
    
    GET params:
        status (str, optional): Filter by status (Pending/Approved/Rejected)
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: List of loan applications.
        
    Security:
        Requires admin authentication.
    """
    status = request.GET.get('status')
    if status:
        loans = LoanApplicationFormData.objects.filter(status=status).order_by('-applied_at')
        logger.info(f"Filtered loans by status {status} for admin {request.user.username}")
    else:
        loans = LoanApplicationFormData.objects.all().order_by('-applied_at')
    
    return render(request, 'loan_list.html', {'loans': loans})


@user_passes_test(admin_required, login_url='admin_login')
def loan_detail_view(request: HttpRequest, id: int) -> HttpResponse:
    """
    Display and update loan application details.
    
    POST: Updates loan status and admin remarks.
    GET: Displays loan application details.
    
    Args:
        request (HttpRequest): Django request object.
        id (int): Loan application ID.
        
    Returns:
        HttpResponse: Loan detail page or redirect on update.
        
    Security:
        Requires admin authentication.
    """
    loan = get_object_or_404(LoanApplicationFormData, id=id)
    
    if request.method == 'POST':
        old_status = loan.status
        loan.status = request.POST.get('status')
        loan.remark = request.POST.get('remark')
        loan.save()
        logger.info(f"Loan {id} status updated by {request.user.username}: {old_status} -> {loan.status}")
        messages.success(request, f"Loan status updated to {loan.status}")
        return redirect('admin_loan_list')
    
    return render(request, 'loan_detail.html', {'loan': loan})


@user_passes_test(admin_required, login_url='admin_login')
def user_list_view(request: HttpRequest) -> HttpResponse:
    """
    List all regular users (non-staff) ordered by registration date.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: User list page.
        
    Security:
        Requires admin authentication.
        Only shows non-staff users.
    """
    users = User.objects.filter(is_staff=False).order_by('-date_joined')
    logger.info(f"User list accessed by admin {request.user.username}: {len(users)} users")
    return render(request, 'user_list.html', {'users': users})


@user_passes_test(admin_required, login_url='admin_login')
def admin_user_profile(request: HttpRequest, user_id: int) -> HttpResponse:
    """
    Display user profile and extended user profile information.
    
    Args:
        request (HttpRequest): Django request object.
        user_id (int): User ID to view profile for.
        
    Returns:
        HttpResponse: User profile details page.
        
    Security:
        Requires admin authentication.
    """
    user = get_object_or_404(User, id=user_id)
    profile = get_object_or_404(UserProfile, user=user)
    logger.info(f"Admin {request.user.username} viewed profile of user {user.username}")
    return render(request, 'user_profile.html', {'user': user, 'profile': profile})


@user_passes_test(admin_required, login_url='admin_login')
def admin_user_loans(request: HttpRequest, user_id: int) -> HttpResponse:
    """
    Display all loan applications submitted by a specific user.
    
    Args:
        request (HttpRequest): Django request object.
        user_id (int): User ID to retrieve loans for.
        
    Returns:
        HttpResponse: User's loan applications page.
        
    Security:
        Requires admin authentication.
    """
    user = get_object_or_404(User, id=user_id)
    loans = LoanApplicationFormData.objects.filter(user=user).order_by('-applied_at')
    logger.info(f"Admin {request.user.username} viewed loans for user {user.username}")
    return render(request, 'user_loans.html', {'user': user, 'loans': loans})


@user_passes_test(admin_required, login_url='admin_login')
def admin_delete_user(request: HttpRequest, user_id: int) -> HttpResponse:
    """
    Delete a user account and all related data.
    
    CASCADE delete removes:
    - User account
    - User profile
    - All loan applications
    - All predictions
    
    Args:
        request (HttpRequest): Django request object.
        user_id (int): User ID to delete.
        
    Returns:
        HttpResponse: Redirect to user list.
        
    Security:
        Requires admin authentication.
        Note: This action is irreversible.
    """
    user = get_object_or_404(User, id=user_id)
    username = user.username
    user.delete()
    logger.warning(f"User {username} deleted by admin {request.user.username}")
    messages.success(request, f"User {username} and all related data deleted successfully.")
    return redirect('user_list')


class AdminProfileForm(forms.ModelForm):
    """
    Form for admin to update their own profile information.
    
    Allows editing of name and email only (not password).
    """
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }


@user_passes_test(admin_required, login_url='admin_login')
def admin_profile_view(request: HttpRequest) -> HttpResponse:
    """
    Display and update admin profile.
    
    POST: Updates admin's first name, last name, and email.
    GET: Displays profile edit form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Admin profile page.
        
    Security:
        Requires admin authentication.
        Password changes handled separately in change_password view.
    """
    user = request.user
    if request.method == 'POST':
        form = AdminProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            logger.info(f"Admin {user.username} updated profile")
            messages.success(request, 'Profile updated successfully!')
            return redirect('admin_profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = AdminProfileForm(instance=user)
    
    return render(request, 'admin_profile.html', {'form': form, 'user': user})


@user_passes_test(admin_required, login_url='admin_login')
def admin_change_password_view(request: HttpRequest) -> HttpResponse:
    """
    Handle admin password change.
    
    POST: Changes password with current password verification.
    GET: Displays password change form.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: Password change form.
        
    Security:
        Requires admin authentication.
        User remains logged in after successful password change.
    """
    if request.method == 'POST':
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            logger.info(f"Admin {user.username} changed password")
            messages.success(request, 'Password changed successfully!')
            return redirect('admin_change_password')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PasswordChangeForm(user=request.user)
    
    return render(request, 'admin_change_password.html', {'form': form})


@user_passes_test(admin_required, login_url='admin_login')
def admin_predictions_view(request: HttpRequest) -> HttpResponse:
    """
    Display all loan predictions with their applications.
    
    Shows prediction history with model confidence scores.
    
    Args:
        request (HttpRequest): Django request object.
        
    Returns:
        HttpResponse: All predictions page.
        
    Security:
        Requires admin authentication.
    """
    applications = LoanApplication.objects.select_related('prediction').order_by('-created_at')
    logger.info(f"Admin {request.user.username} viewed all predictions")
    return render(request, 'admin_predictions.html', {'applications': applications})


@user_passes_test(admin_required, login_url='admin_login')
def delete_prediction(request: HttpRequest, app_id: int) -> HttpResponse:
    """
    Delete a prediction record and its associated application.
    
    Args:
        request (HttpRequest): Django request object.
        app_id (int): LoanApplication ID to delete.
        
    Returns:
        HttpResponse: Redirect to predictions list.
        
    Security:
        Requires admin authentication.
        Note: This action is irreversible.
    """
    application = get_object_or_404(LoanApplication, id=app_id)
    
    if hasattr(application, 'prediction'):
        logger.warning(f"Prediction {application.prediction.id} deleted by admin {request.user.username}")
        application.delete()
        messages.success(request, "Prediction deleted successfully!")
    else:
        messages.warning(request, "No prediction found for this application.")
    
    return redirect('admin_predictions')


@user_passes_test(admin_required, login_url='admin_login')
def admin_user_predictions(request: HttpRequest, user_id: int) -> HttpResponse:
    """
    Display paginated predictions for a specific user with related loan applications.
    
    Shows predictions with:
    - Application details
    - Prediction probability
    - Related formal loan applications
    - Pagination (5 predictions per page)
    
    Args:
        request (HttpRequest): Django request object.
                              GET param 'page' for pagination.
        user_id (int): User ID to retrieve predictions for.
        
    Returns:
        HttpResponse: User predictions page with pagination.
        
    Security:
        Requires admin authentication.
    """
    user = get_object_or_404(User, id=user_id)
    predictions = LoanApplication.objects.filter(user=user).select_related('prediction').order_by('-created_at')
    
    # Get threshold for display
    config = PredictionConfig.objects.first()
    threshold = config.threshold if config else 0.65
    
    # Pagination
    paginator = Paginator(predictions, 5)  # 5 predictions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Build enhanced data for display
    prediction_data = []
    for app in page_obj:
        loans = LoanApplicationFormData.objects.filter(user=user, prediction=app.prediction)
        prediction_data.append({
            'application': app,
            'loans': loans
        })
    
    logger.info(f"Admin {request.user.username} viewed {len(predictions)} predictions for user {user.username}")
    
    return render(request, 'admin_user_predictions.html', {
        'user': user,
        'prediction_data': prediction_data,
        'threshold': threshold,
        'page_obj': page_obj
    })

@user_passes_test(admin_required, login_url='admin_login')
def search_users_view(request):
    query = request.GET.get('q', '')  # Get search query
    users = User.objects.filter(is_staff=False)  # Only non-staff users

    if query:
        users = users.filter(
            Q(username__icontains=query) |
            Q(email__icontains=query) |
            Q(first_name__icontains=query) |
            Q(last_name__icontains=query)
        )

    return render(request, 'search_users.html', {'users': users, 'query': query})


@user_passes_test(admin_required, login_url='admin_login')
def bd_users_reports(request):
    fdate = request.GET.get('fdate')
    tdate = request.GET.get('tdate')
    users = User.objects.filter(is_staff=False)  # only non-staff

    # Apply date filter if both dates are provided
    if fdate and tdate:
        fdate_parsed = parse_date(fdate)
        tdate_parsed = parse_date(tdate)

        if fdate_parsed and tdate_parsed:
            users = users.filter(date_joined__date__range=[fdate_parsed, tdate_parsed])

    return render(request, 'users_reports.html', {
        'users': users,
        'fdate': fdate,
        'tdate': tdate
    })


def admin_reset_password(request):
    if request.method == "POST":
        email = request.POST.get('email')
        new_password = request.POST.get('newpassword')
        confirm_password = request.POST.get('confirmpassword')

        if new_password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('admin_reset_password')

        try:
            user = User.objects.get(email=email, is_staff=True)  # only staff/admin
            user.password = make_password(new_password)
            user.save()
            messages.success(request, "Password changed successfully.")
            return redirect('admin_login')
        except User.DoesNotExist:
            messages.error(request, "Invalid email or not an admin.")
    
    return render(request, "reset_password.html")