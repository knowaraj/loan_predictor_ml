# Loan Predictor Application

A Django-based web application that predicts loan eligibility and provides an admin panel for managing applications and users.

## Prerequisites
- Python 3.12+
- `python3.12-venv` package (on Linux)

## Getting Started

Follow these steps to set up and run the project locally:

1. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Database Migrations**
   ```bash
   python manage.py migrate
   <!-- No migration  paxii -->
   ```

4. **Start the Development Server**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```
   The site will be available at [http://localhost:8000/](http://localhost:8000/).

## Default Credentials

If you need to log in to test the application, default accounts have been configured:

### Admin Account
Use this to access the custom Admin Panel (`/admin-panel/`) or the default Django Admin (`/admin/`).
- **Username:** `admin`
- **Password:** `admin123`

### Standard User Account
Use this to test the regular user loan application flow.
- **Username:** `john12`
- **Password:** `user123`

## Project Structure
- `loan_predictor/` - Core Django settings and main URL routing.
- `predictions/` - The main user-facing app (registration, loan application, and prediction history).
- `admin_panel/` - A custom dashboard for administrators to review users and applications.
