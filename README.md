# Loan Predictor Application

A full-stack web application that uses machine learning to predict loan default risk and provides an admin panel for managing applications and users.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Database Models](#database-models)
- [Default Credentials](#default-credentials)
- [Project Structure](#project-structure)

## Features

✅ **ML-Powered Predictions**
- XGBoost-based loan default prediction model
- SHAP explainability for interpretable predictions
- Configurable prediction threshold

✅ **User Features**
- User registration and authentication
- Loan application form with validation
- Prediction history tracking
- Profile management and password reset

✅ **Admin Panel**
- Dashboard with key metrics (pending, approved, rejected)
- Loan application review and approval workflow
- User management and profile viewing
- Prediction analytics and reporting
- Bulk prediction management

✅ **Frontend**
- React-based modern UI using Vite
- Responsive form components
- Real-time prediction results

## Technology Stack

**Backend:**
- Django 5.2.5 (Web Framework)
- Python 3.12+ (Language)
- SQLite3 (Development Database)
- XGBoost + SHAP (ML Pipeline)
- Pandas & NumPy (Data Processing)

**Frontend:**
- React 18+ (UI Library)
- Vite (Build Tool)
- CSS3 (Styling)
- JavaScript ES6+ (Language)

**DevOps:**
- Virtual Environment (Python isolation)
- Joblib (Model serialization)

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React/Vite)                │
│            (Loan Form, User Dashboard, Admin UI)        │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST
┌──────────────────▼──────────────────────────────────────┐
│              Django Backend (Views + URLs)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ predictions  │  │ admin_panel  │  │ loan_predictor│ │
│  │   (User App) │  │  (Admin App) │  │  (Settings)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
┌─────▼────┐ ┌─────▼────┐ ┌───▼──────┐
│ Models   │ │  ML Ops  │ │ Database │
│ (Django) │ │(predictor)│ │ (SQLite) │
└──────────┘ └──────────┘ └──────────┘
      │            │
      └────────────┼────────────┘
                   │
            ┌──────▼──────┐
            │  ML Models  │
            │  (XGBoost + │
            │   SHAP)     │
            └─────────────┘
```

### Module Responsibilities

**predictions/** (Main User App)
- User authentication and registration
- Loan application form processing
- ML prediction orchestration
- User dashboard and history

**admin_panel/** (Admin App)
- Admin authentication and authorization
- Loan review workflow (approve/reject)
- User management
- Prediction analytics

**ml/** (ML Pipeline)
- `predictor.py`: Inference engine with SHAP explainability
- `train.py`: Model training script (reference)
- Models stored as joblib serialized objects

## Prerequisites

- Python 3.12+
- `pip` (Python package manager)
- `python3.12-venv` package (on Linux) - optional
- Git (for version control)

## Getting Started

### 1. Create and Activate Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Database Migrations

```bash
python manage.py migrate
```

This creates the SQLite database and applies all Django migrations.

### 4. Start the Development Server

```bash
python manage.py runserver 0.0.0.0:8000
```

The application will be available at [http://localhost:8000/](http://localhost:8000/)

### Windows Quick Start (Combined)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## API Documentation

### Prediction Endpoint

**User Loan Prediction Form**
- **URL:** `/predict/`
- **Method:** POST
- **Authentication:** Required (Login)
- **Description:** Submits loan application and receives ML prediction

**Request Body (Form Data):**
```
Gender: str (Male/Female)
Married: str (Yes/No)
Dependents: str (0-3+)
Education: str (Graduate/Undergraduate)
Self_Employed: str (Yes/No)
ApplicantIncome: float (monthly income)
CoapplicantIncome: float (co-applicant income)
LoanAmount: float (requested amount)
Loan_Amount_Term: float (months)
Credit_History: int (0 or 1)
Property_Area: str (Urban/Rural/Semiurban)
```

**Response:**
- Status: 200 OK
- Template: `result.html` with prediction details
- Data includes: prediction probability, default likelihood, top 5 influencing factors

### Admin Endpoints

**Loan Review**
- **URL:** `/admin-panel/loans/`
- **Method:** GET/POST
- **Authentication:** Admin only
- **Description:** List and manage loan applications

**User Management**
- **URL:** `/admin-panel/users/`
- **Method:** GET
- **Authentication:** Admin only
- **Description:** View all registered users

**Predictions Dashboard**
- **URL:** `/admin-panel/predictions/`
- **Method:** GET
- **Authentication:** Admin only
- **Description:** View all predictions and their results

## Database Models

### LoanApplication
Stores initial loan prediction form submissions.
```
- user: ForeignKey(User)
- gender, married, dependents, education, self_employed
- applicant_income, coapplicant_income
- loan_amount, loan_amount_term, credit_history
- property_area
- created_at: DateTime
```

### PredictionResult
Stores ML prediction output for each application.
```
- application: OneToOneField(LoanApplication)
- predicted_default: Boolean
- probability: Float (0.0-1.0)
- model_version: String
- created_at: DateTime
```

### LoanApplicationFormData
Tracks formal loan applications going through approval workflow.
```
- user: ForeignKey(User)
- prediction: ForeignKey(PredictionResult)
- loan_amount, loan_tenure, loan_purpose
- status: Choice(Pending/Approved/Rejected)
- remark: TextField
- applied_at: DateTime
```

### UserProfile
Extended user information (addresses, contact).
```
- user: OneToOneField(User)
- mobile_number: String
- address: TextField
```

### PredictionConfig
System-wide prediction settings.
```
- threshold: Float (default 0.65)
  - Probability above which applicant is classified as defaulter
```

## React Frontend Setup

The `/predict/` page is rendered by a React app bundled with Vite.

### Development Workflow

1. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Build React app:**
   ```bash
   npm run build
   ```

3. **Start Django server from project root:**
   ```bash
   cd ..
   python manage.py runserver 0.0.0.0:8000
   ```

4. **Access prediction form:**
   Navigate to `/predict/` in your browser

The build output is generated in `predictions/static/predictions/react/` and loaded by `predictions/templates/predictions/form.html`.

## Default Credentials

### Admin Account
- **Username:** `admin`
- **Password:** `admin@123`
- **URL:** [http://localhost:8000/admin-panel/](http://localhost:8000/admin-panel/)

### Standard User Account
- **Username:** `john12`
- **Password:** `user123`
- **URL:** [http://localhost:8000/](http://localhost:8000/)

## Project Structure

```
loan_predictor_ml/
├── db.sqlite3                          # Development database
├── manage.py                           # Django CLI
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── loan_predictor/                     # Django project settings
│   ├── settings.py                     # Configuration
│   ├── urls.py                         # URL routing
│   ├── wsgi.py                         # Production deployment
│   └── asgi.py                         # Async support
│
├── predictions/                        # Main user-facing app
│   ├── models.py                       # Database models
│   ├── views.py                        # View handlers
│   ├── urls.py                         # App URLs
│   ├── forms.py                        # Form definitions
│   ├── predictor.py                    # ML inference engine
│   ├── migrations/                     # Database migrations
│   ├── static/predictions/             # Static assets
│   └── templates/predictions/          # Django templates
│
├── admin_panel/                        # Admin dashboard app
│   ├── models.py                       
│   ├── views.py                        # Admin view handlers
│   ├── urls.py                         
│   ├── forms.py                        
│   └── templates/                      # Admin templates
│
├── frontend/                           # React application
│   ├── package.json                    # NPM dependencies
│   ├── vite.config.js                  # Vite bundler config
│   ├── src/
│   │   ├── App.jsx                     # React app entry
│   │   ├── components/
│   │   │   ├── LoanPredictionForm.jsx  # Main form component
│   │   │   ├── LoanField.jsx           # Field component
│   │   │   └── loanFormFields.js       # Field configuration
│   │   └── assets/                     # Images, icons
│   └── public/                         # Public assets
│
└── ml/                                 # Machine learning
    ├── train.py                        # Model training script
    ├── train2.py                       # Alternate training
    ├── exp.ipynb                       # Experimentation notebook
    ├── data/                           # Training datasets
    └── models/                         # Serialized ML models
        ├── pipeline_xgb.joblib         # XGBoost pipeline
        ├── shap_explainer.joblib       # SHAP explainer
        └── feature_names.joblib        # Feature list
```

## Logging

The application includes logging for debugging and monitoring:
- **Location:** Logs sent to console and optionally to file
- **Format:** Timestamp, logger name, level, message
- **Modules:** predictor, views, admin_panel tracked

Configure in `loan_predictor/settings.py` LOGGING section.

## Error Handling

- **Form Validation:** Comprehensive client and server-side validation
- **Database Errors:** Try-catch blocks with user-friendly messages
- **ML Errors:** Fallback values and logging for model issues
- **Authentication:** Login required decorators on protected views

## Next Steps for Production

- [ ] Set DEBUG=False and configure allowed hosts
- [ ] Use PostgreSQL instead of SQLite
- [ ] Implement CI/CD pipeline (GitHub Actions)
- [ ] Add Docker containerization
- [ ] Set up monitoring and alerting
- [ ] Configure HTTPS/SSL certificates
- [ ] Implement automated testing
- [ ] Add API documentation (Swagger/DRF)
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
