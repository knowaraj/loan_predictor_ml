# Logging Configuration Guide

## Overview

The Loan Predictor application includes comprehensive logging to track application behavior, debug issues, and monitor system health.

## Current Setup

Logging is configured in `loan_predictor/settings.py` with the following:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {name} - {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'loan_predictor.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'predictions': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'admin_panel': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'predictions.predictor': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}
```

## Logged Information by Module

### predictions.predictor (ML Inference)
- **Level:** DEBUG (for detailed ML pipeline information)
- **Logs:**
  - Model loading status and paths
  - Feature count verification
  - Prediction probabilities and results
  - SHAP explanation generation
  - Errors during inference

Example:
```
DEBUG 2026-05-10 12:34:56 predictor 1234 5678 Loading ML model from /path/to/pipeline_xgb.joblib
INFO 2026-05-10 12:34:57 predictor 1234 5678 ML model loaded successfully
INFO 2026-05-10 12:34:58 predictor 1234 5678 Prediction created: 42 - Probability: 0.7234
```

### predictions.views (User App)
- **Level:** INFO
- **Logs:**
  - User registration and authentication
  - Loan application submissions
  - Prediction results
  - Profile updates
  - Password changes
  - User error messages

Example:
```
INFO 2026-05-10 12:35:00 views 1234 5678 New user registered: john_doe
INFO 2026-05-10 12:35:02 views 1234 5678 Loan application saved: 15 by user john_doe
```

### admin_panel.views (Admin Dashboard)
- **Level:** INFO
- **Logs:**
  - Admin login/logout
  - Loan review decisions
  - User management actions (view, delete, update)
  - Dashboard access
  - Prediction deletions

Example:
```
INFO 2026-05-10 12:36:00 views 1234 5678 Admin user logged in: admin
WARNING 2026-05-10 12:36:05 views 1234 5678 User john_doe deleted by admin admin
```

### django.db.backends
- **Level:** DEBUG (development) / INFO (production)
- **Logs:**
  - All database queries and execution time
  - Connection details

## Log Levels

- **DEBUG:** Detailed information for diagnosing issues (feature counts, query execution time)
- **INFO:** General informational messages (user actions, successful operations)
- **WARNING:** Something unexpected happened but the app continues
- **ERROR:** Serious problem, partial failure possible
- **CRITICAL:** Very serious error, possible system failure

## Log Output Locations

### Console Output
- Displayed in terminal when running `python manage.py runserver`
- Useful for real-time monitoring during development

### File Output
- **Location:** `loan_predictor.log` (root directory)
- **Format:** Verbose with process/thread IDs
- **Use case:** Historical audit trail and production monitoring

Example file entry:
```
INFO 2026-05-10 12:35:02,123 predictions.views 45678 89012 Loan application saved: 15 by user john_doe
```

## Production Recommendations

### 1. Increase Log Retention
```python
# Use RotatingFileHandler to manage log file size
'file': {
    'class': 'logging.handlers.RotatingFileHandler',
    'filename': 'loan_predictor.log',
    'maxBytes': 10485760,  # 10 MB
    'backupCount': 5,
    'formatter': 'verbose',
},
```

### 2. Reduce Debug Logging in Production
```python
# Change DEBUG level from DEBUG to INFO
if DEBUG:
    LOGGING['loggers']['django.db.backends']['level'] = 'DEBUG'
else:
    LOGGING['loggers']['django.db.backends']['level'] = 'WARNING'
```

### 3. Add Syslog or Cloud Logging
```python
# For syslog integration (Linux)
'syslog': {
    'class': 'logging.handlers.SysLogHandler',
    'address': '/dev/log',
    'formatter': 'verbose',
},

# For Google Cloud Logging, AWS CloudWatch, etc.
# Install appropriate handlers and add to handlers dict
```

### 4. Add Error Notifications
```python
# Email admin on ERROR level
'mail_admins': {
    'class': 'django.utils.log.AdminEmailHandler',
    'level': 'ERROR',
},
```

## Usage in Code

### Getting a Logger
```python
import logging
logger = logging.getLogger(__name__)

# Log messages
logger.info("User login: john_doe")
logger.warning("Unusual activity detected")
logger.error("Database connection failed", exc_info=True)
```

### With Exception Context
```python
try:
    prediction = predict_with_explanations(input_dict)
except Exception as e:
    logger.error(f"Prediction error: {str(e)}", exc_info=True)
    # exc_info=True includes full stack trace
```

## Viewing Logs

### Real-time Console
```bash
python manage.py runserver 0.0.0.0:8000
# Logs appear in terminal
```

### From File
```bash
# View last 100 lines
tail -100 loan_predictor.log

# Follow in real-time
tail -f loan_predictor.log

# Search for errors
grep ERROR loan_predictor.log

# Count by level
grep -c "INFO" loan_predictor.log
grep -c "ERROR" loan_predictor.log
```

## Troubleshooting

### No Logs Appearing
- Check `DEBUG = True` in settings.py
- Verify logger name matches module (e.g., `predictions.views`)
- Ensure handler is configured for that logger

### Log File Not Created
- Check file permissions in root directory
- Ensure `loan_predictor/` directory is writable

### Performance Issues
- High volume of DEBUG logs can slow down application
- In production, use `INFO` level instead of `DEBUG`
- Consider rotating log files to prevent excessive disk usage

## Best Practices

✅ **DO:**
- Log important business events (registrations, predictions, approvals)
- Include user context where relevant
- Use appropriate log levels
- Include stack traces for exceptions

❌ **DON'T:**
- Log sensitive information (passwords, credit card numbers)
- Log every line of code execution (use DEBUG level strategically)
- Leave DEBUG=True in production
- Ignore ERROR logs in production

## Related Documentation

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Django Logging](https://docs.djangoproject.com/en/5.2/topics/logging/)
- Main [README.md](README.md) for project overview
