/**
 * Loan Prediction Form Component
 * 
 * Renders the main loan prediction form with all borrower information fields.
 * Submits data to Django backend for ML-based loan default prediction.
 * 
 * Features:
 * - Form validation via HTML5
 * - CSRF token integration for security
 * - Dynamic field rendering using LOAN_FORM_FIELDS configuration
 * - Professional UI with header and submit button
 * 
 * @component
 * @param {Object} props - Component props
 * @param {string} props.csrfToken - Django CSRF token for form submission
 * @param {string} props.action - Form POST endpoint URL (e.g., "/predict/")
 * @returns {JSX.Element} Rendered loan form with styled container
 * 
 * @example
 * <LoanPredictionForm 
 *   csrfToken="token123" 
 *   action="/predict/"
 * />
 */
import LoanField from './LoanField'
import { LOAN_FORM_FIELDS } from './loanFormFields'
import './LoanPredictionForm.css'

function LoanPredictionForm({ csrfToken, action }) {
  return (
    <div className="loan-form-wrapper">
      <div className="loan-form-container">
        <div className="loan-form-header">
          <h2>Loan Prediction</h2>
          <p className="loan-form-subtitle">
            Fill in your details to estimate loan eligibility
          </p>
        </div>

        <form method="post" action={action} className="loan-form">
          {/* Django CSRF token for security */}
          <input type="hidden" name="csrfmiddlewaretoken" value={csrfToken} />

          <div className="loan-form-fields">
            {/* Dynamically render form fields */}
            {LOAN_FORM_FIELDS.map((field) => (
              <LoanField key={field.name} field={field} />
            ))}
          </div>

          <button type="submit" className="loan-form-submit">
            Get Prediction
          </button>
        </form>
      </div>
    </div>
  )
}

export default LoanPredictionForm