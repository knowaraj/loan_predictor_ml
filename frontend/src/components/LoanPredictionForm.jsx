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
          <input type="hidden" name="csrfmiddlewaretoken" value={csrfToken} />

          <div className="loan-form-fields">
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