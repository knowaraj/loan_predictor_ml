/**
 * Main React Application Component
 * 
 * Root component that renders the loan prediction form.
 * Receives CSRF token and API action endpoint from Django template context.
 * 
 * @component
 * @param {Object} props - Component props
 * @param {string} props.csrfToken - Django CSRF token for form submission security
 * @param {string} props.action - API endpoint URL for form POST requests
 * @returns {JSX.Element} App component with LoanPredictionForm
 * 
 * @example
 * // Rendered from Django template with:
 * // <App csrfToken={csrf_token} action={"/predict/"} />
 */
import LoanPredictionForm from './components/LoanPredictionForm'

function App({ csrfToken, action }) {
  return <LoanPredictionForm csrfToken={csrfToken} action={action} />
}

export default App
