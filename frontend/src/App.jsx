import LoanPredictionForm from './components/LoanPredictionForm'

function App({ csrfToken, action }) {
  return <LoanPredictionForm csrfToken={csrfToken} action={action} />
}

export default App
