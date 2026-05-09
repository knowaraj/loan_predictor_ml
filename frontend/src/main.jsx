import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

const mountNode = document.getElementById('react-loan-form')

if (mountNode) {
  createRoot(mountNode).render(
    <App
      csrfToken={mountNode.dataset.csrf || ''}
      action={mountNode.dataset.action || '/predict/'}
    />,
  )
}
