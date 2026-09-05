/**
 * Loan Form Fields Configuration
 * 
 * Defines all form fields required for loan default prediction.
 * These fields correspond to the ML model's input features and are
 * sent to Django backend for prediction.
 * 
 * @typedef {Object} FormField
 * @property {string} name - HTML input name (maps to Django form field)
 * @property {string} label - User-facing field label
 * @property {string} type - Field type ('select' for dropdowns, 'number' for inputs, etc.)
 * @property {string[]|Object[]} options - Available options for select fields
 * @property {string} [placeholder] - Placeholder text for input fields
 * @property {number} [min] - Minimum value for number inputs
 * 
 * Field Mapping to ML Model:
 * - Categorical: gender, married, dependents, education, self_employed, credit_history, property_area
 * - Numerical: applicant_income, coapplicant_income, loan_amount, loan_amount_term
 * 
 * Form Order:
 * 1. Demographics (Gender, Marital, Dependents, Education)
 * 2. Employment (Self-employed)
 * 3. Financial (Incomes, Loan Amount, Term)
 * 4. Credit (Credit History)
 * 5. Property (Area)
 */

/**
 * Array of loan form field configurations
 * @type {FormField[]}
 */
export const LOAN_FORM_FIELDS = [
  // ===== Demographics =====
  {
    name: 'gender',
    label: 'Gender',
    type: 'select',
    options: ['Male', 'Female'],
  },
  {
    name: 'married',
    label: 'Married',
    type: 'select',
    options: ['Yes', 'No'],
  },
  {
    name: 'dependents',
    label: 'Dependents',
    type: 'select',
    options: ['0', '1', '2', '3+'],
  },
  {
    name: 'education',
    label: 'Education',
    type: 'select',
    options: ['Graduate', 'Not Graduate'],
  },

  // ===== Employment =====
  {
    name: 'self_employed',
    label: 'Self Employed',
    type: 'select',
    options: ['Yes', 'No'],
  },

  // ===== Financial Information =====
  {
    name: 'applicant_income',
    label: 'Applicant Income',
    type: 'number',
    placeholder: 'e.g. 5000',
    min: 0,
  },
  {
    name: 'coapplicant_income',
    label: 'Coapplicant Income',
    type: 'number',
    placeholder: 'e.g. 2000',
    min: 0,
  },
  {
    name: 'loan_amount',
    label: 'Loan Amount',
    type: 'number',
    placeholder: 'e.g. 120',
    min: 0,
  },
  {
    name: 'loan_amount_term',
    label: 'Loan Amount Term',
    type: 'number',
    placeholder: 'e.g. 360',
    min: 0,
  },

  // ===== Credit & Property =====
  {
    name: 'credit_history',
    label: 'Credit History',
    type: 'select',
    options: [
      { value: '1', text: 'Good (1)' },
      { value: '0', text: 'Bad (0)' },
    ],
  },
  {
    name: 'property_area',
    label: 'Property Area',
    type: 'select',
    options: ['Urban', 'Semiurban', 'Rural'],
  },
]