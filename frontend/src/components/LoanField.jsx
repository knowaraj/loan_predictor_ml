/**
 * Individual Loan Form Field Component
 * 
 * Renders a single form field (input or select) based on field configuration.
 * Handles both text inputs (number fields for income, loan amount) and
 * dropdown selects (gender, education, etc.).
 * 
 * @component
 * @param {Object} props - Component props
 * @param {Object} props.field - Field configuration object
 * @param {string} props.field.name - HTML input name attribute (maps to Django form field)
 * @param {string} props.field.label - Display label for the field
 * @param {string} props.field.type - Input type ('select', 'number', 'text', etc.)
 * @param {string[]} [props.field.options] - Array of select options (for select type)
 * @param {string} [props.field.placeholder] - Placeholder text for input fields
 * @param {number} [props.field.min] - Minimum value for number inputs
 * @returns {JSX.Element} Rendered form field with label and input/select
 * 
 * Field Types:
 * - 'select': Renders <select> dropdown with options
 * - 'number': Renders <input type="number"> for numeric values
 * - Other: Renders <input> with specified type
 * 
 * @example
 * // Select field
 * <LoanField field={{
 *   name: 'gender',
 *   label: 'Gender',
 *   type: 'select',
 *   options: ['Male', 'Female']
 * }} />
 * 
 * // Number input field
 * <LoanField field={{
 *   name: 'applicant_income',
 *   label: 'Applicant Income',
 *   type: 'number',
 *   placeholder: 'e.g. 5000',
 *   min: 0
 * }} />
 */
function LoanField({ field }) {
  return (
    <div className="loan-form-field">
      <label htmlFor={field.name}>
        {field.label}
      </label>

      {field.type === 'select' ? (
        // Render select dropdown
        <select name={field.name} id={field.name} defaultValue="" required>
          <option value="" disabled>
            Select {field.label}
          </option>
          {field.options.map((opt) => {
            // Handle both simple string options and object options
            const value = typeof opt === 'string' ? opt : opt.value
            const text = typeof opt === 'string' ? opt : opt.text
            return (
              <option key={value} value={value}>
                {text}
              </option>
            )
          })}
        </select>
      ) : (
        // Render text/number input
        <input
          type={field.type}
          name={field.name}
          id={field.name}
          placeholder={field.placeholder || ''}
          min={field.min}
          step="any"
          required
        />
      )}
    </div>
  )
}

export default LoanField