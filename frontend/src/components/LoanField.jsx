function LoanField({ field }) {
  return (
    <div className="loan-form-field">
      <label htmlFor={field.name}>
        {field.label}
      </label>

      {field.type === 'select' ? (
        <select name={field.name} id={field.name} defaultValue="" required>
          <option value="" disabled>
            Select {field.label}
          </option>
          {field.options.map((opt) => {
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