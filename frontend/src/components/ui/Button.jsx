export function Button(props) {
  const { 
    children, 
    variant = 'primary', 
    className = '',
    disabled = false,
    ...rest 
  } = props

  let variantClass = 'bg-blue-600 text-white hover:bg-blue-700'
  
  if (variant === 'danger') {
    variantClass = 'bg-red-600 text-white hover:bg-red-700'
  } else if (variant === 'secondary') {
    variantClass = 'bg-gray-200 text-gray-900 hover:bg-gray-300'
  }

  const finalClass = `inline-flex items-center justify-center px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 ${variantClass} ${className}`

  return (
    <button
      type="button"
      disabled={disabled}
      className={finalClass}
      {...rest}
    >
      {children}
    </button>
  )
}