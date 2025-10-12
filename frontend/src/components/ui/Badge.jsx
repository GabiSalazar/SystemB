export function Badge({ 
  children, 
  variant = 'default', 
  className = '' 
}) {
  const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
  
  const variantClasses = {
    default: "bg-gray-100 text-gray-800",
    primary: "bg-blue-100 text-blue-800",
    success: "bg-green-100 text-green-800",
    danger: "bg-red-100 text-red-800",
    warning: "bg-yellow-100 text-yellow-800",
    info: "bg-cyan-100 text-cyan-800"
  }
  
  const classes = [
    baseClasses,
    variantClasses[variant] || variantClasses.default,
    className
  ].filter(Boolean).join(' ')
  
  return (
    <span className={classes}>
      {children}
    </span>
  )
}