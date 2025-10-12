export function Spinner({ size = 'md', className = '' }) {
  const sizeMap = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4'
  }

  return (
    <div className={`inline-block ${sizeMap[size]} border-blue-600 border-t-transparent rounded-full animate-spin ${className}`} />
  )
}