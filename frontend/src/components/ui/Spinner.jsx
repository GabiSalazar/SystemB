import { cn } from '../../utils/cn'

export function Spinner({ size = 'md', className }) {
  const sizes = {
    sm: "w-4 h-4 border-2",
    md: "w-8 h-8 border-3",
    lg: "w-12 h-12 border-4"
  }

  return (
    <div
      className={cn(
        "inline-block border-blue-600 border-t-transparent rounded-full animate-spin",
        sizes[size],
        className
      )}
    />
  )
}