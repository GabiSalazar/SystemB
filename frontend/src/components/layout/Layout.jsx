import { Link, useLocation } from 'react-router-dom'
import { Home, UserPlus, Shield, Activity } from 'lucide-react'
import { cn } from '../../utils/cn'

export default function Layout({ children }) {
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Registro', href: '/enrollment', icon: UserPlus },
    { name: 'Autenticación', href: '/authentication', icon: Shield },
    { name: 'Sistema', href: '/system', icon: Activity },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="text-3xl">🖐️</div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Sistema Biométrico</h1>
                <p className="text-xs text-gray-500">Autenticación por Gestos</p>
              </div>
            </div>

            <nav className="flex gap-1">
              {navigation.map((item) => {
                const Icon = item.icon
                const isActive = location.pathname === item.href
                
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={cn(
                      "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                      isActive
                        ? "bg-blue-100 text-blue-700"
                        : "text-gray-700 hover:bg-gray-100"
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  )
}