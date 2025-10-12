import { useState, useEffect } from 'react'
import { systemApi } from '../../lib/api/system'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Badge, Spinner, Button } from '../../components/ui'
import { Activity, Users, Brain, Shield, Clock, AlertCircle } from 'lucide-react'

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadSystemStatus()
    const interval = setInterval(loadSystemStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadSystemStatus = async () => {
    try {
      const data = await systemApi.getStatus()
      setSystemStatus(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleTrainNetworks = async () => {
    if (!confirm('¿Entrenar las redes neuronales ahora?')) return
    
    try {
      setLoading(true)
      await systemApi.trainNetworks()
      alert('Redes entrenadas exitosamente')
      loadSystemStatus()
    } catch (error) {
      alert('Error entrenando redes: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spinner size="lg" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Panel de control del sistema biométrico</p>
      </div>

      {/* Cards de Estadísticas */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Estado General */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Estado</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {systemStatus?.status === 'operational' ? 'Operativo' : 'Error'}
                </p>
              </div>
              <div className={`p-3 rounded-full ${systemStatus?.status === 'operational' ? 'bg-green-100' : 'bg-red-100'}`}>
                <Activity className={`w-6 h-6 ${systemStatus?.status === 'operational' ? 'text-green-600' : 'text-red-600'}`} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Usuarios */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Usuarios</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {systemStatus?.users_count || 0}
                </p>
              </div>
              <div className="p-3 rounded-full bg-blue-100">
                <Users className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Redes Neuronales */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Redes</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {systemStatus?.networks_trained ? 'Entrenadas' : 'Pendiente'}
                </p>
              </div>
              <div className={`p-3 rounded-full ${systemStatus?.networks_trained ? 'bg-green-100' : 'bg-yellow-100'}`}>
                <Brain className={`w-6 h-6 ${systemStatus?.networks_trained ? 'text-green-600' : 'text-yellow-600'}`} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Autenticación */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Autenticación</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {systemStatus?.authentication_active ? 'Activa' : 'Inactiva'}
                </p>
              </div>
              <div className={`p-3 rounded-full ${systemStatus?.authentication_active ? 'bg-green-100' : 'bg-gray-100'}`}>
                <Shield className={`w-6 h-6 ${systemStatus?.authentication_active ? 'text-green-600' : 'text-gray-600'}`} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Información Detallada */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Estado del Sistema */}
        <Card>
          <CardHeader>
            <CardTitle>Estado del Sistema</CardTitle>
            <CardDescription>Información detallada del sistema</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Nivel de Inicialización</span>
              <Badge variant="info">{systemStatus?.initialization_level}</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Base de Datos</span>
              <Badge variant={systemStatus?.database_ready ? 'success' : 'danger'}>
                {systemStatus?.database_ready ? 'Lista' : 'No disponible'}
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Enrollment</span>
              <Badge variant={systemStatus?.enrollment_active ? 'success' : 'danger'}>
                {systemStatus?.enrollment_active ? 'Activo' : 'Inactivo'}
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-gray-600" />
                <span className="text-sm font-medium">Tiempo Activo</span>
              </div>
              <span className="font-mono text-sm">{systemStatus?.uptime}</span>
            </div>
          </CardContent>
        </Card>

        {/* Acciones Rápidas */}
        <Card>
          <CardHeader>
            <CardTitle>Acciones Rápidas</CardTitle>
            <CardDescription>Operaciones del sistema</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {!systemStatus?.networks_trained && systemStatus?.users_count >= 2 && (
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-yellow-900 mb-1">
                      Redes sin entrenar
                    </h4>
                    <p className="text-sm text-yellow-700 mb-3">
                      Hay {systemStatus.users_count} usuarios registrados. Se pueden entrenar las redes.
                    </p>
                    <Button 
                      size="sm" 
                      onClick={handleTrainNetworks}
                      disabled={loading}
                    >
                      {loading ? 'Entrenando...' : 'Entrenar Redes Ahora'}
                    </Button>
                  </div>
                </div>
              </div>
            )}

            {systemStatus?.users_count < 2 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-blue-900 mb-1">
                      Registra usuarios
                    </h4>
                    <p className="text-sm text-blue-700">
                      Se necesitan al menos 2 usuarios para entrenar las redes neuronales.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="pt-2">
              <p className="text-xs text-gray-500 mb-2">Versión del Sistema</p>
              <Badge variant="primary">{systemStatus?.version}</Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}