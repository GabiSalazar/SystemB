import { useState, useEffect } from 'react'
import { systemApi } from '../../lib/api/system'
import { enrollmentApi } from '../../lib/api/enrollment'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Badge, Spinner, Button } from '../../components/ui'
import { Activity, Users, Brain, Shield, Clock, AlertCircle, Zap } from 'lucide-react'

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState(null)
  const [bootstrapStatus, setBootstrapStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [system, bootstrap] = await Promise.all([
        systemApi.getStatus(),
        enrollmentApi.getBootstrapStatus()
      ])
      
      console.log('System Status:', system)
      console.log('Bootstrap Status:', bootstrap)
      
      setSystemStatus(system)
      setBootstrapStatus(bootstrap)
      setError(null)
    } catch (err) {
      console.error('Error cargando datos del dashboard:', err)
      setError('Error al cargar datos del sistema')
    } finally {
      setLoading(false)
    }
  }

  const handleTrainNetworks = async () => {
    if (!window.confirm('¿Entrenar las redes neuronales con los usuarios actuales?\n\nEste proceso puede tardar 2-5 minutos.')) {
      return
    }
    
    try {
      setTraining(true)
      console.log('Iniciando entrenamiento de redes...')
      
      const result = await systemApi.retrainNetworks(true)
      
      console.log('Resultado del entrenamiento:', result)
      
      if (result.success) {
        alert('Redes entrenadas exitosamente!\n\nEl sistema ahora está en modo normal.')
        await loadData()
      } else {
        alert('Error: ' + result.message)
      }
    } catch (error) {
      console.error('Error entrenando redes:', error)
      const errorMsg = error.response?.data?.detail || error.message || 'Error desconocido'
      alert('Error entrenando redes:\n\n' + errorMsg)
    } finally {
      setTraining(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spinner size="lg" />
      </div>
    )
  }

  // Determinar si se puede entrenar
  const canTrain = systemStatus?.can_train && !systemStatus?.networks_trained

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Panel de control del sistema biométrico</p>
      </div>

      {/* Error Alert */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertCircle className="w-6 h-6 text-red-600" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-red-900 mb-1">Error</h3>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ALERTA: Necesita entrenamiento */}
      {canTrain && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <AlertCircle className="w-6 h-6 text-yellow-600" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 mb-1">
                  Sistema listo para entrenamiento
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Tienes {systemStatus?.users_count || 0} usuarios registrados. 
                  Es necesario entrenar las redes neuronales para activar el sistema completo.
                </p>
                <Button 
                  onClick={handleTrainNetworks}
                  disabled={training}
                  className="bg-yellow-600 hover:bg-yellow-700"
                >
                  {training ? (
                    <>
                      <Spinner size="sm" className="mr-2" />
                      Entrenando... (esto puede tardar minutos)
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4 mr-2" />
                      Entrenar Redes Ahora
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Cards de Estadísticas */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Estado General */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Estado</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {systemStatus?.status === 'operational' ? 'Operativo' : 'Parcial'}
                </p>
              </div>
              <div className={`p-3 rounded-full ${systemStatus?.status === 'operational' ? 'bg-green-100' : 'bg-yellow-100'}`}>
                <Activity className={`w-6 h-6 ${systemStatus?.status === 'operational' ? 'text-green-600' : 'text-yellow-600'}`} />
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
              <Badge variant="info">{systemStatus?.initialization_level || 'N/A'}</Badge>
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
              <span className="text-sm font-medium">Modo</span>
              <Badge variant={systemStatus?.bootstrap_mode ? 'warning' : 'success'}>
                {systemStatus?.bootstrap_mode ? 'Bootstrap' : 'Normal'}
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-gray-600" />
                <span className="text-sm font-medium">Tiempo Activo</span>
              </div>
              <span className="font-mono text-sm">{systemStatus?.uptime || '0h 0m 0s'}</span>
            </div>
          </CardContent>
        </Card>

        {/* Estadísticas */}
        <Card>
          <CardHeader>
            <CardTitle>Estadísticas</CardTitle>
            <CardDescription>Métricas del sistema</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Versión</span>
              <Badge variant="primary">{systemStatus?.version || '2.0.0'}</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Templates Bootstrap</span>
              <span className="text-sm font-bold text-gray-700">
                {bootstrapStatus?.templates_count || 0}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium">Usuarios Mínimos</span>
              <span className="text-sm font-bold text-gray-700">
                {bootstrapStatus?.min_users_required || 2}
              </span>
            </div>

            {/* Mensaje de Bootstrap */}
            {bootstrapStatus?.message && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg mt-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <p className="text-sm text-blue-800 font-medium">
                      {bootstrapStatus.message}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Info: Necesita más usuarios */}
            {systemStatus?.users_count < 2 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg mt-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-blue-900 mb-1">
                      Registra más usuarios
                    </h4>
                    <p className="text-sm text-blue-700">
                      Se necesitan al menos 2 usuarios para entrenar las redes neuronales.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Info: Puede entrenar */}
            {canTrain && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg mt-4">
                <div className="flex items-start gap-3">
                  <Brain className="w-5 h-5 text-green-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-green-900 mb-1">
                      Listo para entrenar
                    </h4>
                    <p className="text-sm text-green-700">
                      Tienes suficientes usuarios registrados. Haz clic en "Entrenar Redes Ahora" arriba.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Info: Sistema operativo */}
            {systemStatus?.networks_trained && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg mt-4">
                <div className="flex items-start gap-3">
                  <Brain className="w-5 h-5 text-green-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-semibold text-green-900 mb-1">
                      Sistema operativo
                    </h4>
                    <p className="text-sm text-green-700">
                      Las redes están entrenadas. Puedes registrar más usuarios y autenticar.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}