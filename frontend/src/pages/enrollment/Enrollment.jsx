import { useState } from 'react'
import { enrollmentApi } from '../../lib/api/enrollment'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, Button, Badge } from '../../components/ui'
import WebcamCapture from '../../components/camera/WebcamCapture'
import { UserPlus, CheckCircle, XCircle, Camera, Hand } from 'lucide-react'

export default function Enrollment() {
  const [step, setStep] = useState('form')
  const [userId, setUserId] = useState('')
  const [username, setUsername] = useState('')
  const [selectedGestures, setSelectedGestures] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [sessionStatus, setSessionStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const availableGestures = [
    'Open_Palm',
    'Closed_Fist',
    'Victory',
    'Thumb_Up',
    'Thumb_Down',
    'Pointing_Up',
    'ILoveYou'
  ]

  const handleGestureToggle = (gesture) => {
    if (selectedGestures.includes(gesture)) {
      setSelectedGestures(selectedGestures.filter(g => g !== gesture))
    } else if (selectedGestures.length < 3) {
      setSelectedGestures([...selectedGestures, gesture])
    }
  }

  const handleStartEnrollment = async () => {
    if (!userId || !username || selectedGestures.length !== 3) {
      alert('Por favor completa todos los campos y selecciona 3 gestos')
      return
    }

    try {
      setLoading(true)
      const response = await enrollmentApi.startEnrollment(userId, username, selectedGestures)
      setSessionId(response.session_id)
      setStep('capture')
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error al iniciar enrollment')
    } finally {
      setLoading(false)
    }
  }

  const handleFrameCapture = async (frameBlob) => {
    if (!sessionId) return

    try {
      const response = await enrollmentApi.processFrame(sessionId, frameBlob)
      setSessionStatus(response)

      if (response.session_completed) {
        setStep('success')
      }
    } catch (err) {
      console.error('Error procesando frame:', err)
    }
  }

  const handleCancel = async () => {
    if (sessionId) {
      try {
        await enrollmentApi.cancelEnrollment(sessionId)
      } catch (err) {
        console.error('Error cancelando:', err)
      }
    }
    resetForm()
  }

  const resetForm = () => {
    setStep('form')
    setUserId('')
    setUsername('')
    setSelectedGestures([])
    setSessionId(null)
    setSessionStatus(null)
    setError(null)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Registro de Usuarios</h1>
        <p className="text-gray-600 mt-1">Sistema de enrollment biométrico por gestos</p>
      </div>

      {step === 'form' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserPlus className="w-5 h-5" />
              Nuevo Usuario
            </CardTitle>
            <CardDescription>
              Ingresa los datos del usuario y selecciona 3 gestos para su secuencia biométrica
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  ID de Usuario
                </label>
                <input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Ej: usuario001"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Nombre Completo
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Ej: Juan Pérez"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="block text-sm font-medium text-gray-700">
                  Secuencia de Gestos ({selectedGestures.length}/3)
                </label>
                <Badge variant={selectedGestures.length === 3 ? 'success' : 'default'}>
                  {selectedGestures.length === 3 ? 'Completo' : 'Selecciona gestos'}
                </Badge>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {availableGestures.map((gesture) => (
                  <button
                    key={gesture}
                    onClick={() => handleGestureToggle(gesture)}
                    disabled={!selectedGestures.includes(gesture) && selectedGestures.length >= 3}
                    className={`
                      p-4 rounded-lg border-2 transition-all text-center
                      ${selectedGestures.includes(gesture)
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                      }
                      ${!selectedGestures.includes(gesture) && selectedGestures.length >= 3
                        ? 'opacity-50 cursor-not-allowed'
                        : 'cursor-pointer'
                      }
                    `}
                  >
                    <div className="text-3xl mb-2">
                      {gesture === 'Open_Palm' && '🖐️'}
                      {gesture === 'Closed_Fist' && '✊'}
                      {gesture === 'Victory' && '✌️'}
                      {gesture === 'Thumb_Up' && '👍'}
                      {gesture === 'Thumb_Down' && '👎'}
                      {gesture === 'Pointing_Up' && '☝️'}
                      {gesture === 'ILoveYou' && '🤟'}
                    </div>
                    <p className="text-xs font-medium text-gray-700">
                      {gesture.replace('_', ' ')}
                    </p>
                    {selectedGestures.includes(gesture) && (
                      <Badge variant="primary" className="mt-2">
                        #{selectedGestures.indexOf(gesture) + 1}
                      </Badge>
                    )}
                  </button>
                ))}
              </div>

              {selectedGestures.length > 0 && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-900">
                    Secuencia seleccionada:
                  </p>
                  <p className="text-sm text-blue-700 mt-1">
                    {selectedGestures.join(' → ')}
                  </p>
                </div>
              )}
            </div>

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </CardContent>

          <CardFooter>
            <Button
              onClick={handleStartEnrollment}
              disabled={!userId || !username || selectedGestures.length !== 3 || loading}
              className="w-full"
            >
              {loading ? 'Iniciando...' : (
                <>
                  <Camera className="w-4 h-4 mr-2" />
                  Iniciar Captura
                </>
              )}
            </Button>
          </CardFooter>
        </Card>
      )}

      {step === 'capture' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Captura en Vivo</CardTitle>
                <CardDescription>
                  Realiza los gestos según las indicaciones
                </CardDescription>
              </CardHeader>
              <CardContent>
                <WebcamCapture
                  onFrame={handleFrameCapture}
                  isActive={step === 'capture'}
                />
              </CardContent>
              <CardFooter>
                <Button variant="danger" onClick={handleCancel} className="w-full">
                  <XCircle className="w-4 h-4 mr-2" />
                  Cancelar Captura
                </Button>
              </CardFooter>
            </Card>
          </div>

          <div>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Progreso</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {sessionStatus ? (
                  <>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Completado</span>
                        <span className="text-sm font-bold">
                          {Math.round(sessionStatus.progress || 0)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all"
                          style={{ width: `${sessionStatus.progress || 0}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-medium text-gray-700">
                        Gesto actual:
                      </p>
                      <Badge variant="info" className="text-base">
                        <Hand className="w-4 h-4 mr-1" />
                        {sessionStatus.current_gesture || 'Esperando...'}
                      </Badge>
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-medium text-gray-700">
                        Muestras capturadas:
                      </p>
                      <p className="text-2xl font-bold">
                        {sessionStatus.samples_collected || 0} / {sessionStatus.samples_needed || 21}
                      </p>
                    </div>

                    {sessionStatus.message && (
                      <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <p className="text-xs text-yellow-800">
                          {sessionStatus.message}
                        </p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center py-8 text-gray-400 text-sm">
                    Inicializando...
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {step === 'success' && (
        <Card>
          <CardContent className="pt-12 pb-12 text-center">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              ¡Registro Completado!
            </h2>
            <p className="text-gray-600 mb-6">
              El usuario <strong>{username}</strong> ha sido registrado exitosamente.
            </p>
            <Button onClick={resetForm}>
              <UserPlus className="w-4 h-4 mr-2" />
              Registrar Otro Usuario
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}