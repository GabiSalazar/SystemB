import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../../components/ui'
import { Shield } from 'lucide-react'

export default function Authentication() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Autenticación</h1>
        <p className="text-gray-600 mt-1">Verificación e identificación biométrica</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Módulo en Construcción
          </CardTitle>
          <CardDescription>
            Próximamente: Verificación 1:1 e Identificación 1:N
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600">
            Este módulo permitirá autenticar usuarios mediante gestos de mano.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}