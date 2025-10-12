import { useRef, useEffect, useState } from 'react'
import { Camera, CameraOff } from 'lucide-react'
import { Button } from '../ui'

export default function WebcamCapture({ isActive = true }) {
  const videoRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (isActive) {
      startCamera()
    } else {
      stopCamera()
    }

    return () => stopCamera()
  }, [isActive])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        setError(null)
      }
    } catch (err) {
      setError('No se pudo acceder a la cámara')
      console.error('Error al acceder a la cámara:', err)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }

  return (
    <div className="relative">
      {error ? (
        <div className="bg-gray-900 rounded-lg aspect-video flex items-center justify-center">
          <div className="text-center">
            <CameraOff className="w-12 h-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">{error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-4"
              onClick={startCamera}
            >
              Reintentar
            </Button>
          </div>
        </div>
      ) : (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full rounded-lg bg-gray-900"
          />
          
          {isStreaming && (
            <>
              <div className="absolute top-4 right-4">
                <div className="flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                  VISTA PREVIA
                </div>
              </div>
              
              <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-black/70 text-white px-4 py-2 rounded-lg text-sm">
                  ℹ️ El procesamiento se realiza en el servidor con su propia cámara
                </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}