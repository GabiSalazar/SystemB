import { useRef, useEffect, useState } from 'react'
import { Camera, CameraOff } from 'lucide-react'
import { Button } from '../ui'

export default function WebcamCapture({ onFrame, isActive = true }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (isActive) {
      startCamera()
    } else {
      stopCamera()
    }

    return () => stopCamera()
  }, [isActive])

  useEffect(() => {
    if (isStreaming && onFrame) {
      // Capturar y enviar frames cada 200ms (5 fps)
      intervalRef.current = setInterval(() => {
        captureAndSendFrame()
      }, 200)
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isStreaming, onFrame])

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
    // Detener intervalo de captura
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Detener stream de cámara
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
      setIsStreaming(false)
    }
  }

  const captureAndSendFrame = () => {
    if (videoRef.current && canvasRef.current && onFrame) {
      const video = videoRef.current
      const canvas = canvasRef.current
      
      // Asegurarse de que el video tenga dimensiones válidas
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        return
      }
      
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0)
      
      // Convertir a blob JPEG con calidad 90%
      canvas.toBlob((blob) => {
        if (blob && onFrame) {
          onFrame(blob)
        }
      }, 'image/jpeg', 0.9)
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
          
          {/* Canvas oculto para captura de frames */}
          <canvas ref={canvasRef} className="hidden" />
          
          {isStreaming && (
            <div className="absolute top-4 right-4">
              <div className="flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                CAPTURANDO
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}