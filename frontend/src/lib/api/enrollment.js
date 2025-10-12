import apiClient from './client'

export const enrollmentApi = {
  // Iniciar sesión de enrollment
  startEnrollment: async (userId, username, gestureSequence) => {
    const { data } = await apiClient.post('/enrollment/start', {
      user_id: userId,
      username: username,
      gesture_sequence: gestureSequence
    })
    return data
  },

  // Procesar frame (polling - NO se envía el frame, se procesa del buffer interno)
  processFrame: async (sessionId) => {
    const { data } = await apiClient.get(`/enrollment/${sessionId}/frame`)
    return data
  },

  // Obtener estado de la sesión
  getSessionStatus: async (sessionId) => {
    const { data } = await apiClient.get(`/enrollment/${sessionId}/status`)
    return data
  },

  // Cancelar enrollment
  cancelEnrollment: async (sessionId) => {
    const { data } = await apiClient.post(`/enrollment/${sessionId}/cancel`)
    return data
  },

  // Obtener estadísticas del sistema
  getStats: async () => {
    const { data } = await apiClient.get('/enrollment/stats')
    return data
  },

  // Estado del bootstrap
  getBootstrapStatus: async () => {
    const { data } = await apiClient.get('/enrollment/bootstrap/status')
    return data
  },

  // Forzar entrenamiento (solo para testing)
  forceTraining: async () => {
    const { data } = await apiClient.post('/enrollment/force-training')
    return data
  }
}