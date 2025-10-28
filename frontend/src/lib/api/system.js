/**
 * API Client para endpoints del sistema biométrico
 * VERSIÓN COMPLETA CON TODOS LOS ENDPOINTS
 */

import apiClient from './client'

export const systemApi = {
  /**
   * Obtiene el estado actual del sistema
   */
  getStatus: async () => {
    try {
      const { data } = await apiClient.get('/system/status')
      return data
    } catch (error) {
      console.error('❌ Error obteniendo estado del sistema:', error)
      throw error
    }
  },

  /**
   * Obtiene el estado detallado del sistema (para debugging)
   */
  getDetailedStatus: async () => {
    try {
      const { data } = await apiClient.get('/system/status/detailed')
      return data
    } catch (error) {
      console.error('❌ Error obteniendo estado detallado:', error)
      throw error
    }
  },

  /**
   * Health check del sistema
   */
  getHealth: async () => {
    try {
      const { data } = await apiClient.get('/system/health')
      return data
    } catch (error) {
      console.error('❌ Error en health check:', error)
      throw error
    }
  },

  /**
   * Inicializa el sistema biométrico
   */
  initialize: async () => {
    try {
      const { data } = await apiClient.post('/system/initialize')
      return data
    } catch (error) {
      console.error('❌ Error inicializando sistema:', error)
      throw error
    }
  },

  /**
   * Entrena las redes neuronales (primera vez)
   */
  trainNetworks: async () => {
    try {
      const { data } = await apiClient.post('/system/train')
      return data
    } catch (error) {
      console.error('❌ Error entrenando redes:', error)
      throw error
    }
  },

  /**
   * Reentrena las redes neuronales
   * @param {boolean} force - Forzar reentrenamiento aunque ya estén entrenadas
   */
  retrainNetworks: async (force = false) => {
    try {
      const { data } = await apiClient.post(`/system/retrain?force=${force}`)
      return data
    } catch (error) {
      console.error('❌ Error reentrenando redes:', error)
      throw error
    }
  },

  /**
   * Obtiene el estado de los módulos del sistema
   */
  getModulesStatus: async () => {
    try {
      const { data } = await apiClient.get('/system/modules')
      return data
    } catch (error) {
      console.error('❌ Error obteniendo estado de módulos:', error)
      throw error
    }
  },

  /**
   * Obtiene estadísticas del sistema
   */
  getStatistics: async () => {
    try {
      const { data } = await apiClient.get('/system/statistics')
      return data
    } catch (error) {
      console.error('❌ Error obteniendo estadísticas:', error)
      throw error
    }
  },

  /**
   * Limpia recursos del sistema (cámara, MediaPipe, etc)
   */
  cleanupResources: async () => {
    try {
      const { data } = await apiClient.post('/system/cleanup')
      return data
    } catch (error) {
      console.error('❌ Error limpiando recursos:', error)
      throw error
    }
  },
}

// Exportar también como default
export default systemApi