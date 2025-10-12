import apiClient from './client'

export const systemApi = {
  getStatus: async () => {
    const { data } = await apiClient.get('/system/status')
    return data
  },

  getHealth: async () => {
    const { data } = await apiClient.get('/health')
    return data
  },

  initialize: async () => {
    const { data } = await apiClient.post('/system/initialize')
    return data
  },

  trainNetworks: async () => {
    const { data } = await apiClient.post('/system/train')
    return data
  },
}