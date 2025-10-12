import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Interceptor para logging en desarrollo
if (import.meta.env.DEV) {
  apiClient.interceptors.request.use((config) => {
    console.log('🚀 API Request:', config.method?.toUpperCase(), config.url)
    return config
  })

  apiClient.interceptors.response.use(
    (response) => {
      console.log('✅ API Response:', response.status, response.config.url)
      return response
    },
    (error) => {
      console.error('❌ API Error:', error.response?.status, error.config?.url)
      return Promise.reject(error)
    }
  )
}

export default apiClient