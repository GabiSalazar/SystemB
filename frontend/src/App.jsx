import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/dashboard/Dashboard'
import Enrollment from './pages/enrollment/Enrollment'
import Authentication from './pages/authentication/Authentication'
import './App.css'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/enrollment" element={<Enrollment />} />
          <Route path="/authentication" element={<Authentication />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App