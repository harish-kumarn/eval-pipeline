import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { ModuleRegistry, AllCommunityModule } from 'ag-grid-community'
import { ThemeProvider } from './context/ThemeContext'
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-balham.css'
import './index.css'
import App from './App.jsx'
import Dashboard from './components/Dashboard.jsx'
import AnalyticsDashboard from './components/AnalyticsDashboard.jsx'

// Register AG Grid Modules
ModuleRegistry.registerModules([AllCommunityModule])

const router = createBrowserRouter([
    {
        path: '/',
        element: <App />,
        children: [
            {
                index: true,
                element: <Dashboard />
            },
            {
                path: 'analytics',
                element: <AnalyticsDashboard />
            }
        ]
    }
])

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <ThemeProvider>
            <RouterProvider router={router} />
        </ThemeProvider>
    </StrictMode>,
)
