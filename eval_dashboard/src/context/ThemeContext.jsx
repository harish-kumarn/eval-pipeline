import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState(() => {
        return localStorage.getItem('app-theme') || 'light'
    })

    useEffect(() => {
        localStorage.setItem('app-theme', theme)
        // Apply theme to body for global CSS targeting
        document.documentElement.classList.remove('light-theme', 'dark-theme')
        document.documentElement.classList.add(`${theme}-theme`)
    }, [theme])

    const toggleTheme = () => {
        setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))
    }

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useTheme() {
    const context = useContext(ThemeContext)
    if (!context) throw new Error('useTheme must be used within ThemeProvider')
    return context
}
