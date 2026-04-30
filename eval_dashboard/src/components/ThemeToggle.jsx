import { useTheme } from '../context/ThemeContext'
import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle() {
    const { theme, toggleTheme } = useTheme()

    return (
        <button
            onClick={toggleTheme}
            className="p-1.5 rounded-lg bg-gray-800/10 hover:bg-gray-800/20 dark:bg-white/5 dark:hover:bg-white/10 transition-colors border border-gray-200 dark:border-gray-700 focus:outline-none"
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
            {theme === 'dark' ? (
                <Sun className="h-4 w-4 text-amber-400" />
            ) : (
                <Moon className="h-4 w-4 text-slate-600" />
            )}
        </button>
    )
}
