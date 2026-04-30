import { Outlet } from 'react-router-dom'

export default function App() {
    return (
        <div className="flex flex-col h-full bg-[var(--bg-app)] text-[var(--text-main)] overflow-hidden transition-colors duration-200">
            <Outlet />
        </div>
    )
}
