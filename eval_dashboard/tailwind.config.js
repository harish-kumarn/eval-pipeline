/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,jsx}'],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'ui-monospace', 'Menlo', 'monospace'],
            },
            colors: {
                gray: {
                    950: '#030712',
                },
            },
        },
    },
    plugins: [],
}
