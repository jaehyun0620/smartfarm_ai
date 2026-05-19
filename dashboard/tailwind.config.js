/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-base': '#070B14',
        'dark-card': '#0D1321',
        'dark-border': '#1A2235',
        'dark-hover': '#162030',
        'accent-green': '#00D084',
        'accent-blue': '#4DA6FF',
        'accent-cyan': '#00E5CC',
        'accent-orange': '#FF9A3C',
        'accent-yellow': '#FFD700',
        'accent-red': '#FF4D4D',
        'accent-purple': '#B066FF',
      },
    },
  },
  plugins: [],
}
