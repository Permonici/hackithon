/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#18212f",
        mist: "#f5f7fb",
        mint: "#1e9d73",
        coral: "#e86f51",
        amber: "#d99b22"
      },
      boxShadow: {
        panel: "0 18px 50px rgba(24, 33, 47, 0.10)"
      }
    }
  },
  plugins: []
};
