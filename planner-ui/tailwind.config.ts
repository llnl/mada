import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        mada: {
          50: "#f0f5ff",
          100: "#e0eaff",
          500: "#4f6ef7",
          600: "#3b5de5",
          700: "#2d4bd3",
          900: "#1a2f7a",
        },
      },
    },
  },
  plugins: [],
};

export default config;
