import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/static/predictions/react/',
  plugins: [react()],
  build: {
    outDir: '../predictions/static/predictions/react',
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        entryFileNames: 'loan-form.js',
        chunkFileNames: 'loan-form.js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.css')) {
            return 'loan-form.css'
          }
          return 'assets/[name][extname]'
        },
      },
    },
  },
})
