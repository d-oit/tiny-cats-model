import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.{wasm,mjs}',
          dest: '.'
        }
      ]
    })
  ],
  base: "/tiny-cats-model/",
  // Fix for worker build: use 'es' format instead of default 'iife'
  // See: https://github.com/vitejs/vite/issues/15360
  worker: {
    format: 'es',
  },
});
