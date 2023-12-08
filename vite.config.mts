import legacy from '@vitejs/plugin-legacy'
import { defineConfig } from 'vite'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
    plugins: [
        nodePolyfills({
            include: ['fs', 'crypto', 'path']
        }),
    ],
})