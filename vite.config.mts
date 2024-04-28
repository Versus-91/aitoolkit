import { defineConfig } from 'vite'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
    assetsInclude: ['*.csv'],
    plugins: [
        nodePolyfills({
            include: ['fs', 'crypto', 'path', 'stream']
        }),
    ],
})