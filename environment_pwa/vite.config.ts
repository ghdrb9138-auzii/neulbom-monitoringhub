import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

function noModelHtmlFallbackPlugin() {
  return {
    name: "no-model-html-fallback",
    configureServer(server: {
      middlewares: {
        use: (
          handler: (
            req: { url?: string },
            res: {
              statusCode: number;
              setHeader: (name: string, value: string) => void;
              end: (body?: string) => void;
            },
            next: () => void,
          ) => void,
        ) => void;
      };
    }) {
      server.middlewares.use((req, res, next) => {
        const rawUrl = req.url?.split("?")[0] ?? "";

        if (!rawUrl.startsWith("/models/")) {
          next();
          return;
        }

        const decodedUrl = decodeURIComponent(rawUrl);
        const filePath = path.join(
          process.cwd(),
          "public",
          decodedUrl.replace(/^\/+/, ""),
        );

        if (!fs.existsSync(filePath)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "text/plain; charset=utf-8");
          res.end(`Model file not found: ${decodedUrl}`);
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  server: {
    host: true,
    port: 5176,
    allowedHosts: [".trycloudflare.com", ".ngrok-free.app", ".ngrok.io"],
  },
  plugins: [
    noModelHtmlFallbackPlugin(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.ico", "apple-touch-icon-180x180.png"],
      workbox: {
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      },
      manifest: {
        name: "보행 도로 위험 감지 PWA",
        short_name: "보행 환경",
        description: "도로 위험 감지를 위한 보행 환경 PWA",
        theme_color: "#0f172a",
        background_color: "#f8fafc",
        display: "standalone",
        orientation: "portrait",
        start_url: "/",
        scope: "/",
        lang: "ko",
        icons: [
          { src: "pwa-64x64.png", sizes: "64x64", type: "image/png" },
          { src: "pwa-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512x512.png", sizes: "512x512", type: "image/png" },
          {
            src: "maskable-icon-512x512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
    }),
  ],
});
