const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    createProxyMiddleware("/upload", {
      target: "http://localhost:8000",
      changeOrigin: true,
    }),
    createProxyMiddleware("/progress", {
      target: "http://localhost:8000",
      changeOrigin: true,
    }),
  );
};
