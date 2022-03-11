const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    createProxyMiddleware("/upload", {
      target: "http://localhost:8000",
      changeOrigin: true,
    }),
  );
};
