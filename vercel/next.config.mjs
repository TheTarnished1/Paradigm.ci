const nextConfig = {
  // In production Vercel sends /api to the Python function. Locally it goes to uvicorn.
  async rewrites() {
    if (process.env.NODE_ENV === "production") return [];
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};
export default nextConfig;
