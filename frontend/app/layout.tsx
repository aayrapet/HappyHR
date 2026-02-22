import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HappyHR - AI Recruiter",
  description: "AI-powered voice interview platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          type="importmap"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              imports: {
                three: "https://cdn.jsdelivr.net/npm/three@0.183.1/build/three.module.js",
                "three/addons/":
                  "https://cdn.jsdelivr.net/npm/three@0.183.1/examples/jsm/",
              },
            }),
          }}
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <nav className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
          <a href="/" className="text-xl font-bold text-blue-600">HappyHR</a>
          <div className="flex gap-6 text-sm font-medium text-slate-600">
            <a href="/jobs" className="hover:text-blue-600 transition-colors">Job Posts</a>
            <a href="/dashboard" className="hover:text-blue-600 transition-colors">Dashboard</a>
          </div>
        </nav>
        <main className="min-h-[calc(100vh-57px)]">
          {children}
        </main>
      </body>
    </html>
  );
}
