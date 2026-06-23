import type { Metadata } from "next";
import "./globals.css";

import { Inter, Marcellus } from "next/font/google";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ThemeProvider } from "next-themes";

const inter = Inter({
  subsets: ["latin"],
  preload: true,
  display: "swap",
});

const marcellus = Marcellus({ // eslint-disable-line @typescript-eslint/no-unused-vars
  subsets: ["latin"],
  weight: "400",
  display: "swap",
  variable: "--font-marcellus",
});

export const metadata: Metadata = {
  title: "Agent Chat",
  description: "Agent Chat UX by LangChain",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (

    <html lang="en" suppressHydrationWarning> 
      <body className={`${inter.className} ${marcellus.variable}`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <NuqsAdapter>{children}</NuqsAdapter>
        </ThemeProvider>
      </body>
    </html>
  );
}
