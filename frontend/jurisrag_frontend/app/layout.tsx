import type { Metadata } from "next";
import type React from "react";
import "./styles.css";

export const metadata: Metadata = {
  title: "JurisRAG",
  description: "Grounded answers from the supplied Indian legal corpus.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
