import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Semantic Search",
  description: "Hybrid RAG with query expansion, HyDE, reranking and an agent",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
