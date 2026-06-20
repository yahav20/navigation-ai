"use client";

import Image from "next/image";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function LangGraphLogoSVG({
  className = "",
  width = 32,
  height = 32,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);


  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width, height }} className={className} />;
  }

  const isDark = resolvedTheme === "dark";

  return (
    <Image 
      src={isDark ? "images/dark_logo.png" : "/light_logo.png"} 
      alt="Agent Logo" 
      width={width} 
      height={height} 
      className={className} 
    />
  );
}