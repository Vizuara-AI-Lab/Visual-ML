import { useState, useCallback, useEffect, useRef } from "react";

const STORAGE_KEY = "vml-drawer-height";
const MIN_HEIGHT = 150;
const MAX_HEIGHT_RATIO = 0.6;

function getInitialHeight(): number {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = parseInt(stored, 10);
      if (!isNaN(parsed) && parsed >= MIN_HEIGHT) return parsed;
    }
  } catch {}
  return 300;
}

export function useDrawerResize() {
  const [height, setHeight] = useState(getInitialHeight);
  const [isResizing, setIsResizing] = useState(false);
  const rafRef = useRef<number>(0);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  useEffect(() => {
    if (!isResizing) return;

    document.body.style.userSelect = "none";
    document.body.style.cursor = "ns-resize";

    const onMouseMove = (e: MouseEvent) => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        const maxHeight = window.innerHeight * MAX_HEIGHT_RATIO;
        const newHeight = Math.min(
          maxHeight,
          Math.max(MIN_HEIGHT, window.innerHeight - e.clientY)
        );
        setHeight(newHeight);
      });
    };

    const onMouseUp = () => {
      setIsResizing(false);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      cancelAnimationFrame(rafRef.current);
      // Persist to localStorage
      try {
        localStorage.setItem(STORAGE_KEY, String(Math.round(height)));
      } catch {}
    };

    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);

    return () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      cancelAnimationFrame(rafRef.current);
    };
  }, [isResizing, height]);

  return { height, handleMouseDown, isResizing };
}
