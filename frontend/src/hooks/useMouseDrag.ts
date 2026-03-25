/**
 * useMouseDrag: Custom React hook for mouse-drag brightness/contrast.
 *
 * - Drag up/down → brightness adjustment
 * - Drag left/right → contrast adjustment
 * - Display-only; does not modify underlying data (FR-012)
 */

import { useCallback, useRef, useState } from 'react';

interface MouseDragState {
  brightness: number;
  contrast: number;
  isDragging: boolean;
  handlers: {
    onMouseDown: (e: React.MouseEvent) => void;
    onMouseMove: (e: React.MouseEvent) => void;
    onMouseUp: () => void;
    onMouseLeave: () => void;
  };
  reset: () => void;
}

export function useMouseDrag(sensitivity = 0.005): MouseDragState {
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef<{ x: number; y: number; b: number; c: number }>({
    x: 0,
    y: 0,
    b: 0,
    c: 1,
  });

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return; // left click only
      setIsDragging(true);
      dragStart.current = {
        x: e.clientX,
        y: e.clientY,
        b: brightness,
        c: contrast,
      };
      e.preventDefault();
    },
    [brightness, contrast]
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) return;

      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;

      // Up = brighter (negative dy), Down = darker
      const newBrightness = Math.max(
        -1,
        Math.min(1, dragStart.current.b - dy * sensitivity)
      );

      // Right = more contrast, Left = less contrast
      const newContrast = Math.max(
        0.1,
        Math.min(10, dragStart.current.c + dx * sensitivity * 2)
      );

      setBrightness(newBrightness);
      setContrast(newContrast);
    },
    [isDragging, sensitivity]
  );

  const onMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const onMouseLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const reset = useCallback(() => {
    setBrightness(0);
    setContrast(1);
  }, []);

  return {
    brightness,
    contrast,
    isDragging,
    handlers: {
      onMouseDown,
      onMouseMove,
      onMouseUp,
      onMouseLeave,
    },
    reset,
  };
}
