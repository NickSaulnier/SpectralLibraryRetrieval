import { useCallback, useEffect, useRef, useState } from "react";

export interface Point {
  x: number;
  y: number;
}

interface View {
  scale: number;
  tx: number;
  ty: number;
}

/** Image pixel under a client position, for a given view (pan tx, ty and zoom scale). */
export function imagePixel(clientX: number, clientY: number, box: DOMRect, view: View): Point {
  return {
    x: Math.floor((clientX - box.left - view.tx) / view.scale),
    y: Math.floor((clientY - box.top - view.ty) / view.scale),
  };
}

interface Props {
  src: string | null;
  alt: string;
  marker?: Point | null;
  markerClass?: string;
  dark?: boolean;
  onPick?: (p: Point) => void;
}

/**
 * Pan (drag) and zoom (wheel, buttons) an image, and report clicks as image pixels. A stand-in
 * for the MapLibre map of Phase V1: it gives real zoom on whole-scene images today.
 */
export function ZoomImage({ src, alt, marker, markerClass = "", dark = false, onPick }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [view, setView] = useState<View>({ scale: 1, tx: 0, ty: 0 });
  const drag = useRef<{ x: number; y: number; tx: number; ty: number; moved: boolean } | null>(null);

  const fit = useCallback(() => {
    const el = box.current;
    if (!el || !size) return;
    const scale = Math.min(el.clientWidth / size.w, el.clientHeight / size.h);
    setView({ scale, tx: (el.clientWidth - size.w * scale) / 2, ty: (el.clientHeight - size.h * scale) / 2 });
  }, [size]);

  useEffect(fit, [fit]);
  useEffect(() => setSize(null), [src]);

  const zoomAt = (factor: number, cx: number, cy: number) =>
    setView((v) => {
      const scale = Math.min(Math.max(v.scale * factor, 0.02), 40);
      const k = scale / v.scale;
      return { scale, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k };
    });

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const centre = (factor: number) => {
    const el = box.current;
    if (el) zoomAt(factor, el.clientWidth / 2, el.clientHeight / 2);
  };

  return (
    <div
      ref={box}
      className={`zoom${dark ? " dark" : ""}`}
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        const dx = e.clientX - d.x;
        const dy = e.clientY - d.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
        if (d.moved) setView((v) => ({ ...v, tx: d.tx + dx, ty: d.ty + dy }));
      }}
      onPointerUp={(e) => {
        const d = drag.current;
        drag.current = null;
        if (d && !d.moved && onPick && size && box.current) {
          const p = imagePixel(e.clientX, e.clientY, box.current.getBoundingClientRect(), view);
          if (p.x >= 0 && p.y >= 0 && p.x < size.w && p.y < size.h) onPick(p);
        }
      }}
    >
      {src && (
        <img
          src={src}
          alt={alt}
          onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
          style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})` }}
        />
      )}
      {marker && size && (
        <div
          className={`marker ${markerClass}`}
          style={{ left: view.tx + (marker.x + 0.5) * view.scale, top: view.ty + (marker.y + 0.5) * view.scale }}
        />
      )}
      <div className="tools" onPointerDown={(e) => e.stopPropagation()}>
        <button type="button" aria-label="Zoom in" onClick={() => centre(1.5)}>+</button>
        <button type="button" aria-label="Zoom out" onClick={() => centre(1 / 1.5)}>−</button>
        <button type="button" aria-label="Fit to view" onClick={fit}>Fit</button>
      </div>
    </div>
  );
}
