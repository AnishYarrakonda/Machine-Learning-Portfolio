import React, { useCallback, useEffect, useRef, useState } from 'react';

// ── Color math ────────────────────────────────────────────────────────────────

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h / 60) % 6;
  const f = h / 60 - Math.floor(h / 60);
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  const m = [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i];
  return [Math.round(m[0]*255), Math.round(m[1]*255), Math.round(m[2]*255)];
}

function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r,g,b), min = Math.min(r,g,b), d = max - min;
  let h = 0;
  if (d) {
    if      (max === r) h = ((g-b)/d + 6) % 6;
    else if (max === g) h =  (b-r)/d + 2;
    else                h =  (r-g)/d + 4;
    h *= 60;
  }
  return [h, max ? d/max : 0, max];
}

function hexToRgb(hex: string): [number,number,number] | null {
  const m = hex.match(/^#?([0-9a-f]{6})$/i);
  if (!m) return null;
  return [parseInt(m[1].slice(0,2),16), parseInt(m[1].slice(2,4),16), parseInt(m[1].slice(4,6),16)];
}

function toHex(r: number, g: number, b: number): string {
  return '#' + [r,g,b].map(c => c.toString(16).padStart(2,'0')).join('');
}

// ── Component ─────────────────────────────────────────────────────────────────

const SV_W = 160, SV_H = 110, HUE_W = 160, HUE_H = 12;
const POPUP_W = SV_W + 24;

interface Props {
  value: string;
  onChange: (hex: string) => void;
}

export const ColorPicker: React.FC<Props> = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  const [hsv, setHsv] = useState<[number,number,number]>(() => {
    const rgb = hexToRgb(value);
    return rgb ? rgbToHsv(...rgb) : [210, 0.55, 0.65];
  });
  const [hexInput, setHexInput] = useState(value);
  const [popupPos, setPopupPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });

  const svRef    = useRef<HTMLCanvasElement>(null);
  const hueRef   = useRef<HTMLCanvasElement>(null);
  const popRef   = useRef<HTMLDivElement>(null);
  const btnRef   = useRef<HTMLButtonElement>(null);
  const dragging = useRef<'sv'|'hue'|null>(null);

  // Sync from parent
  useEffect(() => {
    const rgb = hexToRgb(value);
    if (rgb) {
      const newHsv = rgbToHsv(...rgb);
      setHsv(prev => {
        if (prev[0] === newHsv[0] && prev[1] === newHsv[1] && prev[2] === newHsv[2]) return prev;
        return newHsv;
      });
      setHexInput(prev => (prev === value ? prev : value));
    }
  }, [value]);

  const computePopupPos = useCallback(() => {
    if (!btnRef.current) return;
    const rect = btnRef.current.getBoundingClientRect();
    const POPUP_H = SV_H + HUE_H + 80;
    const GAP = 10;
    let top = rect.top - POPUP_H - GAP;
    if (top < 8) top = rect.bottom + GAP;
    let left = rect.left + rect.width / 2 - POPUP_W / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - POPUP_W - 8));
    setPopupPos({ top, left });
  }, []);

  const handleOpen = () => {
    computePopupPos();
    setOpen(o => !o);
  };

  // Click-outside close
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => {
      if (
        popRef.current && !popRef.current.contains(e.target as Node) &&
        btnRef.current && !btnRef.current.contains(e.target as Node)
      ) setOpen(false);
    };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [open]);

  // Draw SV canvas
  const hsvHue = hsv[0];
  useEffect(() => {
    if (!open) return;
    const c = svRef.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const hg = ctx.createLinearGradient(0, 0, SV_W, 0);
    hg.addColorStop(0, '#fff');
    hg.addColorStop(1, `hsl(${hsvHue},100%,50%)`);
    ctx.fillStyle = hg; ctx.fillRect(0, 0, SV_W, SV_H);
    const bg = ctx.createLinearGradient(0, 0, 0, SV_H);
    bg.addColorStop(0, 'rgba(0,0,0,0)');
    bg.addColorStop(1, '#000');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, SV_W, SV_H);
  }, [open, hsvHue]);

  // Draw hue canvas
  useEffect(() => {
    if (!open) return;
    const c = hueRef.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const g = ctx.createLinearGradient(0, 0, HUE_W, 0);
    for (let i = 0; i <= 6; i++) g.addColorStop(i/6, `hsl(${i*60},100%,50%)`);
    ctx.fillStyle = g; ctx.fillRect(0, 0, HUE_W, HUE_H);
  }, [open]);

  const pickSV = useCallback((e: { clientX: number; clientY: number }) => {
    const c = svRef.current; if (!c) return;
    const r = c.getBoundingClientRect();
    const s = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const v = 1 - Math.max(0, Math.min(1, (e.clientY - r.top) / r.height));
    const nHsv: [number,number,number] = [hsv[0], s, v];
    setHsv(nHsv);
    const hex = toHex(...hsvToRgb(...nHsv));
    setHexInput(hex); onChange(hex);
  }, [hsv, onChange]);

  const pickHue = useCallback((e: { clientX: number; clientY: number }) => {
    const c = hueRef.current; if (!c) return;
    const r = c.getBoundingClientRect();
    const h = Math.max(0, Math.min(360, ((e.clientX - r.left) / r.width) * 360));
    const nHsv: [number,number,number] = [h, hsv[1], hsv[2]];
    setHsv(nHsv);
    const hex = toHex(...hsvToRgb(...nHsv));
    setHexInput(hex); onChange(hex);
  }, [hsv, onChange]);

  useEffect(() => {
    const move = (e: MouseEvent) => {
      if      (dragging.current === 'sv')  pickSV(e);
      else if (dragging.current === 'hue') pickHue(e);
    };
    const up = () => { dragging.current = null; };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
  }, [pickSV, pickHue]);

  const handleHexChange = (raw: string) => {
    setHexInput(raw);
    const normalized = raw.startsWith('#') ? raw : '#' + raw;
    const rgb = hexToRgb(normalized);
    if (rgb) { setHsv(rgbToHsv(...rgb)); onChange(normalized); }
  };

  const svCursorX = hsv[1] * SV_W - 5;
  const svCursorY = (1 - hsv[2]) * SV_H - 5;
  const hueCursorX = (hsv[0] / 360) * HUE_W - 5;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      {/* Swatch button */}
      <button
        ref={btnRef}
        onClick={handleOpen}
        style={{
          width: 34, height: 34, borderRadius: '50%',
          background: value,
          border: open ? '2px solid rgba(255,255,255,0.65)' : '2px solid rgba(255,255,255,0.18)',
          cursor: 'pointer',
          transition: 'border-color 0.15s',
          outline: 'none',
          flexShrink: 0,
        }}
      />

      {/* Picker popup — fixed to viewport so it is never clipped by overflow */}
      {open && (
        <div
          ref={popRef}
          style={{
            position: 'fixed',
            top: popupPos.top,
            left: popupPos.left,
            width: POPUP_W,
            background: '#141416',
            border: '1px solid rgba(255,255,255,0.12)',
            padding: 12,
            zIndex: 99999,
            boxShadow: '0 16px 48px rgba(0,0,0,0.8)',
            borderRadius: 6,
          }}
          onMouseDown={e => e.stopPropagation()}
        >
          {/* SV canvas */}
          <div style={{ position: 'relative', marginBottom: 10, cursor: 'crosshair', lineHeight: 0, borderRadius: 3, overflow: 'hidden' }}>
            <canvas
              ref={svRef}
              width={SV_W}
              height={SV_H}
              style={{ display: 'block', width: SV_W, height: SV_H }}
              onMouseDown={e => { dragging.current = 'sv'; pickSV(e); }}
            />
            <div style={{
              position: 'absolute', left: svCursorX, top: svCursorY,
              width: 10, height: 10, borderRadius: '50%',
              border: '2px solid #fff', boxShadow: '0 0 4px rgba(0,0,0,0.6)',
              pointerEvents: 'none',
            }} />
          </div>

          {/* Hue slider */}
          <div style={{ position: 'relative', marginBottom: 14, cursor: 'crosshair', lineHeight: 0 }}>
            <canvas
              ref={hueRef}
              width={HUE_W}
              height={HUE_H}
              style={{ display: 'block', width: HUE_W, height: HUE_H, borderRadius: 2 }}
              onMouseDown={e => { dragging.current = 'hue'; pickHue(e); }}
            />
            <div style={{
              position: 'absolute', left: hueCursorX, top: -2,
              width: 10, height: 16, borderRadius: 2,
              border: '2px solid #fff', background: `hsl(${hsv[0]},100%,50%)`,
              boxShadow: '0 0 4px rgba(0,0,0,0.5)', pointerEvents: 'none',
            }} />
          </div>

          {/* Preview + hex input */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{
              width: 24, height: 24, background: value,
              border: '1px solid rgba(255,255,255,0.15)', borderRadius: 3, flexShrink: 0,
            }} />
            <input
              value={hexInput}
              onChange={e => handleHexChange(e.target.value)}
              maxLength={7}
              spellCheck={false}
              style={{
                flex: 1, background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.10)',
                color: 'rgba(255,255,255,0.8)', fontFamily: 'monospace',
                fontSize: 12, padding: '6px 10px', outline: 'none',
                letterSpacing: '0.08em', borderRadius: 3,
              }}
              onFocus={e => (e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)')}
              onBlur={e => (e.currentTarget.style.borderColor = 'rgba(255,255,255,0.10)')}
            />
          </div>
        </div>
      )}
    </div>
  );
};
