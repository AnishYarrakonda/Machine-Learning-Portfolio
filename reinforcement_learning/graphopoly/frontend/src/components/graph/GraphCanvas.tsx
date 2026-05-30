import React, { useEffect, useRef, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { useUIStore } from '../../stores/uiStore';
import { useGraphStore } from '../../stores/graphStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { api } from '../../api/client';
import { GraphRenderer } from './GraphRenderer';
import { Button } from '../shared';
import { Plus, Minus, RefreshCw, Maximize } from 'lucide-react';

export const GraphCanvas: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity.translate(400, 300).scale(1));
  const [transform, setTransform] = useState<d3.ZoomTransform>(transformRef.current);

  const mode = useUIStore(s => s.mode);
  const selectedAgent = useUIStore(s => s.selectedAgent);
  const agentColors = useUIStore(s => s.agentColors);

  const { data, layout, addNode, addEdge, setOwner, toggleDestination, updateNodePosition, setLayout } = useGraphStore();
  const isTraining = useTrainingStore(s => s.isTraining);
  const isPaused   = useTrainingStore(s => s.isPaused);
  const storeSetPaused   = useTrainingStore(s => s.pauseTraining);
  const storeSetResumed  = useTrainingStore(s => s.resumeTraining);
  const pausedForDragRef = useRef(false);

  const edgeSourceRef = useRef<number | null>(null);
  const dragNodeRef = useRef<number | null>(null);
  const dragMovedRef = useRef(false);
  const mouseDownPosRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (e) => {
        transformRef.current = e.transform;
        setTransform(e.transform);
      })
      .filter((event) => {
        if (event.type === 'wheel') return false;
        if (event.type === 'mousedown' || event.type === 'pointerdown' || event.type === 'touchstart') {
          if (mode !== 'view') return false;
          const tag = (event.target as Element).tagName;
          if (tag !== 'rect' && tag !== 'svg') return false;
          return event.button === 0;
        }
        return false;
      });

    svg.call(zoom);
    zoomRef.current = zoom;
    return () => { svg.on('.zoom', null); };
  }, [mode]);

  const zoomIn = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    zoomRef.current.scaleBy(d3.select(svgRef.current).transition().duration(250) as any, 1.5);
  }, []);
  const zoomOut = useCallback(() => {
    if (!svgRef.current || !zoomRef.current) return;
    zoomRef.current.scaleBy(d3.select(svgRef.current).transition().duration(250) as any, 1 / 1.5);
  }, []);

  const centerView = useCallback(() => {
    if (!svgRef.current || !zoomRef.current || !layout) return;
    const nodes = Object.values(layout);
    if (nodes.length === 0) return;
    const [minX, maxX] = d3.extent(nodes, d => d[0]) as [number, number];
    const [minY, maxY] = d3.extent(nodes, d => d[1]) as [number, number];
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const svgNode = svgRef.current;
    const box = svgNode.getBoundingClientRect();
    const t = d3.zoomIdentity.translate(box.width / 2, box.height / 2).scale(0.75).translate(-midX, -midY);
    d3.select(svgNode).transition().duration(600).call(zoomRef.current.transform, t);
  }, [layout]);

  // Auto-fit when a new graph is loaded (num_nodes changes)
  const prevNumNodes = useRef<number>(0);
  useEffect(() => {
    const n = data?.num_nodes ?? 0;
    if (n > 0 && n !== prevNumNodes.current) {
      prevNumNodes.current = n;
      setTimeout(centerView, 150);
    }
  }, [data?.num_nodes, centerView]);

  const screenToGraph = useCallback((clientX: number, clientY: number): [number, number] => {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const rect = svg.getBoundingClientRect();
    const t = transformRef.current;
    return [(clientX - rect.left - t.x) / t.k, (clientY - rect.top - t.y) / t.k];
  }, []);

  const findNearestNode = useCallback((x: number, y: number, radius: number): number | null => {
    if (!layout) return null;
    let closest: number | null = null;
    let closestDist = radius;
    for (const [idStr, pos] of Object.entries(layout)) {
      const dx = pos[0] - x;
      const dy = pos[1] - y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < closestDist) { closestDist = dist; closest = Number(idStr); }
    }
    return closest;
  }, [layout]);

  const normalizeGraph = useCallback(() => {
    if (!data) return;
    const n = data.num_nodes;
    const nextLayout: Record<number, [number, number]> = {};
    const R = Math.max(250, Math.min(600, n * 35));
    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * i) / n;
      nextLayout[i] = [R * Math.cos(angle), R * Math.sin(angle)];
    }
    setLayout(nextLayout);
    setTimeout(centerView, 50);
  }, [data, setLayout, centerView]);

  const handleMouseDown = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    mouseDownPosRef.current = { x: e.clientX, y: e.clientY };
    const [gx, gy] = screenToGraph(e.clientX, e.clientY);
    const hitNode = findNearestNode(gx, gy, 45 / transformRef.current.k);
    if (hitNode !== null) {
      dragNodeRef.current = hitNode;
      dragMovedRef.current = false;
      e.stopPropagation();
      if (isTraining && !isPaused) {
        pausedForDragRef.current = true;
        storeSetPaused();
        api.train.pause().catch(console.error);
      }
    }
  }, [screenToGraph, findNearestNode, isTraining, isPaused, storeSetPaused]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (dragNodeRef.current === null) return;
    dragMovedRef.current = true;
    const [gx, gy] = screenToGraph(e.clientX, e.clientY);
    updateNodePosition(dragNodeRef.current, [gx, gy]);
  }, [screenToGraph, updateNodePosition]);

  const handleMouseUp = useCallback(() => {
    if (pausedForDragRef.current) {
      if (dragMovedRef.current && layout) {
        const layoutForApi: Record<string, [number, number]> = {};
        for (const [k, v] of Object.entries(layout)) layoutForApi[String(k)] = v;
        api.graph.syncLayout(layoutForApi).catch(console.error);
      }
      api.train.resume().catch(console.error);
      storeSetResumed();
      pausedForDragRef.current = false;
    }
    dragNodeRef.current = null;
    dragMovedRef.current = false;
  }, [layout, storeSetResumed]);

  const handleClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (mode === 'view') return;
    if (mouseDownPosRef.current) {
      const dx = e.clientX - mouseDownPosRef.current.x;
      const dy = e.clientY - mouseDownPosRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) > 6) return;
    }
    const [gx, gy] = screenToGraph(e.clientX, e.clientY);
    const hitNode = findNearestNode(gx, gy, 45 / transformRef.current.k);
    switch (mode) {
      case 'build_node': {
        if (hitNode !== null) return;
        addNode(data?.num_nodes ?? 0, [gx, gy]);
        break;
      }
      case 'build_edge': {
        if (hitNode === null) { edgeSourceRef.current = null; return; }
        if (edgeSourceRef.current === null) { edgeSourceRef.current = hitNode; }
        else {
          if (edgeSourceRef.current !== hitNode) addEdge(edgeSourceRef.current, hitNode);
          edgeSourceRef.current = null;
        }
        break;
      }
      case 'build_owner': {
        if (hitNode !== null) setOwner(hitNode, parseInt(selectedAgent ?? '0'));
        break;
      }
      case 'build_dest': {
        if (hitNode !== null) toggleDestination(selectedAgent ?? '0', hitNode);
        break;
      }
    }
  }, [mode, data, addNode, addEdge, setOwner, toggleDestination, selectedAgent, screenToGraph, findNearestNode]);

  // Helper to convert hex to rgb components for SVG gradients
  const hexToRgb = (hex: string) => {
    const h = hex.replace('#', '');
    return {
      r: parseInt(h.substring(0, 2), 16),
      g: parseInt(h.substring(2, 4), 16),
      b: parseInt(h.substring(4, 6), 16),
    };
  };

  return (
    <div ref={wrapperRef} style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>

      {/* Zoom Controls Pill */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        background: 'var(--color-bg-elevated)',
        backdropFilter: 'blur(8px)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-pill)',
        padding: '4px 14px',
        boxShadow: 'var(--shadow-elevated)',
      }}>
        <button onClick={zoomOut} title="Zoom Out" style={{ background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer', padding: '6px 8px', display: 'flex' }}><Minus size={14} /></button>
        <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text)', minWidth: 42, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
          {Math.round(transform.k * 100)}%
        </div>
        <button onClick={zoomIn} title="Zoom In" style={{ background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer', padding: '6px 8px', display: 'flex' }}><Plus size={14} /></button>
        <div style={{ height: 12, width: 1, background: 'var(--color-border)', margin: '0 10px' }} />
        <button onClick={centerView} title="Recenter View" style={{ background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer', display: 'flex', padding: '2px' }}><Maximize size={14} /></button>
      </div>

      {/* Mode Banner */}
      {mode !== 'view' && (
        <div style={{
          position: 'absolute', top: 72, left: '50%', transform: 'translateX(-50%)', zIndex: 10,
          background: 'var(--color-bg-elevated)',
          borderLeft: '3px solid var(--color-accent)',
          color: 'var(--color-text)',
          padding: '6px 14px',
          borderRadius: 'var(--radius-sm)',
          fontSize: 'var(--text-sm)',
          fontWeight: 500,
          boxShadow: 'var(--shadow-elevated)',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--color-accent)' }} />
          {mode.replace('build_', '').charAt(0).toUpperCase() + mode.replace('build_', '').slice(1)} Mode
        </div>
      )}

      {/* Bottom Actions */}
      <div style={{ position: 'absolute', bottom: 20, right: 20, zIndex: 10 }}>
        <Button onClick={normalizeGraph} variant="secondary" style={{ padding: '8px 14px', fontSize: 'var(--text-sm)', borderRadius: 'var(--radius-pill)', gap: 6 }}>
          <RefreshCw size={13} /> Normalize
        </Button>
      </div>

      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        style={{ width: '100%', height: '100%', display: 'block', cursor: mode === 'view' ? (dragNodeRef.current ? 'grabbing' : 'grab') : (mode === 'build_node' ? 'crosshair' : 'pointer') }}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onClick={handleClick}
        onMouseLeave={handleMouseUp}
      >
        <defs>
          {/* Dot grid pattern */}
          <pattern id="dotgrid" width="32" height="32" patternUnits="userSpaceOnUse">
            <circle cx="16" cy="16" r="1" fill="rgba(255,255,255,0.04)" />
          </pattern>

          {/* Node shadow filter */}
          <filter id="nodeShadow" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="rgba(0,0,0,0.4)" />
          </filter>

          {/* Agent glow filter */}
          <filter id="agentGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Default node gradient (unowned) */}
          <radialGradient id="nodeGradientDefault" cx="35%" cy="35%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.10)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.03)" />
          </radialGradient>

          {/* Per-agent node gradients */}
          {agentColors.map((color, i) => {
            const { r, g, b } = hexToRgb(color);
            return (
              <radialGradient key={`nodeGrad-${i}`} id={`nodeGradient-${i}`} cx="35%" cy="35%">
                <stop offset="0%" stopColor={`rgba(${r},${g},${b},0.20)`} />
                <stop offset="70%" stopColor={`rgba(${r},${g},${b},0.10)`} />
                <stop offset="100%" stopColor={`rgba(${r},${g},${b},0.04)`} />
              </radialGradient>
            );
          })}

          {/* Vignette gradient */}
          <radialGradient id="vignette" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="80%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.25)" />
          </radialGradient>
        </defs>

        {/* Background layers */}
        <rect width="100%" height="100%" fill="url(#dotgrid)" />
        <rect width="100%" height="100%" fill="url(#vignette)" />

        <g transform={transform.toString()}>
          <GraphRenderer />
        </g>
      </svg>
    </div>
  );
};
