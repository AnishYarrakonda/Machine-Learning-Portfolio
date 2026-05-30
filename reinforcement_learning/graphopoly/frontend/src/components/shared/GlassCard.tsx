import React from 'react';
import './shared.css';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className = '', style, ...props }) => {
  return (
    <div 
      className={`glass-panel ${className}`} 
      style={{ padding: 'var(--pad-card)', display: 'flex', flexDirection: 'column', gap: 'var(--gap-md)', ...style }}
      {...props}
    >
      {children}
    </div>
  );
};
