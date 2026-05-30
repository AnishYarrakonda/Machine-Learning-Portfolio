import React, { useState, useRef, useEffect } from 'react';
import { ChevronRight } from 'lucide-react';

interface AccordionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export const Accordion: React.FC<AccordionProps> = ({ title, icon, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const contentRef = useRef<HTMLDivElement>(null);
  const [contentHeight, setContentHeight] = useState(0);

  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(contentRef.current.scrollHeight);
    }
  }, [isOpen, children]);

  return (
    <div style={{ borderBottom: '1px solid var(--color-border)' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 20px',
          background: 'none',
          border: 'none',
          borderLeft: isOpen ? '2px solid var(--color-accent)' : '2px solid transparent',
          cursor: 'pointer',
          color: 'var(--color-text)',
          transition: 'all var(--transition-base)',
        }}
        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.02)'}
        onMouseLeave={e => e.currentTarget.style.background = 'none'}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            color: isOpen ? 'var(--color-accent)' : 'var(--color-text-dim)',
            transition: 'color var(--transition-base)',
            display: 'flex',
          }}>
            {icon}
          </span>
          <span style={{
            fontSize: 'var(--text-md)',
            fontWeight: 500,
            letterSpacing: '0.01em',
            color: isOpen ? 'var(--color-text)' : 'var(--color-text-secondary)',
          }}>
            {title}
          </span>
        </div>
        <ChevronRight
          size={16}
          color="var(--color-text-dim)"
          style={{
            transition: 'transform var(--transition-base)',
            transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
          }}
        />
      </button>
      <div
        ref={contentRef}
        style={{
          maxHeight: isOpen ? contentHeight + 40 : 0,
          overflow: 'hidden',
          opacity: isOpen ? 1 : 0,
          transition: 'max-height 0.25s ease-out, opacity 0.2s ease-out',
        }}
      >
        <div style={{ padding: '4px 20px 20px 20px' }}>
          {children}
        </div>
      </div>
    </div>
  );
};
