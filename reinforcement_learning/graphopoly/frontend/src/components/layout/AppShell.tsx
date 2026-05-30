import React from 'react';
import { Header } from './Header';

interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      width: '100vw',
      background: 'var(--color-bg)',
      overflow: 'hidden', // Root shouldn't scroll; layout handles it
    }}>
      <Header />
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'row',
        paddingTop: 'var(--header-h)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {children}
      </div>
    </div>
  );
};
