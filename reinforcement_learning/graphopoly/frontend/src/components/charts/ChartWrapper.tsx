import React, { useMemo } from 'react';
import { Line, Bar, Doughnut, Radar, Scatter } from 'react-chartjs-2';
import { Download } from 'lucide-react';
import type { ChartDef, BuildParams } from '../../lib/chartRegistry';
import { downloadChartCsv } from '../../lib/csvExport';

interface Props {
  chart: ChartDef;
  params: BuildParams;
}

export const ChartWrapper: React.FC<Props> = ({ chart, params }) => {
  const data = useMemo(() => chart.buildData(params), [chart, params]);
  const options = useMemo(() => chart.buildOptions(params), [chart, params]);

  const handleDownload = () => {
    downloadChartCsv(chart.id, params);
  };

  const ChartComponent = (() => {
    switch (chart.chartType) {
      case 'line': return Line;
      case 'bar': return Bar;
      case 'doughnut': return Doughnut;
      case 'radar': return Radar;
      case 'scatter': return Scatter;
      default: return Line;
    }
  })();

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Download button */}
      <button
        onClick={handleDownload}
        title={`Download ${chart.csvFilename}`}
        style={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 10,
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 6,
          padding: '6px 8px',
          cursor: 'pointer',
          color: 'rgba(255,255,255,0.35)',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          fontSize: 10,
          fontFamily: "'Inter', sans-serif",
          transition: 'all 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.7)'; e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; }}
        onMouseLeave={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.35)'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
      >
        <Download size={12} />
        CSV
      </button>

      <ChartComponent data={data as any} options={options as any} />
    </div>
  );
};
