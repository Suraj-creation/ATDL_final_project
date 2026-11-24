import { useEffect, useRef, useState } from 'react';
import { ChartNoAxesCombined, Cpu, FlaskConical, ScanLine } from 'lucide-react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/instrument-sans';
import '@fontsource/ibm-plex-mono/400.css';
import './style.css';
import { fetchJson } from './api';
import { Inspect } from './Inspect';
import { Results } from './Results';
import { Study } from './Study';
import type { Project, Status, View } from './types';

const views = [
  { id: 'inspect', label: 'Inspect', icon: ScanLine },
  { id: 'experiments', label: 'Experiments', icon: FlaskConical },
  { id: 'results', label: 'Results', icon: ChartNoAxesCombined },
] as const;

function locationView(): View {
  const fragment = window.location.hash.slice(1);
  return fragment === 'experiments' || fragment === 'results' ? fragment : 'inspect';
}

const offline: Status = { ready: false, error: 'The inference service is unavailable.', device: '', device_name: '',
  torch: null, ultralytics: '', loaded_models: [] };

function App() {
  const [view, setView] = useState<View>(locationView);
  const [project, setProject] = useState<Project | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState('');
  const [loadCount, reload] = useState(0);
  const [request, setRequest] = useState<{ id: number; name?: string; resolution?: boolean } | null>(null);
  const counter = useRef(0);
  useEffect(() => {
    const change = () => setView(locationView());
    window.addEventListener('hashchange', change);
    return () => window.removeEventListener('hashchange', change);
  }, []);
  useEffect(() => {
    const signal = new AbortController();
    setError('');
    fetchJson<Status>('/api/health', { signal: signal.signal })
      .then(setStatus).catch(error => { if (error.name !== 'AbortError') setStatus(offline); });
    fetchJson<Project>('/api/project', { signal: signal.signal })
      .then(setProject).catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => signal.abort();
  }, [loadCount]);

  function openInspect(value: { name?: string; resolution?: boolean }) {
    setRequest({ id: ++counter.current, ...value });
    setView('inspect');
    window.location.hash = 'inspect';
    window.scrollTo(0, 0);
  }

  return <>
    <header className="app-shell">
      <div className="shell-inner">
        <a href="#inspect" className="app-brand" aria-label="ATDL Steel Inspection Lab home">
          <span className="brand-mark" aria-hidden="true"><ScanLine size={23} strokeWidth={1.65} /></span>
          <span><strong>ATDL</strong><small>Steel Inspection Lab</small></span>
        </a>
        <nav aria-label="Main navigation">{views.map(({ id, label, icon: Icon }) =>
          <a key={id} href={`#${id}`} aria-current={view === id ? 'page' : undefined}><Icon size={16} strokeWidth={1.7} />{label}</a>)}</nav>
        <span className="device-state" title={status?.ready ? `${status.device_name} · PyTorch ${status.torch} · Ultralytics ${status.ultralytics}` : status?.error ?? ''}>
          <Cpu size={15} /><i className={status?.ready ? 'on' : ''} />
          <strong>{status?.ready ? status.device.startsWith('cuda') ? 'CUDA' : 'CPU' : status ? 'Offline' : 'Connecting'}</strong>
        </span>
      </div>
    </header>
    {error ? <main className="workspace loading"><h1>Project data unavailable</h1><p role="alert">{error}</p>
      <button className="button primary" onClick={() => reload(value => value + 1)}>Retry</button></main> : project && status ? <>
      <Inspect project={project} status={status} active={view === 'inspect'} request={request} />
      {view === 'experiments' && <Study project={project} onCompare={() => openInspect({ resolution: true })} />}
      {view === 'results' && <Results project={project} onOpenCase={name => openInspect({ name })} />}
    </> : <main className="workspace loading" aria-live="polite"><ScanLine size={26} /><h1>Loading the inspection lab</h1></main>}
  </>;
}

createRoot(document.getElementById('root')!).render(<App />);
