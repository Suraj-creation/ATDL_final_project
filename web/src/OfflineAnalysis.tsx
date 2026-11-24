import { useEffect, useRef, useState } from 'react';
import { Download, ListVideo, Square } from 'lucide-react';
import { exportJson } from './api';
import { cutoff, fmt, label, type Thresholds } from './metrics';
import type { ModelId, Project, Source } from './types';

type BatchState = {
  expected: number; frames: number; counts: Record<string, number>;
  sumMs: number; complete: boolean; model: ModelId; cutoffs: Record<string, number>;
};
type Event = { kind: 'start'; expected: number } |
  { kind: 'frame'; index: number; detections: Record<string, number>; inference_ms: number } |
  { kind: 'complete'; frames: number; expected: number } |
  { kind: 'error'; error: string };

export function OfflineAnalysis({ source, project, model, thresholds, disabled }:
{ source: Source; model: ModelId; project: Project; thresholds: Thresholds; disabled: boolean }) {
  const [result, setResult] = useState<BatchState | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    setResult(null);
    setError('');
    setRunning(false);
    return () => { controller.current?.abort(); controller.current = null; };
  }, [source]);

  async function start() {
    const own = new AbortController();
    controller.current = own;
    const cutoffs = Object.fromEntries(project.classes.map(name => [name, cutoff(thresholds, project, model, name)]));
    let state: BatchState = { expected: 0, frames: 0, counts: {}, sumMs: 0, complete: false, model, cutoffs };
    setRunning(true);
    setError('');
    setResult(state);
    const { signal } = own;
    try {
      const blob = source.file ?? await fetch(source.url, { signal }).then(response => {
        if (!response.ok) throw new Error('The clip could not be loaded.');
        return response.blob();
      });
      const query = new URLSearchParams({ model, thresholds: JSON.stringify(cutoffs) });
      const response = await fetch(`/api/batch?${query}`, { method: 'POST', body: blob, signal });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail ?? 'Could not start offline analysis.');
      }
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let remainder = '';
      try {
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          const lines = (remainder + chunk.value).split('\n');
          remainder = lines.pop() ?? '';
          for (const line of lines) {
            if (!line) continue;
            const event = JSON.parse(line) as Event;
            if (event.kind === 'error') throw new Error(event.error);
            if (event.kind === 'start') state = { ...state, expected: event.expected };
            if (event.kind === 'complete') state = { ...state, complete: true };
            if (event.kind === 'frame') {
              const counts = { ...state.counts };
              for (const [name, value] of Object.entries(event.detections)) counts[name] = (counts[name] ?? 0) + value;
              state = { ...state, frames: event.index, sumMs: state.sumMs + event.inference_ms, counts };
            }
          }
          if (controller.current === own) setResult(state);
        }
      } finally {
        await reader.cancel().catch(() => undefined);
      }
      if (!state.complete) throw new Error('The clip ended before analysis completed.');
    } catch (error) {
      if (!signal.aborted && controller.current === own) {
        setError(error instanceof Error ? error.message : 'Offline analysis failed.');
      }
    } finally {
      if (controller.current === own) {
        setRunning(false);
        controller.current = null;
      }
    }
  }

  return <section className="panel offline-section" aria-label="Full clip analysis">
    <div className="panel-toolbar">
      <div className="toolbar-name"><ListVideo size={16} /><strong>Full clip analysis</strong></div>
      <div className="toolbar-actions">
        {running ? <button className="button" onClick={() => controller.current?.abort()}><Square size={14} /> Cancel</button> :
          <button className="button" disabled={disabled} onClick={() => void start()}>Analyze every frame</button>}
        {result?.complete && <button className="button" onClick={() => exportJson('atdl-clip-summary.json', {
          ...result, source: source.name, synthetic: source.sample ?? false, checkpoint: project.models[result.model],
        })}><Download size={14} /> Export summary</button>}
      </div>
    </div>
    {error && <p className="notice error" role="alert">{error}</p>}
    {result ? <div className="offline-summary" aria-live="polite">
      <div><strong>{result.frames} frames analyzed</strong><p className="muted">
        {result.complete ? 'Complete' : running ? result.expected ? `Processing · ${result.expected} frames reported by the decoder` : 'Processing…' : 'Partial run'}
        {result.frames > 0 && ` · ${fmt(result.sumMs / result.frames, 1)} ms mean model inference`}</p></div>
      {result.expected > 0 && running && <progress max={result.expected} value={Math.min(result.frames, result.expected)} aria-label="Clip analysis progress" />}
      {result.frames > 0 && <div className="threshold-summary">{project.classes.map(name =>
        <span key={name}>{label(name)} <strong>{result.counts[name] ?? 0}</strong></span>)}</div>}
      <p className="muted">Counts are frame detections. A defect may appear in more than one frame.</p>
    </div> : <p className="detail-empty">Process each decoded frame in order and export the detection counts.</p>}
  </section>;
}
