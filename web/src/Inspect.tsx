import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera, ChevronDown, Columns2, Download, Eye, Film, ImagePlus, Settings2,
  Maximize, Minus, Pause, Play, Plus, ScanLine, SkipBack, SkipForward, X,
} from 'lucide-react';
import { exportBlob, exportJson, fetchJson } from './api';
import { Detail, Pipeline } from './InspectionDetail';
import { cutoff, fmt, label, matchBoxes, timecode, truthCoordinates, type Thresholds } from './metrics';
import { useInspection } from './useInspection';
import type { Annotation, Box, ModelId, Project, Result, Snapshot, Stage, Status } from './types';
import { OfflineAnalysis } from './OfflineAnalysis';

type OpenRequest = { id: number; name?: string; resolution?: boolean };
const defaultThresholds: Thresholds = { mode: 'project', value: 0.25, overrides: {} };

function boxLabelStyle(left: number, right: number, width: number) {
  const alignRight = left + right > width;
  const room = alignRight ? right : width - left;
  return {
    left: alignRight ? 'auto' : 0,
    right: alignRight ? 0 : 'auto',
    maxWidth: `${100 * room / Math.max(1, right - left)}%`,
  };
}

function Frame({ image, result, visible, annotations, selected, onSelect, zoom, busy }:
{ image?: string; result?: Result; visible: Box[]; annotations: Annotation[]; selected?: number;
  onSelect?: (box: Box) => void; zoom: number; busy?: boolean }) {
  const size = result?.source_size ?? [640, 360];
  const match = annotations.length > 0 ? matchBoxes(visible, annotations, size) : null;
  return <div className="frame-viewport" aria-busy={busy}>
    {image ? <div className="frame-content" style={{ aspectRatio: `${size[0]} / ${size[1]}`, width: `min(100%, calc(var(--frame-h) * ${size[0] / size[1]}))` }}>
      <div className="frame-zoom" style={{ transform: `scale(${zoom})` }}>
        <img className="steel-image" src={image} alt="Steel surface inspection frame" draggable={false} />
        {annotations.map(entry => {
          const [x1, y1, x2, y2] = truthCoordinates(entry, size);
          return <div key={entry.id} className={`box truth ${match?.misses.has(entry.id) ? 'miss' : ''}`} style={{
            left: `${x1 / size[0] * 100}%`, top: `${y1 / size[1] * 100}%`,
            width: `${(x2 - x1) / size[0] * 100}%`, height: `${(y2 - y1) / size[1] * 100}%`,
          }}><span style={boxLabelStyle(x1, x2, size[0])}>{label(entry.class_name)}</span></div>;
        })}
        {visible.map(box => {
          const [x1, y1, x2, y2] = box.xyxy;
          return <button key={box.id} className={`box detection ${selected === box.id ? 'selected' : ''}`}
            aria-label={`Select ${label(box.class_name)} box ${box.id + 1}`}
            onClick={() => onSelect?.(box)} style={{
              left: `${x1 / size[0] * 100}%`, top: `${y1 / size[1] * 100}%`,
              width: `${(x2 - x1) / size[0] * 100}%`, height: `${(y2 - y1) / size[1] * 100}%`,
            }}><span style={boxLabelStyle(x1, x2, size[0])}>{label(box.class_name)} {fmt(box.score, 2)}</span></button>;
        })}
      </div>
    </div> : <div className="viewer-empty"><ScanLine size={26} /><p>{busy ? 'Analyzing frame…' : 'Open an image or clip to inspect it.'}</p></div>}
  </div>;
}

function Crop({ snapshot, box }: { snapshot: Snapshot; box: Box }) {
  const [x1, y1, x2, y2] = box.xyxy;
  const [sourceW, sourceH] = snapshot.result.source_size;
  const width = Math.max(1, x2 - x1), height = Math.max(1, y2 - y1);
  return <div className="crop" role="img" aria-label={`Crop for ${label(box.class_name)}`}
    style={{ aspectRatio: `${width} / ${height}` }}>
    <img aria-hidden="true" alt="" src={snapshot.url} style={{
      width: `${sourceW / width * 100}%`, height: `${sourceH / height * 100}%`,
      left: `${-x1 / width * 100}%`, top: `${-y1 / height * 100}%`,
    }} />
  </div>;
}

export function Inspect({ project, status, active, request }:
{ project: Project; status: Status; active: boolean; request: OpenRequest | null }) {
  const lab = useInspection(status.ready);
  const {
    source, snapshot, compare, playing, position, duration, stats, sampleRate, busy, model,
  } = lab;
  const [stage, setStage] = useState<Stage>('Source');
  const [layer, setLayer] = useState<'original' | 'detections'>('detections');
  const [thresholds, setThresholds] = useState(defaultThresholds);
  const [selected, setSelected] = useState<number>();
  const [truthOn, setTruthOn] = useState(false);
  const [truth, setTruth] = useState<{ key: string; boxes: Annotation[] } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [compareModel, setCompareModel] = useState<ModelId | null>(null);
  const uploads = useRef<HTMLInputElement>(null);
  const lastRequest = useRef(0);
  const pendingResolution = useRef(false);
  const result = snapshot?.result;

  useEffect(() => {
    if (!active) lab.stopWhenHidden();
  }, [active]);

  useEffect(() => {
    if (!request || request.id === lastRequest.current) return;
    lastRequest.current = request.id;
    if (request.name) {
      setTruthOn(true);
      setLayer('detections');
      lab.openCase(request.name);
    }
    if (request.resolution) {
      if (snapshot) {
        if (model === 'standard') {
          setCompareModel('standard320');
          void lab.compareWith('standard320');
        } else if (model === 'standard320') {
          setCompareModel('standard');
          void lab.compareWith('standard');
        } else {
          pendingResolution.current = true;
          lab.chooseModel('standard');
        }
      } else {
        pendingResolution.current = true;
        lab.chooseModel('standard');
      }
    }
  }, [request, snapshot, model]);

  useEffect(() => {
    if (pendingResolution.current && result?.model === 'standard') {
      pendingResolution.current = false;
      setCompareModel('standard320');
      void lab.compareWith('standard320');
    }
  }, [result]);

  const truthUrl = source.caseName ? `/api/cases/${encodeURIComponent(source.caseName)}/annotation` :
    source.sample && snapshot ? `/api/sample/annotation?position=${Math.min(119, Math.round(snapshot.sourceTime * 24))}` : '';
  useEffect(() => {
    if (!truthOn || !truthUrl || playing) return;
    const controller = new AbortController();
    fetchJson<{ boxes: Annotation[] }>(truthUrl, { signal: controller.signal })
      .then(value => setTruth({ key: truthUrl, boxes: value.boxes }))
      .catch(error => { if (error.name !== 'AbortError') lab.setError(error.message); });
    return () => controller.abort();
  }, [truthOn, truthUrl, playing]);
  const annotations = truthOn && !playing && truth?.key === truthUrl ? truth.boxes : [];

  const visible = useMemo(() => result?.boxes.filter(box => box.score >= cutoff(thresholds, project, result.model, box.class_name)) ?? [],
    [result, thresholds, project]);
  const compareVisible = compare?.boxes.filter(box => box.score >= cutoff(thresholds, project, compare.model, box.class_name)) ?? [];
  const match = result && annotations.length > 0 ? matchBoxes(visible, annotations, result.source_size) : null;
  const selectedBox = playing ? undefined : visible.find(box => box.id === selected);
  const image = snapshot?.url ?? (source.sample ? '/api/sample/image' : source.kind === 'image' ? source.url : undefined);
  const stepSize = 1 / 24;
  const activeTime = snapshot?.sourceTime ?? 0;
  useEffect(() => {
    setCompareModel(null); setSelected(undefined); setZoom(1); setLayer('detections');
  }, [lab.sourceSession]);
  const marks = useMemo(() => lab.marks.filter(mark => mark.model === model).map(mark => ({
    time: mark.time,
    hit: Object.entries(mark.scores).some(([name, score]) => score >= cutoff(thresholds, project, model, name)),
  })), [lab.marks, model, project, thresholds]);
  const detectionTimes = marks
    .filter(mark => mark.hit && Math.abs(mark.time - position) > 0.03)
    .map(mark => mark.time)
    .sort((a, b) => a - b);
  const nextDetection = detectionTimes.find(time => time > position) ?? detectionTimes[0];

  function playPause() {
    if (!playing) setCompareModel(null);
    void lab.togglePlayback();
  }

  function toggleCompare() {
    if (compareModel) {
      setCompareModel(null);
      lab.clearCompare();
      return;
    }
    const other = model === 'combined' ? 'standard' : 'combined';
    setLayer('detections');
    setCompareModel(other);
    void lab.compareWith(other);
  }

  function seek(time: number) {
    setCompareModel(null);
    setSelected(undefined);
    lab.seek(time);
  }

  function chooseStage(value: Stage) {
    setStage(value);
    if (value === 'Preprocess') lab.traceCurrent();
  }

  function exportFrame() {
    if (!snapshot) return;
    exportJson('atdl-inspection.json', {
      source: source.name, source_kind: source.kind, synthetic: Boolean(source.sample),
      frame_id: snapshot.id, source_time: snapshot.sourceTime, round_trip_ms: snapshot.elapsed,
      checkpoint: project.models[snapshot.result.model], thresholds, detections: visible, annotations,
      result: { ...snapshot.result, input: { ...snapshot.result.input, preview: undefined } },
      comparison: compare && { checkpoint: project.models[compare.model], detections: compareVisible, result: compare },
    });
  }

  return <main className="workspace inspect-workspace" hidden={!active}>
    <div className="page-heading">
      <div><h1>Steel Inspection</h1><p>Play a clip, freeze a frame, and see what the detector picked up.</p></div>
      <div className="heading-actions">
        <button className="button" onClick={lab.loadSample}><Film size={16} /> Sample clip</button>
        <button className="button" onClick={() => void lab.openCamera()} disabled={!status.ready}><Camera size={16} /> Webcam</button>
        <button className="button primary" onClick={() => uploads.current?.click()}><ImagePlus size={17} /> Open media</button>
        <input ref={uploads} type="file" aria-label="Open steel media" accept="video/*,image/png,image/jpeg,image/webp"
          className="visually-hidden" onChange={event => {
            const file = event.currentTarget.files?.[0];
            if (file) { setSelected(undefined); setTruthOn(false); lab.loadFile(file); }
            event.currentTarget.value = '';
          }} />
      </div>
    </div>

    {(lab.error || lab.captureWarning || status.error) && <div className="notice error" role="alert">
      {lab.error || lab.captureWarning || status.error}
    </div>}

    <div className="workbench">
      <section className="panel viewer" aria-label="Frame viewer" tabIndex={0}
        onKeyDown={event => {
          if (event.target !== event.currentTarget || !status.ready) return;
          if (event.key === ' ' && !event.repeat && source.kind !== 'image' && (duration || source.kind === 'webcam')) {
            event.preventDefault(); playPause();
          }
          if (duration && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
            event.preventDefault();
            const step = event.shiftKey ? 1 : stepSize;
            seek(position + (event.key === 'ArrowRight' ? step : -step));
          }
        }}>
        <div className="panel-toolbar">
          <div className="player-source">
            {source.kind === 'webcam' ? <Camera size={17} /> : source.kind === 'clip' ? <Film size={17} /> : <ImagePlus size={17} />}
            <div><strong className="source-name" title={source.name}>{source.sample ? 'Sample steel clip' : source.name}</strong>
              <span>{source.sample ? 'Synthetic footage' : source.kind === 'webcam' ? 'Camera feed' : source.kind === 'clip' ? 'Recorded clip' : 'Still image'}</span></div>
          </div>
          <div className="toolbar-actions player-tools">
            <div className="frame-layers" role="group" aria-label="Frame layers">
              <button aria-pressed={layer === 'original'} onClick={() => setLayer('original')}>Original</button>
              <button aria-pressed={layer === 'detections'} onClick={() => setLayer('detections')}>Detections</button>
            </div>
            <button className="icon-button" aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(value => Math.max(1, value - 0.25))}><Minus size={15} /></button>
            <button className="icon-button" aria-label="Reset zoom" onClick={() => setZoom(1)}><Maximize size={14} /></button>
            <button className="icon-button" aria-label="Zoom in" disabled={zoom >= 2} onClick={() => setZoom(value => Math.min(2, value + 0.25))}><Plus size={15} /></button>
          </div>
        </div>
        {source.kind !== 'image' && <video key={lab.sourceSession} ref={lab.video}
          className="media-decoder" src={source.url || undefined} muted playsInline preload="auto"
          onLoadedData={lab.loaded} onSeeked={lab.seeked} onEnded={lab.ended} onError={lab.mediaError} aria-hidden="true" />}
        <div className={`viewer-grid ${compare ? 'viewer-comparison' : ''}`}>
          <div>
            {compare && <div className="split-label">{project.models[model].label} · {visible.length} detections</div>}
            <Frame image={image} result={result} zoom={zoom} annotations={layer === 'detections' ? annotations : []} visible={layer === 'detections' ? visible : []}
              selected={selectedBox?.id} onSelect={box => setSelected(box.id)} busy={busy} />
          </div>
          {compare && <div>
            <div className="split-label">{project.models[compare.model].label} · {compareVisible.length} detections</div>
            <Frame image={image} result={compare} zoom={zoom} annotations={layer === 'detections' ? annotations : []} visible={layer === 'detections' ? compareVisible : []} />
          </div>}
        </div>
        <div className="frame-meta">
          <span>{result ? `${result.source_size.join(' × ')} px` : 'Awaiting a frame'}</span>
          <span className="viewer-state">
            {busy ? <span role="status">Analyzing…</span> : playing ? `${stats.fps} analyzed fps` : snapshot ? 'Frame ready' : 'Waiting for media'}
          </span>
        {snapshot && <details className="runtime-details">
          <summary><span>Browser round trip</span><strong>{fmt(snapshot.elapsed, 0)} ms</strong><ChevronDown size={14} /></summary>
          <div className="timing-content">
            {(['preprocess', 'inference', 'postprocess'] as const).map(part => <div key={part}>
              <span>{label(part)}</span><strong>{fmt(snapshot.result.timing[part], 1)} ms</strong>
            </div>)}
          </div>
        </details>}
        </div>

        <div className="timeline" aria-label="Playback timeline">
          <div className="play-controls">
            <button className="icon-button" aria-label="Step back" disabled={!duration} onClick={() => seek(activeTime - stepSize)}><SkipBack size={15} /></button>
            <button className="play-button" aria-label={playing ? 'Pause' : 'Play'} disabled={!status.ready || source.kind === 'image' || (source.kind === 'clip' && !duration)}
              onClick={playPause}>
              {playing ? <Pause size={16} /> : <Play size={16} fill="currentColor" />}
            </button>
            <button className="icon-button" aria-label="Step forward" disabled={!duration} onClick={() => seek(activeTime + stepSize)}><SkipForward size={15} /></button>
          </div>
          <span className="clock" data-testid="frame-time">{timecode(position)}</span>
          <div className="scrub-track">
            <input type="range" className="seek" aria-label="Seek clip" min="0" max={duration || 1} step="0.001"
              value={Math.min(position, duration)} disabled={!duration} onChange={event => seek(Number(event.currentTarget.value))} />
            {duration > 0 && <div className="analysis-marks" aria-hidden="true">
              {marks.map(mark => <i key={mark.time} data-analysis-mark className={mark.hit ? 'hit' : ''}
                style={{ left: `${Math.min(100, mark.time / duration * 100)}%` }} />)}
            </div>}
          </div>
          <span className="clock">{duration ? timecode(duration) : source.kind === 'webcam' ? 'LIVE' : source.kind === 'image' ? 'STILL' : 'CLIP'}</span>
        </div>
        <div className="video-stats">
          {source.kind !== 'image' ? <>
            <span>{stats.analyzed} analyzed · {stats.skipped} skipped samples</span>
            {duration > 0 && (nextDetection === undefined ?
              <span className="mark-legend"><i /> Frame with detections</span> :
              <button className="jump-detection" onClick={() => seek(nextDetection)} title={`Jump to ${timecode(nextDetection)}`}>
                <i /> Jump to detection
              </button>)}
            <span className="keyboard-hint"><kbd>Space</kbd> Play / pause <kbd>←</kbd><kbd>→</kbd> Step</span>
          </> : <span>Use the pipeline below to explore the selected image.</span>}
        </div>
      </section>

      <section className="panel pipeline-panel" aria-label="Pipeline detail">
        <div className="panel-toolbar pipeline-heading"><h2>Pipeline trace</h2>
          <div className="toolbar-actions">
          {stage === 'Output' && snapshot && <button className="button subtle" onClick={() => {
            const extensions: Record<string, string> = { 'image/png': 'png', 'image/webp': 'webp', 'image/jpeg': 'jpg' };
            exportBlob(`atdl-frame.${extensions[snapshot.blob.type] ?? 'image'}`, snapshot.blob);
          }}><Download size={14} /> Export frame</button>}
          <button className="button subtle" disabled={!snapshot} onClick={exportFrame}><Download size={15} /> Export</button>
        </div></div>
        <Pipeline stage={stage} choose={chooseStage} />
        <div className="stage-content" key={stage}><Detail stage={stage} snapshot={snapshot} source={source} project={project} thresholds={thresholds}
          visible={visible} busy={busy} playing={playing} trace={lab.traceCurrent} activate={() => {
            if (lab.compareBusy) setCompareModel(null);
            lab.activateLayers();
          }} /></div>
      </section>
      <aside className="panel inspection-rail" aria-label="Inspection controls">
        <div className="panel-toolbar rail-heading"><h2>Findings</h2>
          <span className="tally" data-testid="box-count">{visible.length} visible</span></div>
        <div className="rail-content">
          <div className="box-list">
            <p className="rail-model">{project.models[model].label} <span>· {project.models[model].size} px</span></p>
            <div className="list-header"><span>Detected class</span><span>Score</span></div>
            {visible.map(box => <button className={`box-row ${selectedBox?.id === box.id ? 'chosen' : ''}`} key={box.id}
              aria-pressed={selectedBox?.id === box.id} onClick={() => { setSelected(box.id); setLayer('detections'); }}>
              <span><i />{label(box.class_name)}</span><b>{fmt(box.score, 2)}</b>
            </button>)}
            {!visible.length && <p className="empty-list">{snapshot ? 'No detections at these settings.' : 'Run a frame to see defect boxes.'}</p>}
          </div>
          {selectedBox && snapshot && <div className="selected-crop">
            <div><strong>{label(selectedBox.class_name)}</strong><button className="icon-button" aria-label="Close crop" onClick={() => setSelected(undefined)}><X size={14} /></button></div>
            <Crop snapshot={snapshot} box={selectedBox} />
            <span className="mono">{selectedBox.xyxy.map(value => fmt(value, 0)).join(' / ')} px</span>
          </div>}
          {source.caseName || source.sample ? <button className="button wide-label" aria-pressed={truthOn}
            onClick={() => { setTruthOn(value => !value); setLayer('detections'); }} disabled={!snapshot || playing}><Eye size={15} />
            {truthOn ? 'Hide annotations' : 'Show annotations'}</button> : null}
          {match && <div className="match-count" aria-label="Annotation matching at IoU 0.50">
            <span>{match.hits.size} matched</span> <span>{match.falsePositives} extra</span> <span>{match.misses.size} missed</span></div>}
          <button className="button wide-label compare-action" onClick={toggleCompare}
            disabled={!snapshot || busy || lab.compareBusy}>
            <Columns2 size={16} /> {compareModel ? 'Exit comparison' : 'Compare models'}
          </button>
          {(compareModel || lab.compareBusy) && <>          <div className="compare-control">
            <label><Columns2 size={15} /><span>Compare with</span>
              <select aria-label="Compare with checkpoint" disabled={!snapshot || busy || lab.compareBusy}
                value={compareModel ?? ''} onChange={event => {
                  const value = event.currentTarget.value as ModelId | '';
                  setCompareModel(value || null);
                  if (value) void lab.compareWith(value);
                  else lab.clearCompare();
                }}>
                <option value="">Off</option>
                {Object.values(project.models).filter(item => item.id !== model).map(item =>
                  <option key={item.id} value={item.id}>{item.label}</option>)}
              </select>
            </label>
            {lab.compareBusy && <p className="muted" role="status">Analyzing the same frame…</p>}
            {compare && <p className="muted">{thresholds.mode === 'global' ? 'Shared global cutoff.' : 'Each checkpoint uses its own default cutoff.'}</p>}
          </div>
</>}
          <details className="model-settings">
            <summary><Settings2 size={16} /><span>Model and thresholds</span><ChevronDown size={15} /></summary>
            <div className="settings-body">
          <label className="select-field">Checkpoint
            <select value={model} aria-label="Checkpoint" onChange={event => {
              setCompareModel(null); setSelected(undefined); lab.chooseModel(event.target.value as ModelId);
            }}>
              {Object.values(project.models).map(item => <option value={item.id} key={item.id}>{item.label} · {item.size}px · seed {item.seed}</option>)}
            </select>
          </label>
          <div className="filter-block">
            <div className="filter-title"><label htmlFor="global-confidence">Confidence</label>
              <span className="mono">{thresholds.mode === 'global' ? fmt(thresholds.value, 2) : model === 'combined' ? 'Per class' : '0.25'}</span></div>
            <input id="global-confidence" type="range" min="0.01" max="1" step="0.01" value={thresholds.value}
              onChange={event => setThresholds({ mode: 'global', value: Number(event.currentTarget.value), overrides: {} })} />
            <div className="filter-foot"><span>{thresholds.mode === 'project' ? 'Checkpoint defaults' : 'Global cutoff'}</span>
              <button className="text-button" onClick={() => setThresholds(defaultThresholds)}>Reset</button></div>
            {model === 'combined' && <details className="class-cutoffs">
              <summary>Per-class thresholds <ChevronDown size={13} /></summary>
              {project.classes.map(name => <label key={name}>{label(name)}
                <input type="number" min="0.01" max="1" step="0.01" aria-label={`${label(name)} confidence threshold`}
                  value={thresholds.overrides[name] ?? project.thresholds[name]}
                  onChange={event => {
                    const value = event.currentTarget.valueAsNumber;
                    if (Number.isFinite(value)) setThresholds(current => ({
                      ...current, mode: 'project', overrides: { ...current.overrides, [name]: Math.max(0.01, Math.min(1, value)) },
                    }));
                  }} />
              </label>)}
            </details>}
          </div>
              {source.kind !== 'image' && <label className="select-field">Sample rate
                <select aria-label="Sample rate" value={sampleRate} onChange={event => lab.setSampleRate(Number(event.currentTarget.value))}>
                  <option value={5}>5 fps</option><option value={10}>10 fps</option><option value={15}>15 fps</option><option value={24}>24 fps</option>
                </select></label>}
            </div>
          </details>
        </div>
      </aside>

    </div>
    {active && source.kind === 'clip' && <OfflineAnalysis source={source} model={model} project={project} thresholds={thresholds}
      disabled={playing || !status.ready} />}
  </main>;
}
