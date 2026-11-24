import { ArrowRight, Braces, Cpu, Focus, Layers3, ScanLine, SlidersHorizontal } from 'lucide-react';
import { cutoff, fmt, label, timecode, type Thresholds } from './metrics';
import type { Box, Project, Snapshot, Source, Stage } from './types';

export const stages = [
  { name: 'Source', icon: Focus },
  { name: 'Preprocess', icon: Layers3 },
  { name: 'Model', icon: Cpu },
  { name: 'Filter', icon: SlidersHorizontal },
  { name: 'Output', icon: ScanLine },
] as const;

export function Pipeline({ stage, choose }: { stage: Stage; choose: (stage: Stage) => void }) {
  return <div className="pipeline" aria-label="Inference pipeline">
    {stages.map(({ name, icon: Icon }, index) => <div className="pipe-node" key={name}>
      <button type="button" aria-pressed={stage === name} onClick={() => choose(name)}>
        <Icon size={15} /> {name}
      </button>
      {index < stages.length - 1 && <ArrowRight className="pipe-arrow" size={13} aria-hidden="true" />}
    </div>)}
  </div>;
}

export function Detail({ stage, snapshot, source, project, thresholds, visible, busy, playing, trace, activate }:
{ stage: Stage; snapshot: Snapshot | null; source: Source; project: Project; thresholds: Thresholds;
  visible: Box[]; busy: boolean; playing: boolean; trace: () => void; activate: () => void }) {
  if (!snapshot) return <p className="detail-empty">Analyze a frame to inspect its pipeline.</p>;
  const { result, sourceTime } = snapshot;
  const model = project.models[result.model];
  if (stage === 'Source') return <div className="detail-line">
    <div><strong>{source.name}</strong><p>{source.sample ? 'Generated steel texture for testing the workflow.' : 'Frame submitted to the detector.'}</p></div>
    <div className="detail-value"><span>Frame size</span><strong>{result.source_size.join(' × ')} px</strong></div>
    {source.kind !== 'image' && <div className="detail-value"><span>Source time</span><strong>{timecode(sourceTime)}</strong></div>}
  </div>;
  if (stage === 'Preprocess') return <div className="detail-stack">
    <div className="detail-line">
      <div><strong>Resize and letterbox</strong><p>Preserve aspect ratio, convert to RGB and normalize pixel values to 0–1.</p></div>
      <div className="detail-value"><span>Input tensor</span><strong>{result.input.shape.join(' × ')}</strong></div>
      <div className="detail-value"><span>Scale</span><strong>{fmt(result.input.scale, 3)}×</strong></div>
      <div className="detail-value"><span>Padding L/T/R/B</span><strong>{result.input.padding.join(' / ')} px</strong></div>
    </div>
    {result.input.preview ? <div className="prep-images">
      <figure><img src={snapshot.url} alt="Frame sent to the detector" /><figcaption>Source frame</figcaption></figure>
      <figure><img src={result.input.preview} alt="Actual normalized and padded tensor rendered as an image" /><figcaption>Actual model input · {result.input.dtype}</figcaption></figure>
    </div> : <button className="button" disabled={busy || playing} onClick={trace}>{playing ? 'Pause to view the actual input' : 'Capture model input'}</button>}
  </div>;
  if (stage === 'Model') return <div className="detail-stack">
    <div className="detail-line">
      <div><strong>{model.label}</strong><p>COCO initialization · NEU-DET fine-tuning · seed {model.seed}</p></div>
      <div className="network"><span>Backbone</span><ArrowRight size={13} /><span>Feature fusion</span><ArrowRight size={13} /><span>Detection head</span></div>
      <button className="button" onClick={activate} disabled={busy}>{busy ? 'Capturing…' : 'Capture activations'}</button>
    </div>
    {result.activations.length > 0 && <>
      <div className="activation-grid">{result.activations.map(tap => <figure key={tap.layer}>
        <img src={tap.image} alt={`Mean absolute activation for layer ${tap.layer}`} />
        <figcaption>{tap.label} · layer {tap.layer}<small>{tap.shape.join(' × ')}</small></figcaption>
      </figure>)}</div>
      <p className="muted">Mean absolute response across channels, normalized within each layer. These maps show feature activity.</p>
    </>}
  </div>;
  if (stage === 'Filter') return <div className="detail-stack">
    <div className="detail-line">
      <div><strong>{visible.length} kept · {result.boxes.length - visible.length} hidden</strong>
        <p>Model candidates above {fmt(result.candidate_floor, 2)}, up to {result.max_candidates} per frame.</p></div>
      <div className="detail-value"><span>Runtime overlap filter</span>
        <strong>{result.postprocess.nms ? `NMS · IoU ${fmt(result.postprocess.iou ?? 0, 2)}` : 'End-to-end head'}</strong></div>
    </div>
    <div className="threshold-summary">{project.classes.map(name => <span key={name}>{label(name)} <strong>{fmt(cutoff(thresholds, project, result.model, name), 2)}</strong></span>)}</div>
  </div>;
  return <div className="detail-line">
    <div><strong><Braces size={15} /> Run record</strong><p>Download the exact source frame or export its coordinates, filters, timings and checkpoint hash.</p></div>
    <div className="detail-value"><span>Coordinate space</span><strong>Source pixels · XYXY</strong></div>
    <div className="detail-value"><span>Checkpoint hash</span><strong>{model.sha256.slice(0, 12)}</strong></div>
  </div>;
}
