import { useEffect, useRef, useState } from 'react';
import { ArrowDownUp, ArrowRight, Boxes, ChartNoAxesCombined, CircuitBoard, FlaskConical, Pause, Play } from 'lucide-react';
import { fetchJson } from './api';
import { fmt, label, whole } from './metrics';
import type { Annotation, ModelId, Project } from './types';

type Topic = 'Dataset' | 'Imbalance' | 'Training' | 'Resolution';
type Transform = 'original' | 'flip' | 'rotate';
const topics: Topic[] = ['Dataset', 'Imbalance', 'Training', 'Resolution'];

function SampleMarkup({ url, box, vertical = false, caption }:
{ url: string; box?: Annotation['xywh']; vertical?: boolean; caption?: string }) {
  return <figure className="study-frame">
    <div className={vertical ? 'vertical' : ''}>
      <img src={url} alt={caption ?? 'Synthetic steel training illustration'} />
      {box && <div className="box truth" style={{
        left: `${(box[0] - box[2] / 2) * 100}%`, top: `${(box[1] - box[3] / 2) * 100}%`,
        width: `${box[2] * 100}%`, height: `${box[3] * 100}%`,
      }} />}
    </div>
    {caption && <figcaption>{caption}</figcaption>}
  </figure>;
}

function Dataset({ project, sample }: { project: Project; sample: Annotation | undefined }) {
  const [offset, setOffset] = useState(0);
  const box = sample ? [...sample.xywh] as Annotation['xywh'] : undefined;
  if (box) box[0] = Math.min(1 - box[2] / 2, Math.max(box[2] / 2, box[0] + offset));
  const width = 640, height = 360;
  const split = [['train', 'Train'], ['val', 'Validation'], ['test', 'Test']] as const;
  const count = split.reduce((total, [key]) => total + project.dataset.image_counts[key], 0);
  const pixel = box ? {
    xmin: Math.round((box[0] - box[2] / 2) * width) + 1,
    ymin: Math.round((box[1] - box[3] / 2) * height) + 1,
    xmax: Math.round((box[0] + box[2] / 2) * width),
    ymax: Math.round((box[1] + box[3] / 2) * height),
  } : null;
  const normalized = pixel ? [
    ((pixel.xmin - 1) + pixel.xmax) / (2 * width), ((pixel.ymin - 1) + pixel.ymax) / (2 * height),
    (pixel.xmax - (pixel.xmin - 1)) / width, (pixel.ymax - (pixel.ymin - 1)) / height,
  ] : null;
  return <div className="research-grid">
    <section className="panel research-major">
      <div className="research-header"><h2>Annotation conversion</h2>
        <p>Move a reference box to see the same geometry in Pascal VOC and YOLO coordinates.</p></div>
      <div className="annotation-layout">
        <SampleMarkup url="/api/sample/image" box={box} caption="Synthetic geometry example" />
        <div className="code-samples">
          <div><span>Pascal VOC · 1-based inclusive</span><pre>{pixel ?
            Object.entries(pixel).map(([key, value]) => `<${key}>${value}</${key}>`).join('\n') : 'Loading box…'}</pre></div>
          <div><span>YOLO · class, center x/y, width, height</span><pre>{normalized && sample ?
            [project.classes.indexOf(sample.class_name), ...normalized.map(value => fmt(value, 4))].join(' ') : 'Loading box…'}</pre></div>
          <label className="select-field">Horizontal box position<input type="range" aria-label="Horizontal box position"
            min={-0.28} max={0.32} step={0.01} value={offset} onChange={event => setOffset(Number(event.currentTarget.value))} /></label>
        </div>
      </div>
    </section>
    <section className="panel research-minor">
      <div className="research-header"><h2>Dataset split</h2><p>Fixed after duplicate checks.</p></div>
      <div className="split-diagram" aria-hidden="true">{split.map(([key]) =>
        <i key={key} style={{ flexGrow: project.dataset.image_counts[key] / count }} />)}</div>
      <div className="data-list split-list">{split.map(([key, name]) =>
        <div key={key}><span><i />{name}</span><strong>{whole(project.dataset.image_counts[key])}</strong></div>)}</div>
      <div className="research-note">The source audit found 1,800 matched pairs. One exact duplicate was excluded, leaving 1,799 images in the converted dataset.</div>
    </section>
  </div>;
}

function Imbalance({ project, sample }: { project: Project; sample?: Annotation }) {
  const [mode, setMode] = useState('full_combined');
  const [transform, setTransform] = useState<Transform>('original');
  const [className, setClassName] = useState('crazing');
  const [power, setPower] = useState(0.5);
  const original = project.dataset.distributions.original;
  const current = project.dataset.distributions[mode];
  const max = Math.max(...Object.values(current.boxes), ...Object.values(original.boxes));
  const repeats = Math.min(3, Math.pow(Math.max(...Object.values(original.boxes)) / original.boxes[className], power));
  const [x, y, w, h] = sample?.xywh ?? [0, 0, 0, 0];
  const adjusted: Annotation['xywh'] = transform === 'flip' ? [1 - x, y, w, h] : transform === 'rotate' ? [1 - y, x, h, w] : [x, y, w, h];
  return <div className="research-grid">
    <section className="panel research-major">
      <div className="research-header row-header"><div><h2>Training exposure by defect</h2>
        <p>{whole(current.images)} image presentations per epoch</p></div>
        <label className="select-field"><span>Training recipe</span>
          <select aria-label="Training recipe" value={mode} onChange={event => setMode(event.target.value)}>
          <option value="original">Original</option><option value="full_repeat">Repeat sampling</option>
          <option value="full_targeted_aug">Targeted augmentation</option><option value="full_combined">Combined</option>
        </select></label></div>
      <div className="bar-chart" aria-label="Box presentations compared with original box counts">
        {project.classes.map(name => <div className="bar-row" key={name}>
          <span>{label(name)}{project.tail_classes.includes(name) && <small> · tail</small>}</span>
          <div className="bar-track"><i style={{ width: `${original.boxes[name] / max * 100}%` }} />
            <b style={{ width: `${current.boxes[name] / max * 100}%` }} /></div>
          <strong>{whole(current.boxes[name])}</strong>
        </div>)}
      </div>
      <div className="chart-legend"><span><i /> Original boxes</span><span><i /> Selected recipe box presentations</span></div>
    </section>

    <section className="panel research-minor">
      <div className="research-header"><h2>Repeat factor</h2><p>Preview how inverse frequency changes a tail class’s sampling weight.</p></div>
      <div className="research-controls">
        <label className="select-field">Tail class<select value={className} onChange={event => setClassName(event.target.value)}>
          {project.tail_classes.map(name => <option key={name} value={name}>{label(name)}</option>)}</select></label>
        <label className="select-field">Frequency exponent <span className="mono">{fmt(power, 2)}</span>
          <input type="range" aria-label="Frequency exponent" min={0} max={1} step={0.05} value={power} onChange={e => setPower(Number(e.target.value))} /></label>
        <div className="readout"><b>{fmt(repeats, 2)}×</b><span>weight before stochastic rounding</span></div>
        <p className="muted">Phase 8 used 0.50 and takes the largest eligible weight on a multi-label image.</p>
      </div>
    </section>

    <section className="panel research-wide">
      <div className="research-header row-header"><div><h2>Targeted augmentation</h2><p>Apply a Phase 8 transform and follow the annotation geometry.</p></div>
        <div className="segmented" aria-label="Augmentation transform">
          {([['original', 'Original'], ['flip', 'Flip · 0.92× brightness'], ['rotate', 'Rotate · 1.08× brightness']] as const)
            .map(([key, name]) => <button key={key} aria-pressed={transform === key} onClick={() => setTransform(key)}>{name}</button>)}
        </div></div>
      <div className="annotation-layout augmentation-layout">
        <SampleMarkup url={`/api/sample/image?transform=${transform}`} box={sample ? adjusted : undefined}
          vertical={transform === 'rotate'} caption="Synthetic transform preview" />
        <div className="code-samples"><div><span>Normalized center and size</span>
          <pre>{transform === 'flip' ? '(1 − x, y, w, h)' : transform === 'rotate' ? '(1 − y, x, h, w)' : '(x, y, w, h)'}</pre></div>
          <div><span>Transformed values</span><pre>{sample ? adjusted.map(value => fmt(value, 4)).join('  ') : 'Loading…'}</pre></div></div>
      </div>
    </section>
  </div>;
}

type TrainingRun = {
  model: string; run: string; seed: number; size: number; batch: number; optimizer: string;
  image_exposures: number; training_seconds: number; history: Record<string, number>[];
};
const chartMetrics: Record<string, { name: string; keys: [string, string]; labels: [string, string] }> = {
  quality: { name: 'Validation AP', keys: ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'], labels: ['mAP50', 'mAP50–95'] },
  loss: { name: 'Box loss', keys: ['train/box_loss', 'val/box_loss'], labels: ['Train box loss', 'Validation box loss'] },
  rate: { name: 'Learning rate', keys: ['lr/pg0', 'lr/pg1'], labels: ['Parameter group 0', 'Parameter group 1'] },
};

function Curve({ rows, epoch, metric }: { rows: TrainingRun['history']; epoch: number; metric: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const definition = chartMetrics[metric];
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rows.length) return;
    function paint() {
      if (!canvas) return;
      const context = canvas.getContext('2d');
      if (!context) return;
      const width = canvas.clientWidth, height = canvas.clientHeight;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio; canvas.height = height * ratio;
      context.scale(ratio, ratio);
      const left = 47, top = 12, right = width - 17, bottom = height - 29;
      const x = (i: number) => left + i / Math.max(1, rows.length - 1) * (right - left);
      const largest = Math.max(...rows.flatMap(row => definition.keys.map(key => row[key] ?? 0)), 0.00001);
      const ceiling = metric === 'quality' ? 1 : largest * 1.1;
      const y = (val: number) => bottom - val / ceiling * (bottom - top);
      context.font = '11px IBM Plex Mono, monospace';
      context.fillStyle = '#586052';
      for (let i = 0; i <= 4; i++) {
        const val = ceiling * i / 4, at = y(val);
        context.strokeStyle = '#e1e4d9';
        context.beginPath(); context.moveTo(left, at); context.lineTo(right, at); context.stroke();
        context.fillText(fmt(val, metric === 'rate' ? 4 : 2), 2, at + 4);
      }
      definition.keys.forEach((key, series) => {
        context.strokeStyle = series === 0 ? '#637718' : '#73928c';
        context.lineWidth = 2;
        context.beginPath();
        rows.forEach((row, index) => {
          if (index === 0) context.moveTo(x(index), y(row[key] ?? 0));
          else context.lineTo(x(index), y(row[key] ?? 0));
        });
        context.stroke();
      });
      context.strokeStyle = '#526325'; context.lineWidth = 1;
      const selected = x(epoch - 1);
      context.beginPath(); context.moveTo(selected, top); context.lineTo(selected, bottom); context.stroke();
      context.fillText('1', left, height - 7);
      context.textAlign = 'right';
      context.fillText(`Epoch ${rows.length}`, right, height - 7);
    }
    const resize = new ResizeObserver(paint);
    resize.observe(canvas);
    paint();
    return () => resize.disconnect();
  }, [rows, epoch, definition, metric]);
  return <canvas ref={ref} className="training-chart" role="img" aria-label={`${definition.name} by training epoch. Epoch ${epoch} selected.`} />;
}

function Training({ project }: { project: Project }) {
  const [model, setModel] = useState<ModelId>('combined');
  const [seed, setSeed] = useState(42);
  const [metric, setMetric] = useState('quality');
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [epoch, setEpoch] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    setPlaying(false); setRun(null); setError('');
    fetchJson<TrainingRun>(`/api/training/${model}/${seed}`, { signal: controller.signal })
      .then(run => { setRun(run); setEpoch(run.history.length); })
      .catch(error => { if (error.name !== 'AbortError') setError(error.message); });
    return () => controller.abort();
  }, [model, seed]);
  useEffect(() => {
    if (!playing || !run) return;
    const timer = window.setInterval(() => setEpoch(value => Math.min(run.history.length, value + 1)), 240);
    return () => window.clearInterval(timer);
  }, [playing, run]);
  useEffect(() => {
    if (run && epoch >= run.history.length) setPlaying(false);
  }, [epoch, run]);
  const row = run?.history[epoch - 1];
  return <div className="research-grid">
    <section className="panel research-major">
      <div className="research-header row-header"><div><h2>Training replay</h2><p>Explore the saved epochs for each seed.</p></div>
        <div className="study-selects">
          <label className="select-field">Run <select aria-label="Training run" value={model} onChange={e => setModel(e.target.value as ModelId)}>
            {Object.values(project.models).map(model => <option key={model.id} value={model.id}>{model.label}</option>)}</select></label>
          <label className="select-field">Seed <select aria-label="Training seed" value={seed} onChange={e => setSeed(Number(e.target.value))}>
            {[42, 43, 44].map(value => <option key={value}>{value}</option>)}</select></label>
        </div></div>
      <div className="research-controls">
        <div className="segmented">{Object.entries(chartMetrics).map(([key, value]) =>
          <button key={key} aria-pressed={metric === key} onClick={() => setMetric(key)}>{value.name}</button>)}</div>
        {run ? <><Curve rows={run.history} epoch={epoch} metric={metric} />
          <div className="chart-legend">{chartMetrics[metric].labels.map((name, index) =>
            <span key={name}><i /> {name} <b>{row ? fmt(row[chartMetrics[metric].keys[index]] ?? 0, metric === 'rate' ? 5 : 3) : '–'}</b></span>)}</div>
          <div className="epoch-controls"><button className="button" onClick={() => {
            if (playing) setPlaying(false);
            else { if (epoch === run.history.length) setEpoch(1); setPlaying(true); }
          }}>{playing ? <Pause size={15} /> : <Play size={15} />}{playing ? 'Pause replay' : 'Replay epochs'}</button>
          <input aria-label="Training epoch" type="range" min={1} max={run.history.length} value={epoch} onChange={e => {
            setPlaying(false); setEpoch(Number(e.currentTarget.value));
          }} /><strong className="mono">{epoch} / {run.history.length}</strong></div></>
          : <p role={error ? 'alert' : 'status'}>{error || 'Loading training log…'}</p>}
      </div>
    </section>
    <section className="panel research-minor">
      <div className="research-header"><h2>Run configuration</h2><p>{run?.run ?? 'Training record'}</p></div>
      {run && <div className="data-list">
        <div><span>Initialization</span><strong>COCO</strong></div>
        <div><span>Input target</span><strong>{run.size} px</strong></div>
        <div><span>Batch size</span><strong>{run.batch}</strong></div>
        <div><span>Optimizer setting</span><strong>{run.optimizer}</strong></div>
        <div><span>Image presentations</span><strong>{whole(run.image_exposures)}</strong></div>
        <div><span>Recorded train time</span><strong>{fmt(run.training_seconds, 0)} s</strong></div>
      </div>}
    </section>
  </div>;
}

function Resolution({ project, onCompare }: { project: Project; onCompare: () => void }) {
  const r = project.resolution;
  return <div className="research-grid">
    <section className="panel research-major">
      <div className="research-header row-header"><div><h2>320 vs 640 pixels</h2>
        <p>Standard YOLO26n · separately trained checkpoints · three paired seeds.</p></div>
        <button className="button primary" onClick={onCompare}>Compare on a frame <ArrowRight size={16} /></button></div>
      <div className="resolution-values">
        {[640, 320].map(size => <div key={size}><h3>{size} px</h3>
          <strong>{fmt(r[`imgsz${size}_mean_tail_macro_AP50`])}</strong><span>Validation tail AP50</span>
          <strong>{fmt(r[`imgsz${size}_mean_inference_ms_per_image`], 2)} ms</strong><span>Recorded GPU model inference per image</span>
        </div>)}
      </div>
    </section>
    <section className="panel research-minor">
      <div className="research-header"><h2>Paired change</h2><p>320 minus 640, averaged across seeds.</p></div>
      <div className="data-list">
        <div><span>Tail AP50</span><strong>{fmt(r.delta_minority_macro_ap50)}</strong></div>
        <div><span>Overall mAP50</span><strong>{fmt(r['delta_metrics/mAP50(B)'])}</strong></div>
        <div><span>GPU inference per image</span><strong>{fmt(r.delta_validation_inference_ms_per_image, 2)} ms</strong></div>
      </div>
    </section>
  </div>;
}

export function Study({ project, onCompare }: { project: Project; onCompare: () => void }) {
  const [topic, setTopic] = useState<Topic>('Dataset');
  const [sample, setSample] = useState<Annotation>();
  useEffect(() => {
    const controller = new AbortController();
    fetchJson<{ boxes: Annotation[] }>('/api/sample/annotation', { signal: controller.signal })
      .then(data => setSample(data.boxes[0])).catch(() => undefined);
    return () => controller.abort();
  }, []);
  const icons = [Boxes, FlaskConical, ChartNoAxesCombined, ArrowDownUp];
  return <main className="workspace">
    <div className="page-heading"><div><h1>Experiments</h1>
      <p>Follow the data, training recipes, and resolution study behind the selected detector.</p></div>
      <div className="heading-meta"><CircuitBoard size={16} /> NEU-DET · Phase 8</div></div>
    <nav className="study-tabs" aria-label="Experiment topics">{topics.map((name, index) => {
      const Icon = icons[index];
      return <button key={name} aria-pressed={topic === name} onClick={() => setTopic(name)}><Icon size={16} />{name}</button>;
    })}</nav>
    {topic === 'Dataset' && <Dataset project={project} sample={sample} />}
    {topic === 'Imbalance' && <Imbalance project={project} sample={sample} />}
    {topic === 'Training' && <Training project={project} />}
    {topic === 'Resolution' && <Resolution project={project} onCompare={onCompare} />}
  </main>;
}
