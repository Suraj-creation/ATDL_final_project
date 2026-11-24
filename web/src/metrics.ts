import type { Annotation, Box, ModelId, Project } from './types';

export const names: Record<string, string> = {
  full_combined: 'Combined YOLO26n', full_targeted_aug: 'Targeted augmentation',
  full_repeat: 'Repeat sampling', yolo26n: 'Standard YOLO26n',
  yolov8n: 'YOLOv8n', yolo11n: 'YOLO11n', yolo11s: 'YOLO11s',
  pitted_surface: 'Pitted surface', 'rolled-in_scale': 'Rolled-in scale',
};

export const label = (key: string) => names[key] ?? key.charAt(0).toUpperCase() + key.slice(1);
export const fmt = (value: number, places = 3) => value.toFixed(places);
export const whole = (value: number) => value.toLocaleString('en-US');

export function timecode(time: number) {
  const minutes = Math.floor(time / 60).toString().padStart(2, '0');
  const seconds = (time % 60).toFixed(2).padStart(5, '0');
  return `${minutes}:${seconds}`;
}

export type Thresholds = { mode: 'project' | 'global'; value: number; overrides: Record<string, number> };

export function cutoff(settings: Thresholds, project: Project, model: ModelId, className: string) {
  if (settings.mode === 'global') return settings.value;
  if (model !== 'combined') return 0.25;
  return settings.overrides[className] ?? project.thresholds[className];
}

export function truthCoordinates(truth: Annotation, size: [number, number]): Box['xyxy'] {
  const [x, y, width, height] = truth.xywh;
  return [(x - width / 2) * size[0], (y - height / 2) * size[1],
    (x + width / 2) * size[0], (y + height / 2) * size[1]];
}

function iou(a: Box['xyxy'], b: Box['xyxy']) {
  const overlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap;
  return union ? overlap / union : 0;
}

export function matchBoxes(boxes: Box[], annotations: Annotation[], size: [number, number]) {
  const unmatched = new Map(annotations.map(truth => [truth.id, truth]));
  const hits = new Set<number>();
  for (const box of [...boxes].sort((a, b) => b.score - a.score)) {
    let best = 0;
    let match = '';
    for (const truth of unmatched.values()) {
      if (truth.class_name !== box.class_name) continue;
      const overlap = iou(box.xyxy, truthCoordinates(truth, size));
      if (overlap >= best) { best = overlap; match = truth.id; }
    }
    if (best >= 0.5) { hits.add(box.id); unmatched.delete(match); }
  }
  return { hits, misses: unmatched, falsePositives: boxes.length - hits.size };
}
