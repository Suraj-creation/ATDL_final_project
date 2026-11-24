export type ModelId = 'combined' | 'standard' | 'standard320';
export type View = 'inspect' | 'experiments' | 'results';
export type Stage = 'Source' | 'Preprocess' | 'Model' | 'Filter' | 'Output';
export type Box = { id: number; class_name: string; score: number; xyxy: [number, number, number, number] };
export type Annotation = { id: string; class_name: string; xywh: [number, number, number, number] };

export type Model = {
  id: ModelId; label: string; run: string; checkpoint: string; sha256: string;
  size: number; seed: number; parameters: number; batch: number; epochs: number;
};
export type Result = {
  model: ModelId; device: string; source_size: [number, number];
  input: { shape: number[]; dtype: string; scale: number; padding: number[]; preview: string | null; rect: boolean };
  boxes: Box[];
  activations: { layer: number; label: string; shape: number[]; image: string }[];
  candidate_floor: number; max_candidates: number;
  postprocess: { nms: boolean; iou: number | null };
  timing: Record<string, number>;
};
export type AnalysisMark = { time: number; model: ModelId; scores: Record<string, number> };
export type Snapshot = {
  blob: Blob; url: string; result: Result; sourceTime: number; id: number; elapsed: number;
};
export type Source = {
  kind: 'clip' | 'image' | 'webcam';
  name: string; url: string; sample?: boolean; caseName?: string; local?: boolean; file?: Blob;
};
export type Status = {
  ready: boolean; error: string | null; device: string; device_name: string;
  torch: string | null; ultralytics: string; loaded_models: ModelId[];
};
export type Operating = {
  confidence_threshold: number; tp: number; fp: number; fn: number;
  test_precision_at_iou50: number; test_recall_at_iou50: number; test_f1_at_iou50: number;
};
export type Project = {
  classes: string[]; tail_classes: string[]; models: Record<ModelId, Model>; default_model: ModelId;
  dataset: {
    image_counts: Record<string, number>;
    distributions: Record<string, { images: number; boxes: Record<string, number>; repeat_rates?: Record<string, number> }>;
  };
  training: Record<ModelId, Record<string, number>[]>;
  runs: Record<ModelId, { optimizer: string; training_seconds: number; image_exposures: number;
    validation: Record<string, number>; per_class: Record<string, Record<string, number>> }>;
  candidates: {
    candidate: string; n_seeds: number; minority_macro_AP50_mean: number; minority_macro_AP50_sd: number;
    mAP50_mean: number; mAP50_95_mean: number; validation_inference_ms_mean: number;
  }[];
  candidate_runs: Record<string, { seed: number; tail_ap50: number; mAP50: number }[]>;
  resolution: Record<string, number>;
  thresholds: Record<string, number>;
  test: { mAP50: number; mAP50_95: number; F1: number; precision: number; recall: number; images: number;
    per_class_ap: Record<string, Record<string, number>>; per_class_operating: Record<string, Operating> };
  validation_cases: { name: string; available: boolean; false_positives: number; missed_boxes: number }[];
};
