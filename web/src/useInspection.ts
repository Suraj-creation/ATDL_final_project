import { useCallback, useEffect, useRef, useState } from 'react';
import { FrameStream, predict } from './api';
import type { AnalysisMark, ModelId, Result, Snapshot, Source } from './types';

const sampleSource: Source = {
  kind: 'clip', name: 'Steel coupon · synthetic', url: '/api/sample/clip', sample: true,
};

function capture(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Blob> {
  if (!video.videoWidth || video.seeking) return Promise.reject(new Error('The video frame is still loading.'));
  const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not encode this frame.')), 'image/jpeg', 0.94);
  });
}

function message(error: unknown) {
  return error instanceof Error ? error.message : 'Frame analysis failed.';
}

export function useInspection(isReady: boolean) {
  const [source, setSource] = useState<Source>(sampleSource);
  const [sourceSession, setSourceSession] = useState(0);
  const [model, setModel] = useState<ModelId>('combined');
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [marks, setMarks] = useState<AnalysisMark[]>([]);
  const [compare, setCompare] = useState<Result | null>(null);
  const [compareBusy, setCompareBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [position, setPosition] = useState(0);
  const [sampleRate, setSampleRate] = useState(10);
  const [stats, setStats] = useState({ analyzed: 0, skipped: 0, fps: 0 });
  const [captureWarning, setCaptureWarning] = useState('');
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const camera = useRef<MediaStream | null>(null);
  const stream = useRef(new FrameStream());
  const version = useRef(0);
  const selectedModel = useRef<ModelId>('combined');
  const pendingSeek = useRef(false);
  const cameraIntent = useRef(0);
  const latestStill = useRef(0);
  const stillRequest = useRef<AbortController | null>(null);
  const seekTimer = useRef<number>(0);
  const presentedTime = useRef<number | null>(null);
  const serial = useRef(0);
  const recent = useRef<number[]>([]);
  const current = useRef<Snapshot | null>(null);
  const lastMediaTime = useRef(-1);
  const inFlight = useRef(false);

  useEffect(() => () => {
    stream.current.close();
    stillRequest.current?.abort();
    window.clearTimeout(seekTimer.current);
    camera.current?.getTracks().forEach(track => track.stop());
  }, []);
  useEffect(() => () => { if (source.local) URL.revokeObjectURL(source.url); }, [source]);
  useEffect(() => () => { if (snapshot) URL.revokeObjectURL(snapshot.url); }, [snapshot]);

  const invalidate = useCallback(() => {
    version.current += 1;
    lastMediaTime.current = -1;
    stream.current.close();
    stillRequest.current?.abort();
    return version.current;
  }, []);

  const commit = useCallback((result: Result, blob: Blob, sourceTime: number, id: number, started: number) => {
    const next = { blob, url: URL.createObjectURL(blob), result, sourceTime, id, elapsed: performance.now() - started };
    current.current = next;
    setSnapshot(next);
    const scores: Record<string, number> = {};
    for (const box of result.boxes) {
      scores[box.class_name] = Math.max(scores[box.class_name] ?? 0, box.score);
    }
    setMarks(history => [
      ...history.filter(mark => mark.model !== result.model || Math.abs(mark.time - sourceTime) > 0.01).slice(-359),
      { time: sourceTime, model: result.model, scores },
    ]);
  }, []);

  const runFrame = useCallback(async (
    input: Blob | Promise<Blob>, time: number, selected: ModelId, epoch: number, activations = false,
  ) => {
    setBusy(true);
    setError('');
    const start = performance.now();
    const id = latestStill.current = ++serial.current;
    stillRequest.current?.abort();
    const request = new AbortController();
    stillRequest.current = request;
    try {
      const blob = await input;
      if (version.current !== epoch || latestStill.current !== id) return;
      const result = await predict(blob, selected, true, activations, request.signal);
      if (version.current === epoch && latestStill.current === id) {
        commit(result, blob, time, id, start);
        setPosition(time);
      }
    } catch (error) {
      if (version.current === epoch && latestStill.current === id) setError(message(error));
    } finally {
      if (version.current === epoch && latestStill.current === id) setBusy(false);
    }
  }, [commit]);

  const getCanvas = () => canvas.current ??= document.createElement('canvas');

  function resetStats() {
    recent.current = [];
    setStats({ analyzed: 0, skipped: 0, fps: 0 });
  }

  const clearCompare = useCallback(() => {
    setCompare(null);
    setCompareBusy(false);
  }, []);

  const changeSource = useCallback((next: Source) => {
    invalidate();
    pendingSeek.current = false;
    window.clearTimeout(seekTimer.current);
    cameraIntent.current += 1;
    presentedTime.current = null;
    video.current?.pause();
    camera.current?.getTracks().forEach(track => track.stop());
    camera.current = null;
    setPlaying(false);
    setPosition(0);
    setDuration(0);
    setBusy(false);
    setError('');
    setCaptureWarning('');
    clearCompare();
    current.current = null;
    setSnapshot(null);
    setMarks([]);
    resetStats();
    setSourceSession(previous => previous + 1);
    setSource({ ...next });
  }, [clearCompare, invalidate]);

  useEffect(() => {
    if (source.kind !== 'image' || !isReady) return;
    const epoch = invalidate();
    const input = source.file ?? fetch(source.url).then(response => {
      if (!response.ok) throw new Error('The selected image is unavailable.');
      return response.blob();
    });
    void runFrame(input, 0, model, epoch);
  }, [source, model, isReady, invalidate, runFrame]);

  useEffect(() => {
    if (source.kind !== 'webcam' || !camera.current || !video.current) return;
    const element = video.current;
    const attached = camera.current;
    element.srcObject = attached;
    void element.play().then(() => {
      if (camera.current === attached) setPlaying(true);
    }).catch(error => {
      if (camera.current === attached) setError(message(error));
    });
  }, [source]);

  useEffect(() => {
    const element = video.current;
    if (!element?.requestVideoFrameCallback) return;
    let callback = element.requestVideoFrameCallback(function observe(_, metadata) {
      presentedTime.current = metadata.mediaTime;
      callback = element.requestVideoFrameCallback(observe);
    });
    return () => element.cancelVideoFrameCallback(callback);
  }, [sourceSession]);

  useEffect(() => {
    const element = video.current;
    if (!element || !playing || !isReady) return;
    const epoch = version.current;
    const spacing = 1 / sampleRate;
    let nextSample = -1;

    async function analyzeFrame(sourceTime: number) {
      if (!element || element.seeking || element.readyState < 2 || element.paused || version.current !== epoch) return;
      if (Math.abs(sourceTime - lastMediaTime.current) < 0.005 || sourceTime + 0.001 < nextSample) return;
      if (nextSample < 0) nextSample = sourceTime + spacing;
      else nextSample += (Math.floor(Math.max(0, sourceTime - nextSample) / spacing) + 1) * spacing;
      lastMediaTime.current = sourceTime;
      if (inFlight.current) {
        setStats(s => ({ ...s, skipped: s.skipped + 1 }));
        return;
      }

      const id = ++serial.current;
      const start = performance.now();
      inFlight.current = true;
      try {
        const blob = await capture(element, canvas.current ??= document.createElement('canvas'));
        if (version.current !== epoch) return;
        const result = await stream.current.analyze({ id, model, source_time: sourceTime }, blob);
        if (version.current !== epoch) return;
        commit(result, blob, sourceTime, id, start);
        setPosition(sourceTime);
        const now = performance.now();
        recent.current = [...recent.current.filter(time => time > now - 1000), now];
        setStats(s => ({ analyzed: s.analyzed + 1, skipped: s.skipped, fps: recent.current.length }));
        setError('');
      } catch (error) {
        if (version.current === epoch) {
          element.pause();
          setPlaying(false);
          setError(message(error));
        }
      } finally {
        inFlight.current = false;
      }
    }
    if (element.requestVideoFrameCallback) {
      let callback = element.requestVideoFrameCallback(function tick(_, metadata) {
        callback = element.requestVideoFrameCallback(tick);
        void analyzeFrame(metadata.mediaTime);
      });
      return () => element.cancelVideoFrameCallback(callback);
    }
    const timer = window.setInterval(() => void analyzeFrame(element.currentTime), 1000 / sampleRate);
    return () => window.clearInterval(timer);
  }, [playing, isReady, sampleRate, model, commit]);

  function loadFile(file: File) {
    const kind = file.type.startsWith('video/') || /\.(mp4|webm|mov|m4v|ogg)$/i.test(file.name) ? 'clip' : 'image';
    changeSource({ kind, name: file.name, url: URL.createObjectURL(file), local: true, file });
  }

  async function openCamera() {
    const intent = ++cameraIntent.current;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS.');
      const input = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      if (cameraIntent.current !== intent) {
        input.getTracks().forEach(track => track.stop());
        return;
      }
      changeSource({ kind: 'webcam', name: 'Local webcam', url: '' });
      camera.current = input;
    } catch (error) {
      if (cameraIntent.current !== intent) return;
      const reason = error instanceof DOMException && error.name === 'NotAllowedError'
        ? 'Camera permission was declined. Allow it in your browser or open a recorded clip.'
        : message(error);
      setError(reason);
    }
  }

  function pauseForInspection(trace = true) {
    video.current?.pause();
    setPlaying(false);
    const epoch = invalidate();
    const frozen = current.current;
    if (trace && frozen && !frozen.result.input.preview) {
      void runFrame(frozen.blob, frozen.sourceTime, model, epoch);
    }
    return epoch;
  }

  async function togglePlayback() {
    const element = video.current;
    if (!element || source.kind === 'image') return;
    if (playing) {
      pauseForInspection();
      return;
    }
    if (source.kind === 'webcam' && !camera.current) {
      await openCamera();
      return;
    }
    invalidate();
    pendingSeek.current = false;
    window.clearTimeout(seekTimer.current);
    clearCompare();
    setBusy(false);
    setError('');
    if (element.ended) element.currentTime = 0;
    else if (source.kind === 'clip' && current.current) element.currentTime = current.current.sourceTime;
    resetStats();
    try {
      await element.play();
      setPlaying(true);
    } catch (error) {
      setError(message(error));
    }
  }

  function loaded() {
    const element = video.current;
    if (!element) return;
    setDuration(Number.isFinite(element.duration) ? element.duration : 0);
    if (source.kind === 'clip' && isReady) {
      void runFrame(capture(element, getCanvas()), element.currentTime, model, version.current);
    }
  }

  function seek(time: number) {
    const element = video.current;
    if (!element || !duration) return;
    invalidate();
    window.clearTimeout(seekTimer.current);
    pendingSeek.current = true;
    clearCompare();
    element.pause();
    setPlaying(false);
    const slid = Math.max(0, Math.min(duration - 0.002, time));
    setPosition(slid);
    setBusy(true);
    if (element.seeking || Math.abs(element.currentTime - slid) > 0.001) {
      presentedTime.current = null;
      element.currentTime = slid;
    } else {
      scheduleSoughtFrame(element);
    }
  }

  function frameTime(element: HTMLVideoElement) {
    const presented = presentedTime.current;
    if (presented !== null && Math.abs(presented - element.currentTime) <= 0.05) return presented;
    return source.sample ? Math.floor((element.currentTime + 0.0005) * 24) / 24 : element.currentTime;
  }

  function scheduleSoughtFrame(element: HTMLVideoElement) {
    window.clearTimeout(seekTimer.current);
    seekTimer.current = window.setTimeout(() => {
      if (!pendingSeek.current || element.seeking || video.current !== element) return;
      pendingSeek.current = false;
      void runFrame(capture(element, getCanvas()), frameTime(element), selectedModel.current, version.current);
    }, 70);
  }

  function seeked() {
    const element = video.current;
    if (pendingSeek.current && element?.paused && isReady) scheduleSoughtFrame(element);
  }

  function chooseModel(key: ModelId) {
    if (model === key) return;
    selectedModel.current = key;
    setModel(key);
    clearCompare();
    setPlaying(false);
    video.current?.pause();
    const epoch = invalidate();
    const frozen = current.current;
    if (source.kind === 'image' || pendingSeek.current || video.current?.seeking) return;
    if (frozen) void runFrame(frozen.blob, frozen.sourceTime, key, epoch);
    else if (video.current && video.current.readyState >= 2 && !video.current.seeking) {
      void runFrame(capture(video.current, getCanvas()), video.current.currentTime, key, epoch);
    }
  }

  async function compareWith(key: ModelId) {
    const frozen = current.current;
    if (!frozen || key === model) return;
    const epoch = pauseForInspection();
    setCompare(null);
    setCompareBusy(true);
    try {
      const result = await predict(frozen.blob, key);
      if (epoch === version.current) setCompare(result);
    } catch (error) {
      if (epoch === version.current) setError(message(error));
    } finally {
      if (epoch === version.current) setCompareBusy(false);
    }
  }

  function activateLayers() {
    const frozen = current.current;
    if (!frozen) return;
    if (compareBusy) clearCompare();
    const epoch = pauseForInspection(false);
    void runFrame(frozen.blob, frozen.sourceTime, model, epoch, true);
  }

  function traceCurrent() {
    const frozen = current.current;
    if (frozen && !frozen.result.input.preview && !playing && !busy) {
      void runFrame(frozen.blob, frozen.sourceTime, model, version.current);
    }
  }

  function stopWhenHidden() {
    cameraIntent.current += 1;
    if (playing) {
      video.current?.pause();
      setPlaying(false);
      invalidate();
    }
    if (camera.current) {
      camera.current.getTracks().forEach(track => track.stop());
      camera.current = null;
    }
  }

  function ended() {
    setPlaying(false);
    // Keep the most recently submitted frame valid through natural playback end.
  }

  function mediaError() {
    setBusy(false);
    setPlaying(false);
    setCaptureWarning('The browser could not decode this clip. Open a WebM or a browser-supported MP4.');
  }

  return {
    source, sourceSession, model, snapshot, marks, compare, compareBusy, busy, error, captureWarning, playing, duration,
    position, sampleRate, stats, video, loadFile, openCamera, togglePlayback, loaded, seek, seeked,
    chooseModel, compareWith, clearCompare, activateLayers, traceCurrent, stopWhenHidden, ended,
    mediaError, setSampleRate, setError,
    loadSample: () => changeSource(sampleSource),
    openCase: (name: string) => changeSource({
      kind: 'image', name, url: `/api/cases/${encodeURIComponent(name)}/image`, caseName: name,
    }),
  };
}
