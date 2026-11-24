import type { ModelId, Result } from './types';

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const problem = await response.json().catch(() => null);
    throw new Error(typeof problem?.detail === 'string' ? problem.detail : `Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

export function predict(blob: Blob, model: ModelId, trace = false, activations = false, signal?: AbortSignal) {
  const query = new URLSearchParams({ model, trace: String(trace), activations: String(activations) });
  return fetchJson<Result>(`/api/predict?${query}`, { method: 'POST', body: blob, signal });
}

type FrameRequest = { id: number; source_time: number; model: ModelId };
type StreamReply = { id: number | null; source_time?: number; result?: Result; error?: string };

export class FrameStream {
  private socket: WebSocket | undefined;

  close() {
    this.socket?.close();
    this.socket = undefined;
  }

  async analyze(frame: FrameRequest, blob: Blob): Promise<Result> {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.close();
      const url = new URL('/api/stream', window.location.href);
      url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = this.socket = new WebSocket(url);
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          socket.close();
          reject(new Error('Timed out connecting to the analysis stream.'));
        }, 8000);
        socket.onopen = () => { window.clearTimeout(timeout); resolve(); };
        socket.onclose = () => {
          window.clearTimeout(timeout);
          reject(new Error('The analysis stream disconnected.'));
        };
      });
    }
    const socket = this.socket;
    return new Promise<Result>((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        socket.close();
        reject(new Error('Frame analysis timed out.'));
      }, 15000);
      const fail = (error: Error) => { window.clearTimeout(timeout); reject(error); };
      socket.onclose = () => fail(new Error('The analysis stream disconnected.'));
      socket.onmessage = (event: MessageEvent<string>) => {
        window.clearTimeout(timeout);
        try {
          const reply = JSON.parse(event.data) as StreamReply;
          if (reply.error || !reply.result || reply.id !== frame.id) {
            throw new Error(reply.error ?? 'The stream returned a different frame.');
          }
          resolve(reply.result);
        } catch (error) {
          reject(error);
        }
      };
      try {
        socket.send(JSON.stringify(frame));
        socket.send(blob);
      } catch (error) {
        window.clearTimeout(timeout);
        reject(error);
      }
    });
  }
}

export function exportJson(name: string, value: unknown) {
  exportBlob(name, new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
}

export function exportBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const link = Object.assign(document.createElement('a'), { href: url, download: name });
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
