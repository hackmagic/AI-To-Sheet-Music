/**
 * api.js —— 后端任务 API 客户端
 *
 * 后端约定（见 docs/02-architecture.md）：
 *   POST   /api/jobs                    multipart: file + options(JSON 字符串)  -> 202 { job_id, status }
 *   GET    /api/jobs/{id}               -> { id,status,stage,stage_label,percent,message,error,warnings[],artifacts{} }
 *   GET    /api/jobs/{id}/score         -> score.json（未完成返回 409）
 *   GET    /api/jobs/{id}/export/{fmt}  -> musicxml | midi | json（未生成返回 409）
 *   GET    /jobs/{id}/audio/playback.mp3 -> 原曲/伴奏音频（静态目录）
 *   GET    /api/demo                    -> 示例 score
 *   GET    /api/voicings                -> 指法库
 *   GET    /api/health                  -> 引擎健康状态
 *
 * 所有方法在网络层失败时抛 ApiError（status=0 表示连不上），调用方负责降级。
 */

const DEFAULT_BASE = '/api';

export class ApiError extends Error {
  constructor(message, { status = 0, code = '', detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const DONE_STATES = new Set(['done', 'completed', 'complete', 'succeeded', 'success', 'finished', 'ready']);
const FAIL_STATES = new Set(['failed', 'failure', 'error', 'cancelled', 'canceled', 'aborted']);

export class ApiClient {
  constructor(base = DEFAULT_BASE) {
    this.base = String(base || DEFAULT_BASE).replace(/\/+$/, '');
  }

  url(path) {
    return `${this.base}${path}`;
  }

  async _request(path, { method = 'GET', body = null, signal = null, timeout = 20000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const onAbort = () => controller.abort();
    if (signal) signal.addEventListener('abort', onAbort, { once: true });
    try {
      const res = await fetch(this.url(path), {
        method,
        body,
        signal: controller.signal,
        headers: body && !(body instanceof FormData) ? { 'content-type': 'application/json' } : undefined,
      });
      if (!res.ok) {
        let detail = null;
        try {
          detail = await res.json();
        } catch {
          detail = await res.text().catch(() => null);
        }
        const msg = detail?.error || detail?.message || detail?.detail || `请求失败 HTTP ${res.status}`;
        throw new ApiError(String(msg), { status: res.status, detail });
      }
      return res;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      if (err.name === 'AbortError') throw new ApiError('请求超时', { status: 0 });
      throw new ApiError('无法连接后端服务', { status: 0, detail: err });
    } finally {
      clearTimeout(timer);
      if (signal) signal.removeEventListener('abort', onAbort);
    }
  }

  /** 上传文件创建任务；onProgress(0..1) 反映真实上传进度 */
  createJob(file, options = {}, { onProgress = null, signal = null } = {}) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append('file', file, file.name || 'upload');
      form.append('options', JSON.stringify(options || {}));
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.url('/jobs'));
      xhr.timeout = 120000;
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable && onProgress) onProgress(ev.loaded / ev.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText || '{}');
            const id = data.job_id || data.id || data.jobId;
            if (!id) return reject(new ApiError('后端未返回 job id', { status: xhr.status, detail: data }));
            resolve({ id, raw: data });
          } catch (err) {
            reject(new ApiError('后端响应不是合法 JSON', { status: xhr.status, detail: err }));
          }
        } else {
          reject(new ApiError(`上传失败 HTTP ${xhr.status}`, { status: xhr.status, detail: xhr.responseText }));
        }
      };
      xhr.onerror = () => reject(new ApiError('上传时无法连接后端服务', { status: 0 }));
      xhr.ontimeout = () => reject(new ApiError('上传超时', { status: 0 }));
      xhr.onabort = () => reject(new ApiError('上传已取消', { status: 0 }));
      if (signal) signal.addEventListener('abort', () => xhr.abort(), { once: true });
      xhr.send(form);
    });
  }

  async getJob(id, { signal = null } = {}) {
    const res = await this._request(`/jobs/${encodeURIComponent(id)}`, { signal });
    return res.json();
  }

  async getScore(id, { signal = null } = {}) {
    try {
      const res = await this._request(`/jobs/${encodeURIComponent(id)}/score`, { signal, timeout: 30000 });
      return res.json();
    } catch (err) {
      // 409 = 任务还没跑到产出谱面，不是错误，UI 应显示“生成中”
      if (err instanceof ApiError && err.status === 409) {
        throw new ApiError('谱面还在生成中', { status: 409, code: 'not_ready', detail: err.detail });
      }
      throw err;
    }
  }

  /** 任务分析细节（质量面板的原始置信度） */
  async getAnalysis(id, { signal = null } = {}) {
    const res = await this._request(`/jobs/${encodeURIComponent(id)}/analysis`, { signal, timeout: 30000 });
    return res.json();
  }

  async cancelJob(id) {
    try {
      await this._request(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE', timeout: 8000 });
    } catch {
      /* 取消失败不阻塞 UI */
    }
  }

  exportUrl(id, fmt) {
    return this.url(`/jobs/${encodeURIComponent(id)}/export/${encodeURIComponent(fmt)}`);
  }

  /** 原曲/伴奏音频的静态 URL（后端把 work/jobs 挂到了 /jobs） */
  mediaPath(id, file = 'playback.mp3') {
    return `/jobs/${encodeURIComponent(id)}/${file}`;
  }

  /** 优先用谱面里写好的 audio_url，其次按任务 id 兜底 */
  audioUrlFor(job, score = null) {
    const fromScore = score?.meta?.source?.audio_url;
    if (fromScore) return fromScore;
    const fromJob = job?.artifacts?.playback || job?.artifacts?.audio;
    if (fromJob && /^(?:https?:)?\//.test(fromJob)) return fromJob;
    return job?.id ? this.mediaPath(job.id) : '';
  }

  async getDemo({ signal = null } = {}) {
    const res = await this._request('/demo', { signal, timeout: 15000 });
    return res.json();
  }

  async getVoicings({ signal = null } = {}) {
    const res = await this._request('/voicings', { signal, timeout: 30000 });
    return res.json();
  }

  async health({ signal = null, timeout = 3000 } = {}) {
    const res = await this._request('/health', { signal, timeout });
    return res.json();
  }

  /** 轻量探活：后端不在时 2s 内失败 */
  async ping({ timeout = 2000 } = {}) {
    try {
      await this._request('/health', { timeout });
      return true;
    } catch (err) {
      return err.status > 0 && err.status < 500;
    }
  }
}

/**
 * 轮询任务直到完成/失败。
 * onUpdate(job) 每次拿到新状态都会调用（UI 用来更新阶段条）。
 */
export async function waitForJob(api, id, { onUpdate = null, signal = null, intervalMs = 900, timeoutMs = 20 * 60 * 1000 } = {}) {
  const started = Date.now();
  let last = null;
  for (;;) {
    if (signal?.aborted) throw new ApiError('已取消', { status: 0 });
    const job = await api.getJob(id, { signal });
    last = job;
    onUpdate?.(job, last);
    const status = String(job.status || job.state || '').toLowerCase();
    if (DONE_STATES.has(status)) return job;
    if (FAIL_STATES.has(status)) {
      throw new ApiError(job.error || job.message || '任务失败', { status: 0, code: status, detail: job });
    }
    if (Date.now() - started > timeoutMs) {
      throw new ApiError('任务超时', { status: 0, detail: last });
    }
    await sleep(intervalMs);
  }
}
