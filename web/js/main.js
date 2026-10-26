/**
 * main.js —— 应用编排：上传 → 渲染 → 播放 → 移调 → 打印/导出 → 质量面板
 *
 * 状态很小，集中在 state 里；视图数据一律由 transposeScore() 从原始 score 派生，
 * 原始数据永远不会被就地修改（保证移调/变调夹可反复叠加、可复位）。
 */

import { ScoreRenderer } from './renderer.js';
import { renderWithFallback } from './abcjs-fallback.js';
import { ScorePlayer } from './player.js';
import { ApiClient, waitForJob } from './api.js';
import { transposeScore } from './transpose.js';
import { normalizePayload, attachTimeline, pickAbc } from './payload.js';
import { PrintController } from './print.js';
import { createEditor, countEdits, buildExport } from './editor.js';
import * as ui from './ui.js';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// 后端在跑时 /api/demo 是首选；纯静态托管时回退到仓库里的样例外
const DEMO_CANDIDATES = [
  '/api/demo',
  '../samples/demo-score.json',
  'samples/demo-score.json',
  '/samples/demo-score.json',
  'demo-score.json',
];

// 指法库：后端 /api/voicings（442 符号 / 1205 把位）优先，静态托管回退到 web/voicings.json
const VOICINGS_URL = new URL('../voicings.json', import.meta.url).href;

const state = {
  raw: null,
  view: null,
  abc: '',
  payload: null,
  transpose: 0,
  capo: 0,
  options: { diagrams: true, tab: true, lyrics: true, metronome: false },
  loop: false,
  jobId: null,
  audioUrl: null,
  voicingJson: null,
  voicingPromise: null,
  devices: null,
  busy: false,
  lastWidth: 0,
};

const dom = {};
const api = new ApiClient('/api');
let renderer = null;
let player = null;
let printer = null;
let editor = null;
let resizeTimer = null;

/* ─────────────────────────── 启动 ─────────────────────────── */

async function init() {
  cacheDom();
  buildRenderer();
  buildPlayer();
  buildPrinter();
  buildEditor();
  bindToolbar();
  bindToggles();
  bindUpload();
  bindPlaybar();
  bindKeyboard();
  bindResize();
  bindPrintLifecycle();
  loadDevices();

  const params = new URLSearchParams(location.search);
  // 允许用 URL 参数预设显示开关（也方便自动化测试）：
  //   ?diagrams=0&tab=0&lyrics=0&metronome=1
  for (const key of ['diagrams', 'tab', 'lyrics', 'metronome']) {
    const value = params.get(key);
    if (value === null) continue;
    const on = value === '1' || value === 'true';
    state.options[key] = on;
    document.querySelector(`[data-toggle="${key}"]`)?.setAttribute('aria-pressed', String(on));
  }
  const custom = params.get('score');
  if (custom) {
    try {
      const res = await fetch(custom, { cache: 'no-store' });
      if (res.ok) {
        applyScore(await res.json(), {});
        return;
      }
    } catch (err) {
      console.warn('自定义 score 加载失败', err);
    }
  }
  // ?job=<id>：直接打开某个已完成任务（可分享链接，也方便自动化验收）
  const jobId = params.get('job');
  if (jobId) {
    try {
      const [score, payload] = await Promise.all([
        api.getScore(jobId),
        api.getPayload(jobId).catch(() => null),
      ]);
      state.jobId = jobId;
      applyScore(score, { audioUrl: api.audioUrlFor({ id: jobId, artifacts: null }, score), payload });
      return;
    } catch (err) {
      console.warn('按 job 加载失败，回退示例谱', err);
      ui.toast('任务未就绪或不存在，已加载示例谱', { kind: 'info' });
    }
  }
  if (params.get('demo') !== '0') await loadDemo({ silent: true });
}

function cacheDom() {
  for (const id of [
    'app', 'tb-title', 'tb-sub', 'transpose-label', 'capo-select', 'export-menu',
    'score', 'empty-state', 'paper-head', 'score-title', 'score-artist', 'factline',
    'paper-foot-meta', 'rail-empty', 'rail-content', 'playbar', 'btn-play', 'pos-bar',
    'pos-time', 'progress-track', 'progress-fill', 'progress-marks', 'speed-range',
    'speed-value', 'play-source', 'upload-modal', 'dropzone', 'file-input',
    'upload-progress', 'stages', 'upload-meter', 'upload-msg', 'toasts', 'print-root',
    'audio', 'opt-key', 'opt-capo', 'opt-difficulty', 'opt-device', 'device-note', 'paper',
    'edit-badge', 'btn-editor',
  ]) {
    dom[id] = $(id);
  }
}

function buildRenderer() {
  renderer = new ScoreRenderer(dom.score, {
    showLyrics: state.options.lyrics,
    showChords: true,
    showDiagrams: state.options.diagrams,
    showTabs: state.options.tab,
    selectable: true,
    showSectionLabels: true,
    // renderer 通过 .bar-hit[data-bar] 回传的是 score 里 1 起的小节号
    onBarClick: (barNumber) => jumpToBar(Number(barNumber)),
  });
}

/**
 * 编辑器接线。
 *
 * 关键约束（见 editor.js 头部注释）：
 * - 编辑写入 state.raw（原始数据），不写 state.view（派生视图），
 *   否则移调/变调夹一变，编辑就丢了。
 * - 编辑后必须重新 render()，让派生层从新的 raw 重算。
 */
function buildEditor() {
  editor = createEditor({
    renderer,
    getState: () => ({
      raw: state.raw,
      view: state.view,
      transpose: state.transpose,
      capo: state.capo,
      options: state.options,
    }),
    getDisplayShift: () => displayShift(),
    getVoicingJson: () => state.voicingJson,
    setRaw: (next) => {
      if (!next) return;
      state.raw = next;
      // 编辑会改动小节结构，之前缓存的播放/高亮映射需要失效
      state.lastWidth = 0;
    },
    render: () => render(),
    toast: (message, opts) => ui.toast(message, opts),
    historyLimit: 60,
  });
  updateEditBadge();
}

/** 质量面板顶部显示「已手工修正 N 处」 */
function updateEditBadge() {
  const count = countEdits(state.raw);
  const badge = dom['edit-badge'];
  if (!badge) return;
  badge.hidden = count === 0;
  badge.textContent = `已手工修正 ${count} 处`;
  const editorBtn = dom['btn-editor'];
  if (editorBtn) editorBtn.setAttribute('aria-pressed', String(Boolean(editor?.enabled)));
}

function buildPlayer() {
  player = new ScorePlayer({
    renderer,
    audioEl: dom.audio,
    onProgress: ({ fraction, barIndex, elapsed, duration, barCount }) =>
      ui.setProgress(fraction, { barIndex, elapsed, duration, totalBars: barCount }),
    onStateChange: (next) => {
      const playing = next === 'playing';
      dom['btn-play']?.setAttribute('data-playing', String(playing));
      dom['btn-play']?.setAttribute('aria-label', playing ? '暂停' : '播放');
    },
  });
}

function buildPrinter() {
  printer = new PrintController({
    printRoot: dom['print-root'],
    getView: () => state.view,
    getVoicings: () => state.voicingJson,
    getOptions: () => state.options,
  });
}

/**
 * 指法库按需加载：只用一次，多个调用共享同一个 Promise。
 * 关闭“和弦图”开关时完全不发请求（首屏省下 577KB）。
 */
function ensureVoicings() {
  if (state.voicingJson) return Promise.resolve(state.voicingJson);
  if (state.voicingPromise) return state.voicingPromise;
  ui.setDiagramStatus('和弦图加载中…');
  state.voicingPromise = (async () => {
    for (const url of ['/api/voicings', VOICINGS_URL]) {
      try {
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) continue;
        const json = await res.json();
        state.voicingJson = json;
        renderer.setVoicingData(json);
        player.setVoicings(renderer.lib);
        ui.setDiagramStatus(null);
        // 指法库到位后重画一次，把“自动生成把位”替换成库里的标准把位
        if (state.raw && state.options.diagrams) render();
        return json;
      } catch (err) {
        /* 试下一个来源 */
      }
    }
    ui.setDiagramStatus(null);
    console.warn('指法库不可用，指法图将按和弦符号自动生成');
    return null;
  })();
  return state.voicingPromise;
}

/* ─────────────────────────── 数据 ─────────────────────────── */

async function loadDemo({ silent = false } = {}) {
  for (const url of DEMO_CANDIDATES) {
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (!res.ok) continue;
      const score = await res.json();
      applyScore(score, { source: url });
      if (!silent) ui.toast('已加载示例谱', { kind: 'ok' });
      return true;
    } catch (err) {
      /* 试下一个路径 */
    }
  }
  if (!silent) {
    ui.toast('找不到示例谱：请从仓库根目录启动静态服务器（python -m http.server）', { kind: 'error', timeout: 6000 });
  }
  return false;
}

/**
 * 探测推理设备（CPU / GPU）并同步到上传弹窗的下拉框。
 *
 * 为什么要问后端而不是前端猜：GPU 能不能用取决于 torch 是不是 CUDA 编译版，
 * 浏览器端完全看不到。后端 /api/health 的 devices 字段是唯一可信来源。
 */
async function loadDevices() {
  const select = dom['opt-device'];
  const note = dom['device-note'];
  try {
    const health = await api.health();
    state.devices = health?.devices || null;
  } catch (err) {
    state.devices = null;
    if (note) {
      note.hidden = false;
      note.dataset.level = 'warn';
      note.textContent = '无法探测推理设备（后端未连接），将使用 CPU。';
    }
    return;
  }
  const devices = state.devices;
  if (!devices) return;

  if (select) {
    select.value = 'auto';
    const gpuOption = select.querySelector('option[value="cuda"]');
    if (gpuOption) {
      // 无 GPU 时仍允许选（用户可能就是想确认回落行为），但标注不可用
      gpuOption.textContent = devices.gpu_usable ? 'GPU（可用）' : 'GPU（本机不可用）';
    }
  }
  updateDeviceNote();
}

/** 根据当前选择 + 后端探测结果，给出「实际会用哪个设备、为什么」的提示 */
function updateDeviceNote() {
  const note = dom['device-note'];
  const select = dom['opt-device'];
  if (!note || !select) return;
  const devices = state.devices;
  const wanted = select.value;

  if (!devices) {
    note.hidden = true;
    return;
  }
  note.hidden = false;

  if (wanted === 'cuda' && !devices.gpu_usable) {
    note.dataset.level = 'warn';
    note.textContent = `GPU 不可用：${devices.summary}。本次会回落 CPU，整曲约需 10–80 分钟。`;
    return;
  }
  if (wanted === 'cuda' && devices.gpu_usable) {
    note.dataset.level = 'ok';
    note.textContent = `将使用 GPU：${devices.summary}`;
    return;
  }
  if (wanted === 'auto') {
    note.dataset.level = devices.gpu_usable ? 'ok' : 'warn';
    note.textContent = devices.gpu_usable
      ? `自动选择 GPU：${devices.summary}`
      : `自动使用 CPU（${devices.summary}）。CPU 上整曲较慢，可先试片段。`;
    return;
  }
  note.dataset.level = 'warn';
  note.textContent = `强制使用 CPU。整曲在 CPU 上约需 10–80 分钟（VocalParse 是主要耗时）。`;
}

function applyScore(score, { audioUrl = null, payload = null } = {}) {
  state.raw = score;
  // payload（aitsm.payload/v1）优先：ABC 由后端统一生成，时间轴也以它为准。
  // 拿不到 payload 时退回 score.json 自带字段，行为与之前完全一致。
  state.payload = normalizePayload(payload);
  state.abc = pickAbc(state.payload, score);
  state.transpose = 0;
  state.capo = Number(score?.meta?.capo) || 0;
  // 显式传入的地址优先；其次是谱面里写的 URL 形态地址（如 /jobs/demo/playback.mp3）。
  // 绝对本地路径（D:\...\playback.mp3）在浏览器里没有意义，一律忽略。
  const declared = score?.meta?.source?.audio_url;
  const fallback = typeof declared === 'string' && declared.startsWith('/') ? declared : null;
  const fromPayload = state.payload?.audioUrl?.startsWith('/') ? state.payload.audioUrl : null;
  const resolved = audioUrl || fromPayload || fallback;
  state.audioUrl = resolved;
  state.jobId = state.jobId || null;

  if (resolved) {
    dom.audio.src = resolved;
    dom.audio.load();
    if (dom['play-source']) {
      dom['play-source'].hidden = false;
      dom['play-source'].textContent = '原曲';
      dom['play-source'].title = '播放跟随原曲音频';
    }
  } else {
    dom.audio.removeAttribute('src');
    if (dom['play-source']) {
      dom['play-source'].hidden = false;
      dom['play-source'].textContent = '合成';
      dom['play-source'].title = '无原曲音频，使用合成伴奏';
    }
  }

  dom['playbar'].hidden = false;
  dom['empty-state'].hidden = true;
  dom.app.dataset.state = 'ready';
  document.title = `${score?.meta?.title || '弹唱谱'} · 弹唱谱工作台`;
  render();
}

function displayShift() {
  const base = Number(state.raw?.meta?.capo) || 0;
  return state.transpose - (state.capo - base);
}

function render() {
  if (!state.raw) return;
  const base = Number(state.raw.meta?.capo) || 0;
  const shift = displayShift();
  const view0 = transposeScore(state.raw, shift, {
    capo: state.capo,
    baseCapo: base,
    soundingShift: state.transpose,
  });
  // payload 的绝对时间轴覆盖转录层估算的小节起止秒（播放与高亮以它为准）；
  // 只改视图不改 raw，编辑/导出仍是原始数据。
  const view = state.payload ? attachTimeline(view0, state.payload) : view0;
  // 给每个音符补一个稳定 id，播放高亮与渲染定位共用
  for (const bar of view.bars || []) {
    for (const [i, note] of (bar.vocal || []).entries()) {
      if (!note.id) note.id = `n-${bar.index}-${i}`;
    }
  }
  state.view = view;

  // 和弦指法库懒加载：只有真正需要画指法图时才请求
  if (state.options.diagrams && !state.voicingJson) ensureVoicings();

  Object.assign(renderer.opts, {
    showLyrics: state.options.lyrics,
    showDiagrams: state.options.diagrams,
    showTabs: state.options.tab,
    showChords: true,
  });

  const width = Math.max(320, Math.floor(dom.score.clientWidth || dom.paper.clientWidth - 64 || 900));
  state.lastWidth = width;
  try {
    renderer.render(view, { width });
  } catch (err) {
    console.error(err);
    ui.toast(`谱面渲染失败：${err.message}`, { kind: 'error', timeout: 6000 });
    fallbackRender(err);
    return;
  }

  ui.renderDocHeader(state.raw, view);
  ui.renderFactline(state.raw, view);
  ui.updateToolbar(state.raw, view, { transpose: state.transpose, capo: state.capo });
  ui.updatePaperFoot(state.raw);
  ui.renderRail(state.raw, { onJumpBar: jumpToBar });
  ui.renderProgressMarks(state.raw.sections || [], state.raw.bars.length);
  updateEditBadge();

  player.setScore(view, { pitchOffset: state.capo - base });
  player.setMetronome(state.options.metronome);
  player.setLoopSection(state.loop);

  ui.setProgress(0, {
    barIndex: 1,
    elapsed: 0,
    duration: player.duration,
    totalBars: view.bars.length,
  });

  printer.invalidate();
}

/**
 * VexFlow 渲染失败时的最后一道兜底：把谱面自带的 ABC 交给 abcjs。
 * abcjs 懒加载、离线环境下会降级为可读提示；这里不抛错、不阻塞主流程。
 */
function fallbackRender(err) {
  if (!state.abc) return;
  renderWithFallback(dom.score, state.abc, { clear: true, reason: err?.message || '' })
    .then((ok) => {
      if (ok) ui.toast('已切换到 abcjs 备用渲染器显示简版谱面', { kind: 'ok', timeout: 5000 });
    })
    .catch((error) => console.warn('[fallback] abcjs 兜底失败', error));
}

/* ─────────────────────────── 交互 ─────────────────────────── */

function bindToolbar() {
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-act], [data-export], [data-toggle]');
    if (!target) {
      closeExportMenu();
      return;
    }
    const act = target.dataset.act;
    const exp = target.dataset.export;
    if (exp) {
      handleExport(exp);
      closeExportMenu();
      return;
    }
    switch (act) {
      case 'trans-down':
        setTranspose(state.transpose - 1);
        break;
      case 'trans-up':
        setTranspose(state.transpose + 1);
        break;
      case 'trans-reset':
        setTranspose(0);
        state.capo = Number(state.raw?.meta?.capo) || 0;
        render();
        break;
      case 'demo':
        ui.closeModal(dom['upload-modal']);
        loadDemo();
        break;
      case 'upload':
        openUpload();
        break;
      case 'close-upload':
        ui.closeModal(dom['upload-modal']);
        break;
      case 'print':
        doPrint();
        break;
      case 'export':
        toggleExportMenu();
        break;
      case 'play':
        player.toggle();
        break;
      case 'stop':
        player.stop();
        ui.setProgress(0, { barIndex: 1, elapsed: 0, duration: player.duration });
        break;
      case 'loop':
        state.loop = !state.loop;
        player.setLoopSection(state.loop);
        target.setAttribute('aria-pressed', String(state.loop));
        break;
      default:
        break;
    }
  });

  dom['capo-select']?.addEventListener('change', (event) => {
    state.capo = Number(event.target.value) || 0;
    render();
  });

  dom['opt-device']?.addEventListener('change', () => updateDeviceNote());
}

function setTranspose(next) {
  const value = clamp(next, -6, 6);
  if (value === state.transpose) return;
  state.transpose = value;
  render();
}

function bindToggles() {
  document.querySelectorAll('[data-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.toggle;
      const pressed = btn.getAttribute('aria-pressed') === 'true';
      btn.setAttribute('aria-pressed', String(!pressed));
      if (key === 'metronome') {
        state.options.metronome = !pressed;
        player.setMetronome(state.options.metronome);
        return;
      }
      // 编辑开关不是显示选项：它切换交互模式，不触发重渲染
      if (key === 'editor') {
        const on = editor?.toggleEnabled();
        ui.toast(on ? '编辑模式：点击和弦、歌词或小节号即可修改' : '已退出编辑模式', {
          kind: on ? 'ok' : 'info',
          timeout: 2600,
        });
        return;
      }
      if (!(key in state.options)) return;
      state.options[key] = !pressed;
      render();
    });
  });
}

function bindPlaybar() {
  const track = dom['progress-track'];
  track?.addEventListener('click', (event) => {
    const rect = track.getBoundingClientRect();
    const fraction = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    player.seek(fraction * (player.duration || 0));
  });
  dom['speed-range']?.addEventListener('input', (event) => {
    const value = Number(event.target.value) || 100;
    dom['speed-value'].textContent = `${value}%`;
    player.setRate(value / 100);
  });
}

function bindKeyboard() {
  document.addEventListener('keydown', (event) => {
    const tag = event.target?.tagName?.toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || event.target?.isContentEditable) return;
    if (event.code === 'Space' && !event.metaKey && !event.ctrlKey && !event.altKey) {
      if (!state.raw) return;
      event.preventDefault();
      player.toggle();
      return;
    }
    if (event.key === 'Escape') {
      ui.closeModal(dom['upload-modal']);
      closeExportMenu();
    }
  });
}

function bindResize() {
  const onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!state.raw) return;
      // 打印/打印预览会临时隐藏 .app，尺寸变化不能触发重排，否则分页文档会被清掉
      if (document.body.classList.contains('is-printing')) return;
      if (typeof window.matchMedia === 'function' && window.matchMedia('print').matches) return;
      const width = Math.floor(dom.score.clientWidth || 0);
      if (Math.abs(width - state.lastWidth) > 40) render();
    }, 180);
  };
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(onResize).observe(dom.score);
  } else {
    window.addEventListener('resize', onResize);
  }
}

function jumpToBar(barNumber) {
  player.seekBar(barNumber);
  renderer.setActiveBar(barNumber - 1);
  renderer.scrollBarIntoView(barNumber - 1);
}

/* ─────────────────────────── 上传 ─────────────────────────── */

function openUpload() {
  ui.openModal(dom['upload-modal']);
  ui.resetStages();
  dom['upload-progress'].hidden = true;
}

function bindUpload() {
  const drop = dom.dropzone;
  const input = dom['file-input'];
  drop?.addEventListener('click', () => input?.click());
  drop?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input?.click();
    }
  });
  input?.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) startJob(file);
    input.value = '';
  });

  const prevent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  ['dragenter', 'dragover'].forEach((type) => {
    drop?.addEventListener(type, (event) => {
      prevent(event);
      drop.classList.add('is-over');
    });
  });
  ['dragleave', 'drop'].forEach((type) => {
    drop?.addEventListener(type, (event) => {
      prevent(event);
      drop.classList.remove('is-over');
    });
  });
  drop?.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) startJob(file);
  });

  // 整页拖拽
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => {
    if (event.dataTransfer?.files?.length) {
      event.preventDefault();
      openUpload();
      startJob(event.dataTransfer.files[0]);
    }
  });
}

/* ─────────────────────────── MusicXML 导入 ─────────────────────────── */

/** 文件名是否是 MusicXML（`.musicxml` / `.xml` / `.mxl`） */
function isMusicXMLFile(file) {
  return /\.(musicxml|xml|mxl)$/i.test(String(file?.name || ''));
}

/**
 * 本地解析 MusicXML 并渲染。
 *
 * 不联网、不上传：读文本 → `parseMusicXML()` → 得到 `aitsm.score/v1` 兼容对象
 * → 走与后端任务完全相同的 `applyScore()` 渲染链路。
 *
 * 解析不完整时把 `warnings` 原样弹出（**不隐藏降级**）：不支持的布局、多 part、
 * 装饰音、关系大小调歧义等都会列出来。
 */
async function importMusicXML(file) {
  state.busy = true;
  try {
    const raw = await file.text();
    const { parseMusicXML } = await import('./musicxml.js');
    const { score, warnings } = parseMusicXML(raw);

    if (!score) {
      ui.toast(`MusicXML 解析失败：${warnings[0] || '未知原因'}`, { kind: 'error', timeout: 7000 });
      return;
    }

    score.meta.title = score.meta.title || file.name.replace(/\.[^.]+$/, '');
    state.jobId = null;
    state.audioUrl = null;
    applyScore(score, { payload: null });

    // 解析告警必须展示：用户要能判断「这份谱读全了没有」
    if (warnings.length) {
      const head = `已导入 ${file.name}，但有 ${warnings.length} 处未完全支持`;
      ui.toast(`${head}：${warnings.slice(0, 2).join('；')}`, { kind: 'info', timeout: 9000 });
      score.meta.warnings = [...(score.meta.warnings || []), ...warnings.map((text) => ({
        level: 'info', code: 'musicxml.import', message: text, stage: 'import',
      }))];
    } else {
      ui.toast(`已导入 ${file.name}（MusicXML，本地解析）`, { kind: 'ok' });
    }
  } catch (err) {
    console.error('MusicXML 导入失败', err);
    ui.toast(`MusicXML 导入失败：${err.message}`, { kind: 'error', timeout: 6000 });
  } finally {
    state.busy = false;
  }
}

async function startJob(file) {
  if (state.busy) return;

  // MusicXML 走**本地解析**，不上传后端：用户可能直接从 MuseScore / Guitar Pro
  // 导出了谱，不需要重新扒音频。解析失败时再退回「当音频上传」的原路径。
  if (isMusicXMLFile(file)) {
    await importMusicXML(file);
    return;
  }

  state.busy = true;
  dom['upload-modal'].hidden = false;
  ui.openModal(dom['upload-modal']);
  dom['upload-progress'].hidden = false;
  ui.resetStages();
  ui.setUploadMeter(0.02, '正在上传音频…');

  // 后端契约：null 表示让分析器自己判断
  const keySel = dom['opt-key']?.value || 'auto';
  const capoSel = dom['opt-capo']?.value || 'auto';
  const device = dom['opt-device']?.value || 'auto';
  const options = {
    key: keySel === 'auto' ? null : keySel,
    capo: capoSel === 'auto' ? null : Number(capoSel),
    difficulty: dom['opt-difficulty']?.value || 'elementary',
    style: 'pop',
    transpose: state.transpose,
    include_tab: true,
    // 设备选择透传给后端；后端会再校验一次，不可用时回落 CPU 并写进 warnings
    device,
    language: 'auto',
  };

  try {
    const created = await api.createJob(file, options, {
      onProgress: (p) => ui.setUploadMeter(0.02 + p * 0.08, `正在上传… ${Math.round(p * 100)}%`),
    });
    state.jobId = created.id;
    ui.setUploadMeter(0.1, '上传完成，开始分析…');
    const done = await waitForJob(api, created.id, {
      onUpdate: (job) => ui.updateStages(job),
    });
    ui.updateStages({ ...done, status: 'done', stage: 'arranging', progress: 1 });
    const score = await api.getScore(created.id);
    // payload 可缺省（老后端 / 生成失败）：失败只降级，不影响谱面显示
    let payload = null;
    try {
      payload = await api.getPayload(created.id);
    } catch (err) {
      console.warn('渲染 payload 不可用，退回 score.json', err);
    }
    applyScore(score, {
      audioUrl: api.audioUrlFor({ id: created.id, artifacts: done?.artifacts }, score),
      payload,
    });
    ui.closeModal(dom['upload-modal']);
    ui.toast('解析完成，谱面已生成', { kind: 'ok' });
  } catch (err) {
    console.error('任务失败', err);
    if (err.status === 409) {
      ui.setUploadMeter(1, '谱面还在生成中，请稍候再试');
      ui.toast('后端还在生成谱面（409），稍后可从任务里重新获取', { kind: 'info', timeout: 5200 });
    } else if (err.status === 0) {
      ui.setUploadMeter(1, '后端服务未就绪');
      ui.toast('后端 API 未就绪，已加载示例谱供预览', { kind: 'error', timeout: 5200 });
      await loadDemo({ silent: true });
      setTimeout(() => ui.closeModal(dom['upload-modal']), 900);
    } else {
      ui.updateStages({ status: 'failed', stage: 'arranging', error: err.message });
      ui.toast(`解析失败：${err.message}`, { kind: 'error', timeout: 6000 });
    }
  } finally {
    state.busy = false;
  }
}

/* ─────────────────────────── 导出 / 打印 ─────────────────────────── */

function toggleExportMenu() {
  const menu = dom['export-menu'];
  const list = menu?.querySelector('.menu__list');
  const trigger = menu?.querySelector('[data-act="export"]');
  if (!list) return;
  const open = list.hidden;
  list.hidden = !open;
  trigger?.setAttribute('aria-expanded', String(open));
}

function closeExportMenu() {
  const menu = dom['export-menu'];
  const list = menu?.querySelector('.menu__list');
  if (list) list.hidden = true;
  menu?.querySelector('[data-act="export"]')?.setAttribute('aria-expanded', 'false');
}

async function handleExport(format) {
  if (!state.raw) return;
  if (format === 'print') {
    doPrint();
    return;
  }
  if (format === 'json') {
    downloadJson();
    return;
  }
  if (format === 'json-clean') {
    // 「干净版」：导出 raw（原始数据，未经移调）并剥掉编辑器加的渲染标记，
    // 同时保留 edits 日志，便于以后追溯哪些地方是手工改过的。
    const payload = buildExport(state.raw, { clean: false });
    const name = `${safeFileName(state.raw?.meta?.title || 'score')}-edited.json`;
    downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }), name);
    const count = countEdits(state.raw);
    ui.toast(count ? `已导出（含 ${count} 处手工修正记录）` : '已导出干净版', { kind: 'ok' });
    return;
  }
  if (!state.jobId) {
    ui.toast('当前谱面不是来自后端任务，已改为下载 JSON 数据', { kind: 'info' });
    downloadJson();
    return;
  }
  try {
    const res = await fetch(api.exportUrl(state.jobId, format));
    if (res.status === 409) {
      ui.toast('这个格式后端还没生成（409），请稍后再试', { kind: 'info' });
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const ext = format === 'midi' ? 'mid' : format === 'musicxml' ? 'xml' : 'json';
    downloadBlob(blob, `${safeFileName(state.raw?.meta?.title || 'score')}.${ext}`);
    ui.toast('导出完成', { kind: 'ok' });
  } catch (err) {
    ui.toast(`导出失败（${err.message}），已改为下载 JSON 数据`, { kind: 'error', timeout: 5200 });
    downloadJson();
  }
}

function downloadJson() {
  const data = JSON.stringify(state.view || state.raw, null, 2);
  downloadBlob(new Blob([data], { type: 'application/json' }), `${safeFileName(state.raw?.meta?.title || 'score')}.json`);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function safeFileName(name) {
  return String(name).replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 80) || 'score';
}

function doPrint() {
  if (!state.raw) return;
  ui.closeModal(dom['upload-modal']);
  printer.print();
}

function bindPrintLifecycle() {
  window.addEventListener('beforeprint', () => {
    document.body.classList.add('is-printing');
    if (!printer.ready) printer.build();
  });
  window.addEventListener('afterprint', () => {
    document.body.classList.remove('is-printing');
  });
}

/* ─────────────────────────── go ─────────────────────────── */

// 自动化截图/回归测试用的只读入口（不参与业务逻辑）。
// 用 defineProperty 定义成不可枚举，避免污染 window 的常规枚举面。
Object.defineProperty(window, '__aitsm', {
  configurable: true,
  value: {
    get state() { return state; },
    get renderer() { return renderer; },
    get player() { return player; },
    get printer() { return printer; },
    get editor() { return editor; },
    get payload() { return state.payload; },
    loadDemo,
    applyScore,
    render,
    jumpToBar,
    setTranspose,
    setEditMode: (on) => editor?.setEnabled(on),
    countEdits: () => countEdits(state.raw),
  },
});

init().catch((err) => {
  console.error(err);
  ui.toast(`初始化失败：${err.message}`, { kind: 'error', timeout: 8000 });
});
