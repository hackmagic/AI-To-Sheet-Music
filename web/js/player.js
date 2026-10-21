/**
 * player.js —— Web Audio 播放器
 *
 * 两种时钟：
 *  - synth：用 AudioContext 时间调度合成音（和弦扫弦 + 旋律 + 节拍器）；
 *  - audio：有原曲音频时以 <audio> 为时钟，谱面光标跟随音频时间，
 *    合成旋律静音，避免与原曲打架。
 *
 * 时间轴一律使用 score 的秒单位（bar.start_sec / bar.end_sec）。
 */

import { midiToFreq } from './music.js';

const LOOKAHEAD = 0.14; // 提前调度的秒数
const TICK_MS = 40;
const DEFAULT_TUNING = [40, 45, 50, 55, 59, 64];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/** 二分：返回最后一个 start <= t 的索引 */
function indexAt(items, t) {
  let lo = 0;
  let hi = items.length - 1;
  let ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].start <= t + 1e-6) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

export class ScorePlayer {
  constructor({
    renderer = null,
    audioEl = null,
    onBarChange = null,
    onNoteChange = null,
    onStateChange = null,
    onProgress = null,
  } = {}) {
    this.renderer = renderer;
    this.audioEl = audioEl;
    this.onBarChange = onBarChange;
    this.onNoteChange = onNoteChange;
    this.onStateChange = onStateChange;
    this.onProgress = onProgress;

    this.ctx = null;
    this.master = null;
    this.score = null;
    this.bars = [];
    this.events = [];
    this.sections = [];
    this.voiceLib = null;
    this.pitchOffset = 0;
    this.rate = 1;
    this.metronome = false;
    this.loopSection = false;
    this.mode = 'synth';

    this.state = 'stopped';
    this.cursor = 0;
    this.anchorCtx = 0;
    this.anchorScore = 0;
    this.timer = null;
    this.raf = null;
    this.sources = [];
    this.currentBar = -1;
    this.currentNote = null;
    this.lastProgressAt = 0;
    this._lastUiAt = 0;
    this._resumeAudioAt = 0;
  }

  /* ───────────── 数据 ───────────── */

  setScore(score, { pitchOffset = 0 } = {}) {
    this.stop();
    this.score = score || null;
    this.pitchOffset = pitchOffset;
    this.bars = [];
    this.events = [];
    this.sections = score?.sections ? [...score.sections] : [];
    if (!score) {
      this.onStateChange?.('stopped');
      return;
    }
    const tempo = score.meta?.tempo || 84;
    const ts = score.meta?.time_signature || [4, 4];
    const beatsPerBar = (ts[0] || 4) * (4 / (ts[1] || 4));
    let cursor = 0;
    for (const [i, bar] of (score.bars || []).entries()) {
      const fallbackDur = beatsPerBar * 60 / tempo;
      let start = typeof bar.start_sec === 'number' ? bar.start_sec : cursor;
      let end = typeof bar.end_sec === 'number' ? bar.end_sec : start + fallbackDur;
      if (!(end > start)) end = start + fallbackDur;
      const barInfo = { index: i, number: bar.index ?? i + 1, start, end, data: bar };
      this.bars.push(barInfo);
      cursor = end;

      for (const note of bar.vocal || []) {
        const beatSec = (end - start) / beatsPerBar;
        const t = start + (Number(note.beat) || 0) * beatSec;
        const dur = Math.max(0.08, (Number(note.dur) || 1) * beatSec);
        this.events.push({
          t,
          type: 'note',
          midi: (Number(note.midi) || 60) + this.pitchOffset,
          dur,
          id: note.id || null,
          bar: i,
          beat: Number(note.beat) || 0,
        });
      }

      for (const chord of bar.chords || []) {
        const beatSec = (end - start) / beatsPerBar;
        const t = start + (Number(chord.beat) || 0) * beatSec;
        this.events.push({ t, type: 'strum', symbol: chord.symbol, bar: i, beat: Number(chord.beat) || 0 });
      }

      for (let b = 0; b < Math.ceil(beatsPerBar); b++) {
        const t = start + (b * (end - start)) / beatsPerBar;
        this.events.push({ t, type: 'click', strong: b === 0, bar: i, beat: b });
      }
    }
    this.events.sort((a, b) => a.t - b.t || a.type.localeCompare(b.type));
    this.duration = Math.max(0, this.bars.at(-1)?.end || 0);
  }

  setVoicings(lib) {
    this.voiceLib = lib;
  }

  get isPlaying() {
    return this.state === 'playing';
  }

  get position() {
    return this._scoreTime();
  }

  _scoreTime() {
    if (this.state !== 'playing') return this.anchorScore;
    if (this.mode === 'audio' && this.audioEl) return this.audioEl.currentTime;
    if (!this.ctx) return this.anchorScore;
    return this.anchorScore + (this.ctx.currentTime - this.anchorCtx) * this.rate;
  }

  /* ───────────── 生命周期 ───────────── */

  _ensureCtx() {
    if (this.ctx) return this.ctx;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) throw new Error('当前浏览器不支持 Web Audio');
    this.ctx = new Ctx();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.85;
    this.master.connect(this.ctx.destination);
    return this.ctx;
  }

  _hasAudio() {
    const el = this.audioEl;
    return !!(el && el.getAttribute('src') && el.readyState >= 1 && el.error == null);
  }

  play() {
    if (!this.bars.length) return false;
    if (this.state === 'playing') return true;
    this._ensureCtx();
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    this.mode = this._hasAudio() ? 'audio' : 'synth';

    if (this.state === 'paused') {
      this.state = 'playing';
      this._startClock();
      if (this.mode === 'audio') this.audioEl.play().catch(() => { this._fallbackSynth(); });
      this.onStateChange?.('playing');
      return true;
    }

    // 从头或从当前 anchor 播放
    this.anchorScore = clamp(this.anchorScore, 0, Math.max(0, this.duration - 0.01));
    this.state = 'playing';
    this.cursor = this._lowerBound(this.anchorScore);
    if (this.mode === 'audio') {
      try {
        this.audioEl.currentTime = this.anchorScore;
        this.audioEl.playbackRate = this.rate;
        this.audioEl.play().catch(() => this._fallbackSynth());
      } catch {
        this._fallbackSynth();
      }
    } else {
      this.anchorCtx = this.ctx.currentTime;
    }
    this._startClock();
    this.onStateChange?.('playing');
    return true;
  }

  _fallbackSynth() {
    this.mode = 'synth';
    this.anchorScore = this.audioEl?.currentTime || this.anchorScore;
    this.anchorCtx = this.ctx.currentTime;
    this.cursor = this._lowerBound(this.anchorScore);
  }

  pause() {
    if (this.state !== 'playing') return;
    const at = this._scoreTime();
    this._silence();
    this.state = 'paused';
    this.anchorScore = at;
    this._stopClock();
    if (this.audioEl && !this.audioEl.paused) this.audioEl.pause();
    this.onStateChange?.('paused');
  }

  stop({ keepBar = false } = {}) {
    const wasPlaying = this.state === 'playing';
    this._silence();
    this._stopClock();
    if (this.audioEl && !this.audioEl.paused) this.audioEl.pause();
    this.state = 'stopped';
    this.anchorScore = 0;
    this.cursor = 0;
    this.currentBar = -1;
    this.currentNote = null;
    if (!keepBar) {
      this.renderer?.setActiveBar?.(0);
      this.renderer?.clearPlayhead?.();
    }
    this.onStateChange?.(wasPlaying ? 'stopped' : this.state);
  }

  toggle() {
    if (this.state === 'playing') this.pause();
    else this.play();
  }

  setRate(rate) {
    const next = clamp(Number(rate) || 1, 0.5, 1.2);
    if (Math.abs(next - this.rate) < 1e-3) return;
    const at = this.state === 'playing' ? this._scoreTime() : this.anchorScore;
    this.rate = next;
    this.anchorScore = at;
    if (this.ctx) this.anchorCtx = this.ctx.currentTime;
    this.cursor = this._lowerBound(at);
    this._silence();
    if (this.audioEl) this.audioEl.playbackRate = next;
  }

  setMetronome(on) {
    this.metronome = !!on;
  }

  setLoopSection(on) {
    this.loopSection = !!on;
  }

  setAudio(el) {
    this.audioEl = el;
  }

  sectionAt(barIndex) {
    for (const section of this.sections) {
      const s = (section.start_bar ?? 1) - 1;
      const e = (section.end_bar ?? s + 1) - 1;
      if (barIndex >= s && barIndex <= e) return section;
    }
    return null;
  }

  seekBar(barNumber) {
    const idx = this.bars.findIndex((b) => b.number === barNumber || b.index === barNumber - 1);
    const bar = this.bars[idx >= 0 ? idx : 0];
    if (!bar) return;
    this.seek(bar.start);
  }

  seek(time) {
    const target = clamp(Number(time) || 0, 0, Math.max(0, (this.duration || 0) - 0.02));
    this.anchorScore = target;
    this.cursor = this._lowerBound(target);
    this._silence();
    if (this.ctx) this.anchorCtx = this.ctx.currentTime;
    if (this.mode === 'audio' && this.audioEl) {
      try {
        this.audioEl.currentTime = target;
      } catch {
        /* 元数据未就绪时忽略 */
      }
    }
    this._applyPosition(target, true);
    this.currentNote = null;
    this.renderer?.setPlayingNote?.(null);
  }

  _lowerBound(t) {
    const events = this.events;
    let lo = 0;
    let hi = events.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (events[mid].t < t - 1e-4) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  _startClock() {
    this._stopClock();
    if (this.mode === 'audio') {
      const loop = () => {
        if (this.state !== 'playing') return;
        this._tickAudio();
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
      this.timer = setInterval(() => this._tickUi(), 90);
    } else {
      this.anchorCtx = this.ctx.currentTime;
      this.timer = setInterval(() => this._tickSynth(), TICK_MS);
      this._tickSynth();
    }
  }

  _stopClock() {
    if (this.timer) clearInterval(this.timer);
    if (this.raf) cancelAnimationFrame(this.raf);
    this.timer = null;
    this.raf = null;
  }

  _silence() {
    for (const node of this.sources) {
      try {
        node.stop();
      } catch {
        /* 已停止 */
      }
    }
    this.sources = [];
    this.renderer?.setPlayingNote?.(null);
    this.currentNote = null;
  }

  /* ───────────── 调度 ───────────── */

  _tickSynth() {
    if (this.state !== 'playing') return;
    const now = this._scoreTime();
    const horizon = now + LOOKAHEAD;
    while (this.cursor < this.events.length) {
      const ev = this.events[this.cursor];
      if (ev.t > horizon) break;
      if (ev.t >= now - 0.05) this._schedule(ev, now);
      this.cursor++;
    }
    this._tickUi(now);
  }

  _tickAudio() {
    if (this.state !== 'playing') return;
    const now = this.audioEl.currentTime;
    const horizon = now + LOOKAHEAD * this.rate;
    while (this.cursor < this.events.length) {
      const ev = this.events[this.cursor];
      if (ev.t > horizon) break;
      if (ev.t >= now - 0.05 && ev.type === 'click' && this.metronome) this._scheduleClick(ev);
      this.cursor++;
    }
    this._tickUi(now);
  }

  _ctxTimeFor(scoreT, now) {
    const delta = (scoreT - now) / this.rate;
    return this.ctx.currentTime + delta;
  }

  _schedule(ev, now) {
    if (!this.ctx) return;
    const when = Math.max(this.ctx.currentTime, this._ctxTimeFor(ev.t, now));
    if (ev.type === 'note') {
      this._pluck(ev.midi, when, Math.max(0.18, ev.dur), { gain: 0.2, type: 'triangle' });
    } else if (ev.type === 'strum') {
      this._strumForSymbol(ev.symbol, when);
    } else if (ev.type === 'click' && this.metronome) {
      this._click(when, ev.strong);
    }
  }

  _track(node) {
    this.sources.push(node);
    if (this.sources.length > 200) this.sources.splice(0, this.sources.length - 200);
    node.onended = () => {
      const i = this.sources.indexOf(node);
      if (i >= 0) this.sources.splice(i, 1);
    };
    return node;
  }

  _pluck(midi, when, dur, { gain = 0.18, type = 'triangle', detune = 0 } = {}) {
    const ctx = this.ctx;
    const freq = midiToFreq(midi);
    if (!Number.isFinite(freq) || freq <= 0) return;
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    osc.detune.value = detune;
    const attack = 0.008;
    const hold = Math.min(Math.max(dur * 0.55, 0.09), 1.6);
    amp.gain.setValueAtTime(0.0001, when);
    amp.gain.exponentialRampToValueAtTime(gain, when + attack);
    amp.gain.exponentialRampToValueAtTime(Math.max(0.0002, gain * 0.32), when + attack + hold);
    amp.gain.exponentialRampToValueAtTime(0.0001, when + attack + hold + 0.25);
    osc.connect(amp).connect(this.master);
    osc.start(when);
    osc.stop(when + attack + hold + 0.3);
    this._track(osc);
    // 泛音一点，让拨弦更真实
    if (type === 'triangle') {
      const osc2 = ctx.createOscillator();
      const amp2 = ctx.createGain();
      osc2.type = 'sine';
      osc2.frequency.value = freq * 2;
      amp2.gain.setValueAtTime(0.0001, when);
      amp2.gain.exponentialRampToValueAtTime(gain * 0.22, when + attack);
      amp2.gain.exponentialRampToValueAtTime(0.0001, when + attack + hold * 0.6);
      osc2.connect(amp2).connect(this.master);
      osc2.start(when);
      osc2.stop(when + attack + hold + 0.1);
      this._track(osc2);
    }
  }

  _strumForSymbol(symbol, when) {
    const tuning = this.score?.meta?.guitar?.tuning || DEFAULT_TUNING;
    const voicing = this.voiceLib?.select?.(symbol) || this.voiceLib?.getById?.(symbol) || null;
    if (!voicing) return;
    const frets = voicing.frets || [];
    let order = 0;
    for (let i = 0; i < frets.length; i++) {
      const fret = frets[i];
      if (fret == null || fret < 0) continue;
      const midi = tuning[i] + fret;
      const delay = order * 0.022;
      order++;
      this._pluck(midi, when + delay, 0.85, { gain: 0.085, type: 'triangle' });
    }
  }

  _strumNotes(midis, when) {
    midis.forEach((midi, i) => this._pluck(midi, when + i * 0.022, 0.8, { gain: 0.09 }));
  }

  _click(when, strong = false) {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    const amp = ctx.createGain();
    osc.type = 'square';
    osc.frequency.value = strong ? 1568 : 1046;
    amp.gain.setValueAtTime(0.0001, when);
    amp.gain.exponentialRampToValueAtTime(strong ? 0.1 : 0.055, when + 0.002);
    amp.gain.exponentialRampToValueAtTime(0.0001, when + 0.045);
    osc.connect(amp).connect(this.master);
    osc.start(when);
    osc.stop(when + 0.06);
    this._track(osc);
  }

  /* ───────────── UI 同步 ───────────── */

  /** 把谱面位置同步到 UI（播放中与 seek 后共用） */
  _applyPosition(now, force = false) {
    if (!this.bars.length) return false;
    const barIndex = indexAt(this.bars, now);
    const bar = this.bars[barIndex];
    if (!bar) return false;
    const beatsPerBar =
      (this.score?.meta?.time_signature?.[0] || 4) * (4 / (this.score?.meta?.time_signature?.[1] || 4));
    const beat = clamp(
      ((now - bar.start) / Math.max(0.0001, bar.end - bar.start)) * beatsPerBar,
      0,
      beatsPerBar
    );

    if (barIndex !== this.currentBar || force) {
      this.currentBar = barIndex;
      this.renderer?.setActiveBar?.(barIndex);
      this.onBarChange?.(barIndex, bar, this.bars);
      if (this.state === 'playing') this.renderer?.scrollBarIntoView?.(barIndex);
    }
    this.renderer?.setPlayhead?.(barIndex, beat);

    const nowMs = performance.now();
    if (force || nowMs - this._lastUiAt > 80) {
      this._lastUiAt = nowMs;
      this.onProgress?.({
        fraction: this.duration ? now / this.duration : 0,
        elapsed: now,
        duration: this.duration,
        barIndex: bar.number,
        barCount: this.bars.length,
        bar,
      });
    }
    return true;
  }

  _tickUi(nowGiven = null) {
    if (this.state !== 'playing' || !this.bars.length) return;
    const now = nowGiven ?? this._scoreTime();

    // 循环当前段落
    if (this.loopSection) {
      const barIndex = indexAt(this.bars, now);
      const bar = this.bars[barIndex];
      const section = this.sectionAt(barIndex);
      if (bar && section && now >= (section.end_sec ?? bar.end) - 0.02) {
        this.seek(section.start_sec ?? this.bars[(section.start_bar ?? 1) - 1]?.start ?? 0);
        return;
      }
    }

    if (!this._applyPosition(now)) return;

    // 当前音符高亮
    const active = this.events.find((ev) => ev.type === 'note' && ev.t <= now && now < ev.t + ev.dur);
    const id = active?.id ?? null;
    if (id !== this.currentNote) {
      this.currentNote = id;
      this.renderer?.setPlayingNote?.(id);
      this.onNoteChange?.(id, active || null);
    }
  }
}
