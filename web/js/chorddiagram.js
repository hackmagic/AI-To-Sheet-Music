/**
 * chorddiagram.js —— 纯 SVG 和弦指法图（6 弦 5 品，空弦圈 / 闷弦叉 / 横按条 / 把位标号）
 *
 * voicing 数据结构（与 schemas/score.schema.json#/$defs/voicing 一致）：
 *   { symbol, frets: [低..高 6 弦], fingers, base_fret, barres:[{fret,from,to}], difficulty, label }
 *   frets: -1 不弹 / null 闷弦 / 0 空弦 / n 品
 *
 * 另外提供“自动生成指法”能力：score 里没有对应把位时（例如移调后），
 * 用弦品搜索算法生成若干可弹把位，保证任何调都能出图。
 */

import { chordPcs, chordBassPc, nameToPc, parseChord } from './music.js';

export const STANDARD_TUNING = [40, 45, 50, 55, 59, 64]; // E2 A2 D3 G3 B3 E4

const SVG_NS = 'http://www.w3.org/2000/svg';

const BOARD = {
  width: 54,
  height: 80,
  left: 10,
  right: 46,
  top: 18,
  fretGap: 10,
  frets: 5,
};

/** 显示用：把 fret 值换算成弦序号索引 */
function stringX(i) {
  return BOARD.left + i * ((BOARD.right - BOARD.left) / 5);
}

function fretY(relativeFret) {
  return BOARD.top + (relativeFret - 0.5) * BOARD.fretGap;
}

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

/**
 * 计算一个 voicing 的显示参数（相对品、是否需要把位标号、横按越界修正）
 */
export function voicingLayout(voicing) {
  const frets = (voicing.frets || []).slice(0, 6);
  while (frets.length < 6) frets.push(-1);
  const played = frets.filter((f) => typeof f === 'number' && f > 0);
  const maxFret = played.length ? Math.max(...played) : 0;
  const minFret = played.length ? Math.min(...played) : 0;
  let base = Number.isFinite(voicing.base_fret) ? voicing.base_fret : 0;
  if (base === 0 && maxFret > BOARD.frets) base = Math.max(1, minFret - (minFret > 2 ? 1 : 0));
  if (base === 0 && maxFret <= BOARD.frets) {
    return { frets, baseFret: 0, relative: (f) => f };
  }
  const baseFret = base || 1;
  return { frets, baseFret, relative: (f) => f - baseFret + 1 };
}

/**
 * 渲染一个和弦指法图 SVG。
 * @param {object} voicing
 * @param {object} [opts] { size, showFingers, title }
 * @returns {SVGElement}
 */
export function renderDiagram(voicing, opts = {}) {
  const { frets, baseFret, relative } = voicingLayout(voicing);
  const scale = opts.size ? opts.size / 56 : 1;
  const svg = el('svg', {
    viewBox: `0 0 ${BOARD.width} ${BOARD.height + 6}`,
    width: Math.round(BOARD.width * scale),
    height: Math.round((BOARD.height + 6) * scale),
    class: 'chord-diagram',
    role: 'img',
    'aria-label': `${voicing.symbol || opts.title || '和弦'} 指法图`,
    focusable: 'false',
  });

  const title = el('title', {}, svg);
  title.textContent = `${voicing.symbol || ''} 指法：${frets
    .map((f) => (f === -1 || f === null ? 'x' : f === 0 ? '0' : f))
    .join(' ')}`;

  el('rect', { x: 0, y: 0, width: BOARD.width, height: BOARD.height + 6, fill: 'transparent' }, svg);

  const boardBottom = BOARD.top + BOARD.fretGap * BOARD.frets;

  // 品格线
  for (let k = 0; k <= BOARD.frets; k++) {
    const y = BOARD.top + k * BOARD.fretGap;
    const isNut = k === 0 && baseFret === 0;
    el(
      'line',
      {
        x1: BOARD.left,
        y1: y,
        x2: BOARD.right,
        y2: y,
        stroke: isNut ? 'currentColor' : 'currentColor',
        'stroke-width': isNut ? 2.6 : 1.2,
        opacity: isNut ? 1 : 0.6,
      },
      svg
    );
  }

  // 弦线
  for (let i = 0; i < 6; i++) {
    el(
      'line',
      {
        x1: stringX(i),
        y1: BOARD.top,
        x2: stringX(i),
        y2: boardBottom,
        stroke: 'currentColor',
        'stroke-width': i === 0 ? 1.35 : 0.95,
        opacity: 0.8,
      },
      svg
    );
  }

  // 把位标号（如 3fr）。第一把位是默认位置，实体谱书不标，避免“1fr”噪音
  if (baseFret > 1) {
    const t = el(
      'text',
      {
        x: BOARD.left - 3,
        y: boardBottom - 1,
        'text-anchor': 'end',
        'font-size': 8,
        'font-family': 'var(--font-ui, sans-serif)',
        fill: 'currentColor',
        opacity: 0.75,
      },
      svg
    );
    t.textContent = `${baseFret}fr`;
  }

  // 空弦 / 闷弦
  for (let i = 0; i < 6; i++) {
    const f = frets[i];
    const x = stringX(i);
    if (f === 0) {
      el(
        'circle',
        { cx: x, cy: BOARD.top - 6, r: 2.2, fill: 'none', stroke: 'currentColor', 'stroke-width': 1 },
        svg
      );
    } else if (f === -1 || f === null) {
      const s = 2;
      el('line', { x1: x - s, y1: BOARD.top - 8, x2: x + s, y2: BOARD.top - 4, stroke: 'currentColor', 'stroke-width': 0.9, opacity: 0.7 }, svg);
      el('line', { x1: x + s, y1: BOARD.top - 8, x2: x - s, y2: BOARD.top - 4, stroke: 'currentColor', 'stroke-width': 0.9, opacity: 0.7 }, svg);
    }
  }

  // 横按
  const barres = Array.isArray(voicing.barres) ? voicing.barres : [];
  const barreDrawn = new Set();
  for (const bar of barres) {
    if (!bar || typeof bar.fret !== 'number') continue;
    const rel = relative(bar.fret);
    if (rel < 1 || rel > BOARD.frets + 0.001) continue;
    const from = Math.max(0, Math.min(5, bar.from ?? 0));
    const to = Math.max(0, Math.min(5, bar.to ?? 5));
    const x1 = stringX(Math.min(from, to));
    const x2 = stringX(Math.max(from, to));
    el(
      'rect',
      {
        x: x1 - 2.4,
        y: fretY(rel) - 2.8,
        width: Math.max(0, x2 - x1) + 4.8,
        height: 5.6,
        rx: 2.8,
        fill: 'currentColor',
      },
      svg
    );
    barreDrawn.add(`${rel}`);
  }

  // 指位圆点
  const showFingers = opts.showFingers !== false && Array.isArray(voicing.fingers);
  for (let i = 0; i < 6; i++) {
    const f = frets[i];
    if (typeof f !== 'number' || f <= 0) continue;
    const rel = relative(f);
    if (rel < 1 || rel > BOARD.frets + 0.001) continue;
    const x = stringX(i);
    const y = fretY(rel);
    const finger = showFingers ? voicing.fingers[i] : null;
    if (finger && barreDrawn.has(String(rel))) {
      // 横按上的其他按弦位置：画白点 + 数字
      el('circle', { cx: x, cy: y, r: 3, fill: 'var(--paper, #fff)', stroke: 'currentColor', 'stroke-width': 0.8 }, svg);
      const t = el('text', { x, y: y + 2.4, 'text-anchor': 'middle', 'font-size': 5.4, fill: 'currentColor', 'font-family': 'var(--font-ui, sans-serif)' }, svg);
      t.textContent = String(finger);
      continue;
    }
    el('circle', { cx: x, cy: y, r: 3.1, fill: 'currentColor' }, svg);
    if (finger) {
      const t = el(
        'text',
        {
          x,
          y: y + 2.3,
          'text-anchor': 'middle',
          'font-size': 5.2,
          fill: 'var(--paper, #fff)',
          'font-family': 'var(--font-ui, sans-serif)',
        },
        svg
      );
      t.textContent = String(finger);
    }
  }

  return svg;
}

/* ───────────────────────── 指法库 / 自动生成 ───────────────────────── */

function pcAt(tuning, stringIndex, fret) {
  return (tuning[stringIndex] + fret) % 12;
}

/**
 * 从弦品搜索生成可弹把位。
 * @param {string} symbol 和弦符号
 * @param {number[]} tuning 6 弦空弦 midi（低->高）
 * @param {number} maxCandidates
 * @returns {object[]} voicing 数组（已按可弹性排序）
 */
export function generateVoicings(symbol, tuning = STANDARD_TUNING, maxCandidates = 3) {
  const { root } = parseChord(symbol);
  const pcs = chordPcs(symbol);
  const bassPc = chordBassPc(symbol);
  if (!pcs.length) return [];

  const results = [];
  const seen = new Set();
  const SPAN = 4;

  for (let base = 0; base <= 10; base++) {
    // 每根弦的候选品味
    const options = [];
    for (let s = 0; s < 6; s++) {
      const list = [];
      const openPc = pcAt(tuning, s, 0);
      for (let fret = base; fret <= base + SPAN; fret++) {
        if (fret > 12) continue;
        if (pcs.includes(pcAt(tuning, s, fret))) list.push(fret);
      }
      if (base <= 2 && pcs.includes(openPc) && !list.includes(0)) list.push(0);
      options.push(list);
    }

    const frets = new Array(6).fill(-1);
    const walk = (s, usedMask, sounding) => {
      if (results.length > 400) return;
      if (s === 6) {
        if (sounding < 4) return;
        // 所有和弦音都要出现
        const fullMask = (1 << pcs.length) - 1;
        let mask = 0;
        for (let i = 0; i < 6; i++) {
          const f = frets[i];
          if (f < 0) continue;
          const pc = pcAt(tuning, i, f);
          for (let k = 0; k < pcs.length; k++) if (pcs[k] === pc) mask |= 1 << k;
        }
        if (mask !== fullMask) return;
        const played = frets.filter((f) => f > 0);
        const span = played.length ? Math.max(...played) - Math.min(...played) : 0;
        if (span > SPAN) return;
        // 最低发声音必须是目标低音
        let lowest = -1;
        for (let i = 0; i < 6; i++) {
          if (frets[i] >= 0) {
            lowest = i;
            break;
          }
        }
        const lowestPc = pcAt(tuning, lowest, frets[lowest]);
        const openCount = frets.filter((f) => f === 0).length;
        const mutedCount = frets.filter((f) => f === -1).length;
        const interiorMute = frets.some((f, i) => f === -1 && i > lowest && i < 5);

        let score = 0;
        score -= openCount * 1.8;
        score += span * 0.9;
        score += Math.min(...played.map((f) => f)) * 0.55;
        score += mutedCount * 1.5 + (interiorMute ? 4 : 0);
        score += (6 - sounding) * 1.2;
        score += lowestPc === bassPc ? -6 : 0;
        score += pcs.length > 3 ? 0.4 : 0;

        const key = frets.join(',');
        if (!seen.has(key)) {
          seen.add(key);
          const baseFret = played.length ? Math.max(1, Math.min(...played)) : 0;
          results.push({
            symbol,
            frets: frets.slice(),
            fingers: null,
            base_fret: Math.min(...played.map((f) => f)) <= 1 ? 0 : Math.min(...played),
            barres: detectBarres(frets, Math.min(...played)),
            difficulty: score > 6 ? 4 : score > 3 ? 3 : score > 1 ? 2 : 1,
            label: `${symbol}（自动把位）`,
            _score: score,
          });
        }
        return;
      }
      // 闷弦
      frets[s] = -1;
      walk(s + 1, usedMask, sounding);
      for (const fret of options[s]) {
        frets[s] = fret;
        walk(s + 1, usedMask, sounding + 1);
      }
      frets[s] = -1;
    };
    walk(0, 0, 0);
    if (results.length > 400) break;
  }

  results.sort((a, b) => a._score - b._score);
  const picked = [];
  for (const r of results) {
    if (picked.some((p) => p.frets.join(',') === r.frets.join(','))) continue;
    picked.push(r);
    if (picked.length >= maxCandidates) break;
  }
  return picked.map(({ _score, ...rest }) => rest);
}

function detectBarres(frets, lowestFret) {
  const barres = [];
  if (!frets || !frets.length) return barres;
  const byFret = new Map();
  frets.forEach((f, i) => {
    if (f > 0) {
      if (!byFret.has(f)) byFret.set(f, []);
      byFret.get(f).push(i);
    }
  });
  for (const [fret, strings] of byFret) {
    if (strings.length >= 2 && fret === lowestFret) {
      barres.push({ fret, from: Math.min(...strings), to: Math.max(...strings) });
      break;
    }
  }
  return barres;
}

/**
 * 指法库：接受 score.voicings（id -> voicing）或 web/voicings.json
 * （{voicings:{id->voicing}} / 或直接 id->voicing）。
 */
export class VoicingLibrary {
  constructor(json = {}) {
    this.byId = new Map();
    this.bySymbol = new Map();
    this.cache = new Map();
    this.addAll(json);
  }

  addAll(json) {
    if (!json || typeof json !== 'object') return this;
    const map = json.voicings && typeof json.voicings === 'object' ? json.voicings : json;
    for (const [id, v] of Object.entries(map)) {
      if (!v || typeof v !== 'object' || !v.symbol || !Array.isArray(v.frets)) continue;
      const voicing = { ...v, id };
      this.byId.set(id, voicing);
      const list = this.bySymbol.get(v.symbol) || [];
      list.push(voicing);
      this.bySymbol.set(v.symbol, list);
    }
    this.cache.clear();
    return this;
  }

  getById(id) {
    return this.byId.get(id) || null;
  }

  /**
   * 为一个和弦选把位：优先库内（难度低者），否则自动生成。
   * @param {string} symbol
   * @param {object} [opts] { tuning, difficulty, preferId }
   */
  select(symbol, opts = {}) {
    if (!symbol) return null;
    const preferId = opts.preferId;
    if (preferId && this.byId.has(preferId)) {
      const v = this.byId.get(preferId);
      if (v.symbol === symbol) return v;
    }
    const cached = this.cache.get(symbol);
    if (cached) return cached;

    const list = (this.bySymbol.get(symbol) || []).slice();
    let picked = null;
    if (list.length) {
      list.sort((a, b) => (a.difficulty ?? 3) - (b.difficulty ?? 3));
      picked = list[0];
    } else {
      const generated = generateVoicings(symbol, opts.tuning || STANDARD_TUNING, 3);
      picked = generated[0] || null;
    }
    this.cache.set(symbol, picked);
    return picked;
  }
}

/** 六弦空弦音高与音名显示（供调试） */
export function tuningLabel(tuning = STANDARD_TUNING) {
  return tuning.map((m) => `${['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][m % 12]}${Math.floor(m / 12) - 1}`).join(' ');
}

export { nameToPc };
