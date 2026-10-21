/**
 * editor.js —— 谱面编辑器：和弦 / 歌词 / 小节级编辑 + 撤销重做 + 编辑记录
 *
 * 设计要点
 *  - 所有编辑都写到 **raw score**（state.raw），绝不写视图层（transposeScore 的产物）。
 *    视图是派生数据，移调/变调夹随时会重算，改视图等于改影子。
 *  - 用户在谱面上看到的是「记谱视图」的和弦。当 displayShift ≠ 0 时，
 *    输入值必须反算回 raw：raw = view - displayShift（见 toRawSemitone）。
 *  - 撤销：编辑前把 raw 用 structuredClone 深拷入栈；88 小节约 500KB×60 步，
 *    结构化克隆比 JSON 序列化快一个数量级。
 *  - 编辑痕迹两种：①raw 里 `edits[]` 日志（持久化、可审计）；
 *    ②对象上的 `_edited` / `_lyric_cleared` 标记（只给渲染器画虚线，导出时剥掉）。
 *
 * 纯数据函数（applyChordEdit / stripEditMarks / countEdits / toRawSemitone…）不碰 DOM，
 * 可以在 node 里直接单测；
 * DOM 部分集中在文件后半的 ScoreEditor 类，由 createEditor() 在浏览器里实例化。
 */

import {
  transposeChord,
  nameToPc,
  pcName,
  fifthsFor,
  barBeats,
} from './music.js';

/* ─────────────────────────── 常量 ─────────────────────────── */

/** 对象上的「已编辑」标记（只用于渲染，导出时剥掉） */
export const EDIT_MARK = '_edited';
/** 歌词被清成延音/无词的标记 */
export const LYRIC_CLEAR_MARK = '_lyric_cleared';
/** 删除/插入小节留下的内部标记 */
export const STRUCT_MARK = '_struct';

const MARK_KEYS = new Set([EDIT_MARK, LYRIC_CLEAR_MARK, STRUCT_MARK]);

/** 撤销栈深度（需求 ≥50） */
export const DEFAULT_HISTORY = 60;

const TIME_SIGNATURES = [
  [4, 4],
  [3, 4],
  [6, 8],
];

const DEGREE_MAJOR = [
  { degree: 'I', step: 0, quality: '' },
  { degree: 'ii', step: 2, quality: 'm' },
  { degree: 'iii', step: 4, quality: 'm' },
  { degree: 'IV', step: 5, quality: '' },
  { degree: 'V', step: 7, quality: '' },
  { degree: 'V7', step: 7, quality: '7' },
  { degree: 'vi', step: 9, quality: 'm' },
  { degree: 'vii°', step: 11, quality: 'dim' },
];

const DEGREE_MINOR = [
  { degree: 'i', step: 0, quality: 'm' },
  { degree: 'ii°', step: 2, quality: 'dim' },
  { degree: 'III', step: 3, quality: '' },
  { degree: 'iv', step: 5, quality: 'm' },
  { degree: 'v', step: 7, quality: 'm' },
  { degree: 'V', step: 7, quality: '' },
  { degree: 'VI', step: 8, quality: '' },
  { degree: 'VII', step: 10, quality: '' },
];

/** 快捷按钮用的级数顺序（大调经典 I-IV-V-vi） */
const QUICK_MAJOR = ['I', 'IV', 'V', 'V7', 'vi', 'ii'];
const QUICK_MINOR = ['i', 'iv', 'v', 'V', 'VI', 'VII'];

/* ─────────────────────────── 基础工具 ─────────────────────────── */

export function clone(value) {
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

export function nowIso() {
  return new Date().toISOString();
}

function roundBeat(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

/* ─────────────────────────── 移调：视图 ↔ raw ─────────────────────────── */

/**
 * 把「视图层 → raw 层」所需的半音偏移取反。
 * 视图位移 displayShift = transpose - (capo - baseCapo)，transposeScore 用它把 raw 变成视图；
 * 反过来写回 raw 时，要减掉同一个位移。
 *
 * @param {number} offset 当前视图位移（半音，main.js 的 displayShift()）
 * @returns {number} 应用到和弦符号上的半音数（-offset）
 */
export function toRawSemitone(offset) {
  const shift = Number(offset);
  if (!Number.isFinite(shift)) return 0;
  // 加 0 把 -0 归一成 0：-0 会让 assert.equal / JSON 输出出现难查的差异
  return -Math.round(shift) + 0;
}

/** 反向：把 raw 层值换算成「视图层 → raw」的偏移（= offset）。成对提供，便于自检。 */
export function toViewSemitone(offset) {
  const shift = Number(offset);
  if (!Number.isFinite(shift)) return 0;
  return Math.round(shift);
}

/** raw 记号层 → 视图层和弦符号 */
export function rawChordToView(symbol, offset = 0, { preferFlats = false } = {}) {
  if (!symbol) return symbol;
  const shift = toViewSemitone(offset);
  if (!shift) return symbol;
  return transposeChord(symbol, shift, preferFlats);
}

/** 视图层和弦符号 → raw 记号层（用户「看到什么就输什么」） */
export function viewChordToRaw(symbol, offset = 0, { preferFlats = false } = {}) {
  if (!symbol) return symbol;
  const shift = toRawSemitone(offset);
  if (!shift) return symbol;
  return transposeChord(symbol, shift, preferFlats);
}

/** 某个 meta.key 的记谱是否偏好降号（决定反算时的拼写） */
export function preferFlatsForKey(keyMeta) {
  if (!keyMeta) return false;
  if (typeof keyMeta.num_sharps_flats === 'number') return keyMeta.num_sharps_flats < 0;
  const tonic = keyMeta.play || keyMeta.sounding || 'C';
  return fifthsFor(String(tonic).replace(/m$/, ''), keyMeta.mode === 'minor' ? 'minor' : 'major') < 0;
}

/* ─────────────────────────── 和弦符号：规范化和查表 ─────────────────────────── */

/** 用户输入规范化：#/♭ 全角转半角、根音大写、去掉多余空白 */
export function normalizeChordInput(text) {
  let s = String(text ?? '').trim();
  if (!s) return '';
  s = s
    .replace(/[♯＃]/g, '#')
    .replace(/[♭ｂ]/g, 'b')
    .replace(/[（]/g, '(')
    .replace(/[）]/g, ')')
    .replace(/\s+/g, '');
  // 根音与斜杠低音都转大写
  s = s.replace(/^([a-g])/, (m) => m.toUpperCase());
  s = s.replace(/\/([a-g])/g, (m, ch) => `/${ch.toUpperCase()}`);
  return s;
}

/** 语法上像一个和弦符号（根音 A-G + 可选升降 + 任意后缀 + 可选 /低音） */
export function isChordLike(symbol) {
  const s = String(symbol || '').trim();
  if (!s) return false;
  if (!/^[A-G][#b]?[^/]*(\/[A-G][#b]?)?$/.test(s)) return false;
  // 后缀只允许常见的和弦记号字符
  return /^[A-G][#b]?[a-zA-Z0-9#b()+\-°øΔ]*(\/[A-G][#b]?)?$/.test(s);
}

/**
 * 用 web/voicings.json 查表。
 * @returns {{status:'empty'|'invalid'|'lib'|'derived', canonical?:string, note?:string}}
 */
export function chordLibraryStatus(symbol, voicingJson) {
  const raw = String(symbol || '').trim();
  if (!raw) return { status: 'empty' };
  if (!isChordLike(raw)) return { status: 'invalid' };
  const standard = voicingJson?.standard;
  const aliases = voicingJson?.aliases || {};
  if (standard && typeof standard === 'object') {
    if (Array.isArray(standard[raw]) && standard[raw].length) return { status: 'lib', canonical: raw };
    const canonical = aliases[raw];
    if (canonical && Array.isArray(standard[canonical]) && standard[canonical].length) {
      return { status: 'lib', canonical, note: canonical !== raw ? `库中按 ${canonical} 收录` : '' };
    }
  }
  const map = voicingJson?.voicings;
  if (!map || typeof map !== 'object') return { status: 'derived', note: '指法库未加载' };
  for (const voicing of Object.values(map)) {
    if (voicing && voicing.symbol === raw) return { status: 'lib', canonical: raw };
  }
  return { status: 'derived' };
}

/* ─────────────────────────── 当前调内的和弦 ─────────────────────────── */

/** 当前调内的顺阶和弦（快捷按钮用）；tonic 是**显示调**的主音 */
export function diatonicChords(tonic, mode = 'major', { preferFlats = false } = {}) {
  const table = mode === 'minor' ? DEGREE_MINOR : DEGREE_MAJOR;
  const order = mode === 'minor' ? QUICK_MINOR : QUICK_MAJOR;
  const tonicPc = nameToPc(tonic || 'C');
  const byDegree = new Map();
  for (const item of table) {
    const rootPc = (((tonicPc + item.step) % 12) + 12) % 12;
    byDegree.set(item.degree, {
      degree: item.degree,
      symbol: `${pcName(rootPc, preferFlats)}${item.quality}`,
    });
  }
  const picked = [];
  for (const degree of order) {
    const item = byDegree.get(degree);
    if (item) picked.push(item);
  }
  // 补齐到 6 个，避免小调表缺项时按钮太少
  for (const [degree, item] of byDegree) {
    if (picked.length >= 6) break;
    if (!picked.some((x) => x.degree === degree)) picked.push(item);
  }
  return picked;
}

/* ─────────────────────────── 定位 ─────────────────────────── */

export function findRawBar(score, barIndex) {
  const bars = asArray(score?.bars);
  const target = Number(barIndex);
  return bars.find((bar) => Number(bar?.index) === target) || null;
}

/** 在一个小节里定位和弦（优先数组下标，退化为 beat 匹配） */
export function resolveChordTarget(bar, { chordIdx = null, beat = null } = {}) {
  const chords = asArray(bar?.chords);
  if (Number.isInteger(chordIdx) && chords[chordIdx]) return { chord: chords[chordIdx], idx: chordIdx };
  if (beat != null) {
    const idx = chords.findIndex((c) => Math.abs((Number(c?.beat) || 0) - Number(beat)) < 1e-6);
    if (idx >= 0) return { chord: chords[idx], idx };
  }
  return null;
}

function previousSoundingNote(bar, noteIdx) {
  const vocal = asArray(bar?.vocal);
  for (let i = noteIdx - 1; i >= 0; i--) {
    const note = vocal[i];
    if (note && note.midi != null && !note.rest) return note;
  }
  return null;
}

/* ─────────────────────────── 编辑日志 ─────────────────────────── */

export function recordEdit(score, entry) {
  if (!score || typeof score !== 'object') return null;
  if (!Array.isArray(score.edits)) score.edits = [];
  const full = { at: nowIso(), ...entry };
  score.edits.push(full);
  return full;
}

export function countEdits(score) {
  return asArray(score?.edits).length;
}

function fmtTimeSignature(ts) {
  if (!Array.isArray(ts) || ts.length < 2) return '';
  return `${ts[0]}/${ts[1]}`;
}

/** 日志 -> 人能读的一句话 */
export function describeEdit(entry) {
  if (!entry) return '';
  const bar = Number(entry.bar);
  const shift = Number(entry.shift) || 0;
  const from = shift && entry.from_view != null ? entry.from_view : entry.from;
  const to = shift && entry.to_view != null ? entry.to_view : entry.to;
  switch (entry.kind) {
    case 'chord':
      return `第 ${bar} 小节 和弦 ${from || '（空）'} → ${to || '（空）'}`;
    case 'chord-move':
      return `第 ${bar} 小节 和弦 ${to || ''} 移到第 ${(Number(entry.beat) || 0) + 1} 拍`;
    case 'chord-delete':
      return `第 ${bar} 小节 删除和弦 ${from || ''}`;
    case 'lyric':
      return `第 ${bar} 小节 歌词「${from || '—'}」→「${to || '—'}」`;
    case 'bar-insert':
      return `在第 ${bar} 小节后插入 1 小节`;
    case 'bar-delete':
      return `删除第 ${bar} 小节`;
    case 'time-signature':
      return `第 ${bar} 小节拍号 ${entry.from || ''} → ${entry.to || ''}`;
    default:
      return `第 ${bar} 小节 ${entry.kind || '编辑'}`;
  }
}

/** 最近 N 条日志（新的在前） */
export function recentEdits(score, limit = 4) {
  const list = asArray(score?.edits);
  return list.slice(Math.max(0, list.length - limit)).reverse().map(describeEdit);
}

/** 递归剥掉渲染用标记（导出用；不修改入参） */
export function stripEditMarks(score) {
  const out = clone(score);
  const walk = (node) => {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    for (const key of Object.keys(node)) {
      if (MARK_KEYS.has(key)) {
        delete node[key];
        continue;
      }
      walk(node[key]);
    }
  };
  walk(out);
  return out;
}

/**
 * 导出用 score。
 * @param {object} score raw score
 * @param {{clean?:boolean}} [opts] clean=true 去掉 edits 字段与编辑标记（干净版本）
 */
export function buildExport(score, { clean = false } = {}) {
  const out = stripEditMarks(score);
  if (clean) {
    delete out.edits;
  }
  return out;
}

/* ─────────────────────────── 和弦编辑 ─────────────────────────── */

/** 把 hits 里引用旧把位的弦品换成新把位的（保持原弦序/扫弦方向） */
function retargetHits(bar, oldVoicingId, chord, chordBeat, voicing) {
  const hits = asArray(bar?.guitar?.hits);
  const frets = voicing?.frets;
  if (!hits.length || !Array.isArray(frets)) return 0;
  const sorted = asArray(bar.chords)
    .slice()
    .sort((a, b) => (Number(a?.beat) || 0) - (Number(b?.beat) || 0));
  const pos = sorted.indexOf(chord);
  const nextBeat = pos >= 0 && sorted[pos + 1] ? Number(sorted[pos + 1].beat) : Infinity;
  let changed = 0;
  for (const hit of hits) {
    if (!hit || typeof hit !== 'object') continue;
    const byId = !!oldVoicingId && hit.voicing_id === oldVoicingId;
    const beat = Number(hit.beat) || 0;
    const byRange = !oldVoicingId && beat >= chordBeat - 1e-6 && beat < nextBeat;
    if (!byId && !byRange) continue;
    const strings = [];
    for (const pair of asArray(hit.strings)) {
      const str = Array.isArray(pair) ? Number(pair[0]) : NaN;
      if (!Number.isFinite(str) || str < 1 || str > 6) continue;
      const fret = frets[6 - str];
      if (typeof fret === 'number' && fret >= 0) strings.push([str, fret]);
    }
    if (!strings.length) continue;
    hit.strings = strings;
    if (voicing.id) hit.voicing_id = voicing.id;
    else delete hit.voicing_id;
    changed++;
  }
  return changed;
}

/**
 * 改和弦（符号与/或 beat 位置）。
 * 入参的 viewSymbol 是**用户看到的记谱符号**；内部会按 displayShift 反算回 raw。
 *
 * @param {object} score raw score（会被就地修改）
 * @param {{bar:number, chordIdx?:number, beat?:number}} target 定位
 * @param {string} viewSymbol 视图层和弦符号
 * @param {object} [opts] { displayShift, rawPreferFlats, viewPreferFlats, moveToBeat, resolveVoicing }
 */
export function applyChordEdit(score, target, viewSymbol, opts = {}) {
  const bar = findRawBar(score, target?.bar);
  if (!bar) return { ok: false, message: '找不到该小节' };
  const found = resolveChordTarget(bar, target);
  if (!found) return { ok: false, message: '找不到该和弦' };
  const { chord } = found;

  const displayShift = Number(opts.displayShift) || 0;
  const rawPreferFlats = opts.rawPreferFlats ?? preferFlatsForKey(score?.meta?.key);
  const viewPreferFlats = opts.viewPreferFlats ?? false;
  const typed = normalizeChordInput(viewSymbol);
  if (!typed) return { ok: false, message: '和弦符号为空' };
  if (!isChordLike(typed)) return { ok: false, message: `「${typed}」不是合法的和弦符号` };

  const rawSymbol = viewChordToRaw(typed, displayShift, { preferFlats: rawPreferFlats });
  const oldRaw = chord.symbol || null;
  const viewFrom = oldRaw ? rawChordToView(oldRaw, displayShift, { preferFlats: viewPreferFlats }) : '';
  const beatFrom = roundBeat(chord.beat ?? 0);
  const beatTo = typeof opts.moveToBeat === 'number' && Number.isFinite(opts.moveToBeat)
    ? roundBeat(Math.max(0, opts.moveToBeat))
    : beatFrom;

  const symbolChanged = rawSymbol !== oldRaw;
  const beatChanged = beatTo !== beatFrom;
  if (!symbolChanged && !beatChanged) return { ok: true, noop: true, rawSymbol };

  const oldVoicingId = chord.voicing_id || null;
  let voicing = null;
  if (symbolChanged) {
    chord.symbol = rawSymbol;
    delete chord.voicing_id;
    // concert = 发声音高下的符号（有 capo 时与记号不同）
    if (Object.prototype.hasOwnProperty.call(chord, 'concert')) {
      const capo = Number(score?.meta?.capo) || 0;
      chord.concert = capo ? transposeChord(rawSymbol, capo, rawPreferFlats) : rawSymbol;
    }
    voicing = typeof opts.resolveVoicing === 'function' ? opts.resolveVoicing(rawSymbol) : null;
    if (voicing) {
      if (voicing.id) chord.voicing_id = voicing.id;
      retargetHits(bar, oldVoicingId, chord, beatTo, voicing);
    }
  }
  if (beatChanged) {
    chord.beat = beatTo;
    const chords = asArray(bar.chords);
    chords.sort((a, b) => (Number(a?.beat) || 0) - (Number(b?.beat) || 0));
  }
  chord[EDIT_MARK] = true;

  const entry = {
    kind: 'chord',
    bar: Number(bar.index),
    beat: beatTo,
    from: oldRaw || '',
    to: rawSymbol,
    shift: displayShift,
  };
  if (displayShift) {
    entry.from_view = viewFrom || '';
    entry.to_view = typed;
  } else {
    entry.from_view = viewFrom || '';
    entry.to_view = typed;
  }
  if (beatChanged) entry.beat_from = beatFrom;
  if (oldVoicingId && voicing?.id && oldVoicingId !== voicing.id) entry.voicing = [oldVoicingId, voicing.id];
  recordEdit(score, entry);
  return { ok: true, rawSymbol, viewSymbol: typed, entry, moved: beatChanged };
}

/** 删除和弦 */
export function applyChordDelete(score, target) {
  const bar = findRawBar(score, target?.bar);
  if (!bar) return { ok: false, message: '找不到该小节' };
  const found = resolveChordTarget(bar, target);
  if (!found) return { ok: false, message: '找不到该和弦' };
  const { chord, idx } = found;
  const displayShift = Number(target?.displayShift) || 0;
  const from = chord.symbol || '';
  const viewFrom = from
    ? rawChordToView(from, displayShift, { preferFlats: !!target?.viewPreferFlats })
    : '';
  bar.chords.splice(idx, 1);
  if (!bar.chords.length) delete bar.chords;
  recordEdit(score, {
    kind: 'chord-delete',
    bar: Number(bar.index),
    beat: roundBeat(chord.beat ?? 0),
    from,
    from_view: viewFrom,
    shift: displayShift,
  });
  return { ok: true, from };
}

/* ─────────────────────────── 歌词编辑 ─────────────────────────── */

/**
 * 改/加/删歌词。
 * text 为空 → lyric = null，并在与前音同音高时标 tie（该音变成延音）。
 *
 * @param {object} score raw score
 * @param {{bar:number, note:number}} target note = bar.vocal 数组下标
 * @param {string|null} text
 */
export function applyLyricEdit(score, target, text) {
  const bar = findRawBar(score, target?.bar);
  if (!bar) return { ok: false, message: '找不到该小节' };
  const vocal = asArray(bar.vocal);
  const noteIdx = Number(target?.note);
  const note = vocal[noteIdx];
  if (!note || typeof note !== 'object') return { ok: false, message: '找不到该音符' };
  if (note.rest || note.midi == null) return { ok: false, message: '这个位置是休止符，不能填词' };

  const before = note.lyric == null ? null : String(note.lyric);
  const next = text == null ? null : String(text).trim();
  const after = next ? next : null;
  if ((before ?? null) === (after ?? null)) return { ok: true, noop: true };

  if (after) {
    note.lyric = after;
    // 有独立音节就不再是上一音的延音
    if (note.tie === 'stop' || note.tie === 'continue') delete note.tie;
    delete note[LYRIC_CLEAR_MARK];
  } else {
    note.lyric = null;
    // 变成延音：与前一个发声音同音高时标 tie，让渲染器合并成一个长音
    const prev = previousSoundingNote(bar, noteIdx);
    if (prev && Number(prev.midi) === Number(note.midi) && !note.tie) note.tie = 'stop';
    note[LYRIC_CLEAR_MARK] = true;
  }
  note[EDIT_MARK] = true;
  recordEdit(score, {
    kind: 'lyric',
    bar: Number(bar.index),
    note: noteIdx,
    beat: roundBeat(note.beat ?? 0),
    from: before,
    to: after,
  });
  return { ok: true, from: before, to: after, cleared: !after };
}

/* ─────────────────────────── 小节级操作 ─────────────────────────── */

/** 编辑日志里的小节号跟着重编号（insert/delete 之后） */
function renumberEdits(score, atBar, delta, { dropBar = null } = {}) {
  const edits = asArray(score?.edits);
  if (!edits.length) return;
  const kept = [];
  for (const entry of edits) {
    if (typeof entry?.bar !== 'number') {
      kept.push(entry);
      continue;
    }
    if (dropBar != null && entry.bar === dropBar) continue;
    if (entry.bar > atBar) entry.bar += delta;
    kept.push(entry);
  }
  score.edits = kept;
}

function barSeconds(score, beats) {
  const tempo = Number(score?.meta?.tempo) || 84;
  return (beats * 60) / tempo;
}

/** 在 barIndex 之后插入一个小节，后续小节/段落重编号 */
export function applyBarInsert(score, barIndex) {
  const bars = asArray(score?.bars);
  if (!bars.length) return { ok: false, message: '谱面为空' };
  const pos = bars.findIndex((bar) => Number(bar?.index) === Number(barIndex));
  if (pos < 0) return { ok: false, message: '找不到该小节' };
  const src = bars[pos];
  const newIndex = Number(src.index) + 1;
  const ts = Array.isArray(src.time_signature)
    ? [...src.time_signature]
    : Array.isArray(score.meta?.time_signature)
      ? [...score.meta.time_signature]
      : [4, 4];
  const beats = barBeats(ts);
  const startSec = Number.isFinite(Number(src.end_sec)) ? Number(src.end_sec) : undefined;
  const next = bars[pos + 1];

  const newBar = {
    index: newIndex,
    section: src.section,
    time_signature: ts.slice(),
  };
  if (startSec != null) {
    newBar.start_sec = Number(startSec.toFixed(4));
    newBar.end_sec = Number((startSec + barSeconds(score, beats)).toFixed(4));
    void next;
  }
  // 继承前一节最后一个和弦，用户只需改符号，不用先新建
  const lastChord = asArray(src.chords).at(-1);
  if (lastChord?.symbol) {
    newBar.chords = [{ beat: 0, dur: beats, symbol: lastChord.symbol }];
    if (lastChord.concert) newBar.chords[0].concert = lastChord.concert;
  }
  newBar.vocal = [];
  if (src.guitar?.pattern_id) newBar.guitar = { pattern_id: src.guitar.pattern_id };
  newBar[STRUCT_MARK] = 'insert';

  bars.splice(pos + 1, 0, newBar);
  for (const bar of bars.slice(pos + 2)) {
    if (Number.isFinite(Number(bar?.index))) bar.index = Number(bar.index) + 1;
  }
  for (const section of asArray(score.sections)) {
    if (Number(section.start_bar) > Number(barIndex)) section.start_bar = Number(section.start_bar) + 1;
    if (Number(section.end_bar) >= Number(barIndex)) section.end_bar = Number(section.end_bar) + 1;
  }
  renumberEdits(score, Number(barIndex), 1);
  recordEdit(score, { kind: 'bar-insert', bar: Number(barIndex) });
  return { ok: true, index: newIndex };
}

/** 删除一个小节，后续小节/段落重编号；段落被删空时一并移除 */
export function applyBarDelete(score, barIndex) {
  const bars = asArray(score?.bars);
  if (bars.length <= 1) return { ok: false, message: '只剩一个小节，不能再删' };
  const pos = bars.findIndex((bar) => Number(bar?.index) === Number(barIndex));
  if (pos < 0) return { ok: false, message: '找不到该小节' };
  const removed = bars[pos];
  bars.splice(pos, 1);
  for (const bar of bars.slice(pos)) {
    if (Number.isFinite(Number(bar?.index))) bar.index = Number(bar.index) - 1;
  }
  const sections = asArray(score.sections);
  const survivors = [];
  for (const section of sections) {
    const start = Number(section.start_bar) || 1;
    const end = Number(section.end_bar) || start;
    // 先按删除位置平移区间，再判断这段是否被删空：
    // 「删掉的小节」落在段内时，段的长度减一；一旦 end < start 就说明
    // 这个段落一小节都不剩了，必须整段移除（否则会留下 1-1 这种
    // 与后一段重叠的幽灵段落，bar.section 归属也会错）。
    let nextStart = start > Number(barIndex) ? start - 1 : start;
    let nextEnd = end >= Number(barIndex) ? end - 1 : end;
    if (nextEnd < nextStart) continue;
    nextStart = Math.max(1, nextStart);
    nextEnd = Math.max(nextStart, nextEnd);
    if (nextStart > bars.length) continue; // 段落整个被删掉
    section.start_bar = nextStart;
    section.end_bar = Math.min(nextEnd, bars.length);
    survivors.push(section);
  }
  if (sections.length) score.sections = survivors;
  // 小节引用的段落 id 失效时，按小节号重新归属
  const ids = new Set(survivors.map((s) => s.id));
  for (const bar of bars) {
    if (bar.section && ids.has(bar.section)) continue;
    const owner = survivors.find(
      (s) => Number(bar.index) >= Number(s.start_bar) && Number(bar.index) <= Number(s.end_bar)
    );
    if (owner) bar.section = owner.id;
    else if (survivors.length) bar.section = survivors[survivors.length - 1].id;
    else delete bar.section;
  }
  renumberEdits(score, Number(barIndex), -1, { dropBar: Number(barIndex) });
  recordEdit(score, { kind: 'bar-delete', bar: Number(barIndex), from: removed?.section || '' });
  return { ok: true, removed };
}

/** 改该小节拍号 */
export function applyTimeSignature(score, barIndex, timeSignature) {
  const bar = findRawBar(score, barIndex);
  if (!bar) return { ok: false, message: '找不到该小节' };
  if (!Array.isArray(timeSignature) || timeSignature.length !== 2) {
    return { ok: false, message: '拍号格式不对' };
  }
  const next = timeSignature.map((n) => Math.round(Number(n))).slice(0, 2);
  if (!next.every((n) => Number.isFinite(n) && n > 0)) return { ok: false, message: '拍号数值不对' };
  const before = Array.isArray(bar.time_signature) ? [...bar.time_signature] : null;
  const beforeText = fmtTimeSignature(before || score?.meta?.time_signature);
  const afterText = fmtTimeSignature(next);
  if (beforeText === afterText && before) return { ok: true, noop: true };
  bar.time_signature = next;
  bar[EDIT_MARK] = true;
  recordEdit(score, {
    kind: 'time-signature',
    bar: Number(bar.index),
    from: beforeText,
    to: afterText,
  });
  return { ok: true, from: beforeText, to: afterText };
}

export { TIME_SIGNATURES };

/* ─────────────────────────── 撤销 / 重做 ─────────────────────────── */

export class EditHistory {
  constructor(limit = DEFAULT_HISTORY) {
    this.limit = Math.max(2, Math.floor(Number(limit) || DEFAULT_HISTORY));
    this.undoStack = [];
    this.redoStack = [];
  }

  get canUndo() {
    return this.undoStack.length > 0;
  }

  get canRedo() {
    return this.redoStack.length > 0;
  }

  get size() {
    return this.undoStack.length;
  }

  /** 记一次「编辑前」的快照 */
  push(raw, label = '') {
    if (!raw) return;
    this.undoStack.push({ raw: clone(raw), label });
    while (this.undoStack.length > this.limit) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  /** @returns {{raw:object, label:string}|null} 恢复到的快照 */
  undo(current) {
    const entry = this.undoStack.pop();
    if (!entry) return null;
    // 当前状态入 redo 栈，供 redo 前进回来。current 缺失时不做记录，
    // 因为「重做到当前状态」这件事本身无意义（canRedo 保持 false）。
    if (current != null) this.redoStack.push({ raw: clone(current), label: entry.label });
    return entry;
  }

  redo(current) {
    const entry = this.redoStack.pop();
    if (!entry) return null;
    if (current) this.undoStack.push({ raw: clone(current), label: entry.label });
    return entry;
  }

  clear() {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
  }
}

/* ─────────────────────────── DOM 小工具 ─────────────────────────── */

function el(tag, cls = '', text = '') {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== '' && text != null) node.textContent = String(text);
  return node;
}

function isTypingTarget(target) {
  if (!target) return false;
  const tag = target.tagName?.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable === true;
}

/* ─────────────────────────── 编辑器（浏览器 UI） ─────────────────────────── */

/**
 * ScoreEditor：把数据层接到现有渲染管线上。
 * 依赖由 main.js 注入，编辑器自己不持有 score（永远从 getState() 读，避免副本失同步）。
 *
 * deps = {
 *   getState,        // () => state（main.js 的私有状态）
 *   setRaw,          // (score) => void  写回 raw（不触发渲染）
 *   render,          // () => void  现有的重渲染管线
 *   getDisplayShift, // () => number  main.js 的 displayShift()
 *   getVoicingJson,  // () => voicings.json
 *   resolveVoicing,  // (symbol) => voicing|null  用渲染器的指法库选把位
 *   toast,           // (msg, opts) => void
 *   renderer,        // ScoreRenderer 实例
 *   historyLimit,    // 可选
 * }
 */
export class ScoreEditor {
  constructor(deps) {
    this.deps = deps || {};
    this.renderer = this.deps.renderer || null;
    this.history = new EditHistory(this.deps.historyLimit || DEFAULT_HISTORY);
    this.layer = null;
    this.pop = null;
    this.menu = null;
    this.lyricInput = null;
    this.openState = null;
    this._committing = false;
    // 编辑态默认关闭，由 main.js 的「编辑」按钮打开
    this.enabled = false;
    this._onKeyDown = (event) => this.handleKeyDown(event);
    this._onPointerDown = (event) => this.handleOutsidePointer(event);
    this._onViewportChange = () => {
      if (this.openState) this.cancel();
    };
  }

  /* ── 生命周期 ── */

  init() {
    this.mount();
    this.attach();
    document.addEventListener('keydown', this._onKeyDown, true);
    document.addEventListener('pointerdown', this._onPointerDown, true);
    window.addEventListener('resize', this._onViewportChange);
    document.addEventListener('scroll', this._onViewportChange, true);
    return this;
  }

  mount() {
    if (typeof document === 'undefined') return;
    let layer = document.getElementById('editor-layer');
    if (!layer) {
      layer = el('div', 'editor-layer');
      layer.id = 'editor-layer';
      document.body.appendChild(layer);
    }
    layer.hidden = false;
    this.layer = layer;
  }

  /** 把编辑动作挂到渲染器的点击回调上 */
  attach() {
    const renderer = this.renderer || this.deps.renderer;
    if (!renderer) return;
    this.renderer = renderer;
    renderer.opts.onChordClick = (barIndex, info) => {
      if (!this.enabled) return;
      this.openChord(barIndex, info);
    };
    renderer.opts.onLyricClick = (barIndex, info) => {
      if (!this.enabled) return;
      this.openLyric(barIndex, info);
    };
    renderer.opts.onBarNumberClick = (barIndex, info) => {
      if (!this.enabled) return;
      this.openBarMenu(barIndex, info);
    };
    renderer.opts.editLyrics = true;
  }

  /**
   * 编辑开关。
   *
   * 默认关闭：普通用户只是想看谱，误点和弦就弹出编辑框会打断阅读。
   * 打开后页面进入编辑态（CSS 依赖 body[data-edit] 画可点提示）。
   */
  setEnabled(on) {
    this.enabled = Boolean(on);
    if (typeof document !== 'undefined') {
      document.body.dataset.edit = this.enabled ? '1' : '0';
    }
    // 编辑时让出整块 bar-hit 热区，否则它会吃掉点和弦/歌词的点击
    if (this.renderer) {
      this.renderer.opts.editing = this.enabled;
      const score = this.getState().view || this.getState().raw;
      if (score) this.renderer.render(score, { width: this.renderer.width });
    }
    if (!this.enabled) this.close();
    return this.enabled;
  }

  toggleEnabled() {
    return this.setEnabled(!this.enabled);
  }

  /** 换了一份 score（含 loadDemo / 上传结果）：清空历史 */
  resetHistory() {
    this.history.clear();
    this.close();
  }

  dispose() {
    document.removeEventListener('keydown', this._onKeyDown, true);
    document.removeEventListener('pointerdown', this._onPointerDown, true);
    window.removeEventListener('resize', this._onViewportChange);
    document.removeEventListener('scroll', this._onViewportChange, true);
    this.close();
  }

  /* ── 状态读取 ── */

  getState() {
    return this.deps.getState?.() || {};
  }

  getRaw() {
    return this.getState().raw || null;
  }

  getView() {
    const state = this.getState();
    return state.view || state.raw || null;
  }

  getShift() {
    const shift = Number(this.deps.getDisplayShift?.());
    return Number.isFinite(shift) ? Math.round(shift) : 0;
  }

  getVoicingJson() {
    return this.deps.getVoicingJson?.() || null;
  }

  resolveVoicing(symbol) {
    try {
      return this.deps.resolveVoicing?.(symbol) || null;
    } catch (err) {
      return null;
    }
  }

  toast(message, opts) {
    this.deps.toast?.(message, opts);
  }

  get canUndo() {
    return this.history.canUndo;
  }

  get canRedo() {
    return this.history.canRedo;
  }

  /* ── 提交 / 撤销 ── */

  /**
   * 统一的提交入口：克隆 raw → 在草稿上改 → 入栈 → 写回 → 重渲染。
   * @param {string} label
   * @param {(draft:object)=>{ok?:boolean, noop?:boolean, message?:string}} mutate
   */
  commit(label, mutate) {
    const state = this.getState();
    const raw = state.raw;
    if (!raw) return { ok: false, message: '还没有加载谱面' };
    const draft = clone(raw);
    let result = null;
    try {
      result = mutate(draft) || { ok: true };
    } catch (err) {
      console.error('[editor] 编辑失败', err);
      return { ok: false, message: err.message || '编辑失败' };
    }
    if (!result || result.ok === false) {
      if (result?.message) this.toast(result.message, { kind: 'error' });
      return result || { ok: false };
    }
    if (result.noop) return result;
    this.history.push(raw, label);
    this.deps.setRaw?.(draft);
    this.close();
    this.deps.render?.();
    return result;
  }

  undo() {
    const state = this.getState();
    if (!state.raw) return false;
    const entry = this.history.undo(state.raw);
    if (!entry) {
      this.toast('没有可撤销的编辑', { kind: 'info' });
      return false;
    }
    this.close();
    this.deps.setRaw?.(entry.raw);
    this.deps.render?.();
    this.toast(`已撤销：${this.history.redoStack.at(-1)?.label || '上一步编辑'}`, { kind: 'ok' });
    return true;
  }

  redo() {
    const state = this.getState();
    if (!state.raw) return false;
    const entry = this.history.redo(state.raw);
    if (!entry) {
      this.toast('没有可重做的编辑', { kind: 'info' });
      return false;
    }
    this.close();
    this.deps.setRaw?.(entry.raw);
    this.deps.render?.();
    this.toast(`已重做：${entry.label || '编辑'}`, { kind: 'ok' });
    return true;
  }

  /* ── 和弦面板 ── */

  openChord(barIndex, info = {}) {
    const raw = this.getRaw();
    if (!raw) return null;
    const bar = findRawBar(raw, barIndex);
    if (!bar) return null;
    const found = resolveChordTarget(bar, { chordIdx: info.chordIdx, beat: info.beat });
    if (!found) return null;
    const shift = this.getShift();
    const viewKey = this.getView()?.meta?.key || raw.meta?.key || {};
    const current = found.chord.symbol
      ? rawChordToView(found.chord.symbol, shift, { preferFlats: preferFlatsForKey(viewKey) })
      : '';
    this.close();
    const pop = el('div', 'editor-pop');
    pop.setAttribute('role', 'dialog');
    pop.setAttribute('aria-label', '编辑和弦');
    pop.dataset.kind = 'chord';

    const head = el('div', 'editor-pop__head');
    head.append(
      el('span', 'editor-pop__title', `第 ${bar.index} 小节`),
      el('span', 'editor-pop__meta', `第 ${(Number(found.chord.beat) || 0) + 1} 拍`)
    );
    pop.appendChild(head);

    const field = el('label', 'editor-field');
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'editor-input';
    input.value = current || '';
    input.setAttribute('spellcheck', 'false');
    input.setAttribute('autocomplete', 'off');
    input.setAttribute('aria-label', '和弦符号');
    field.appendChild(input);
    pop.appendChild(field);

    // 常用和弦（当前显示调内的 I-IV-V-vi）
    const quick = el('div', 'editor-quick');
    quick.setAttribute('role', 'group');
    quick.setAttribute('aria-label', '当前调常用和弦');
    const tonic = viewKey.play || viewKey.sounding || 'C';
    const quickItems = diatonicChords(tonic, viewKey.mode || 'major', {
      preferFlats: preferFlatsForKey(viewKey),
    });
    for (const item of quickItems) {
      const btn = el('button', 'editor-quick__btn');
      btn.type = 'button';
      btn.dataset.chord = item.symbol;
      btn.title = `${item.degree} · ${item.symbol}`;
      btn.append(el('span', 'editor-quick__deg', item.degree), el('span', 'editor-quick__sym', item.symbol));
      quick.appendChild(btn);
    }
    pop.appendChild(quick);

    // 和弦位置（等价于拖到别的 beat，P2 的轻量实现）
    const beats = Math.max(1, Math.ceil(barBeats(bar.time_signature || raw.meta?.time_signature || [4, 4])));
    const beatRow = el('div', 'editor-beats');
    beatRow.append(el('span', 'editor-beats__label', '位置'));
    let pendingBeat = roundBeat(found.chord.beat ?? 0);
    for (let b = 0; b < beats; b++) {
      const btn = el('button', 'editor-beats__btn', `${b + 1}`);
      btn.type = 'button';
      btn.dataset.beat = String(b);
      if (b === Math.round(pendingBeat)) btn.classList.add('is-active');
      beatRow.appendChild(btn);
    }
    pop.appendChild(beatRow);

    const hint = el('p', 'editor-hint');
    hint.dataset.role = 'hint';
    pop.appendChild(hint);

    const actions = el('div', 'editor-actions');
    const del = el('button', 'editor-btn editor-btn--danger', '删除和弦');
    del.type = 'button';
    del.dataset.editorAct = 'delete';
    const cancel = el('button', 'editor-btn', '取消');
    cancel.type = 'button';
    cancel.dataset.editorAct = 'cancel';
    const confirm = el('button', 'editor-btn editor-btn--primary', '确定');
    confirm.type = 'button';
    confirm.dataset.editorAct = 'confirm';
    actions.append(del, el('span', 'editor-spacer'), cancel, confirm);
    pop.appendChild(actions);

    this.layer.appendChild(pop);
    this.pop = pop;
    this.openState = {
      kind: 'chord',
      bar: Number(bar.index),
      chordIdx: found.idx,
      beat: roundBeat(found.chord.beat ?? 0),
      original: current,
      target: info.target || null,
    };

    const validate = () => {
      const value = normalizeChordInput(input.value);
      const status = chordLibraryStatus(value, this.getVoicingJson());
      pop.dataset.status = status.status;
      hint.className = 'editor-hint';
      if (status.status === 'empty') {
        hint.textContent = '清空后点「删除和弦」即可移除。';
      } else if (status.status === 'invalid') {
        hint.classList.add('editor-hint--error');
        hint.textContent = `「${value}」不是合法的和弦符号（根音用 A–G）。`;
      } else if (status.status === 'lib') {
        hint.classList.add('editor-hint--ok');
        hint.textContent = status.note ? `指法库命中（${status.note}）` : '指法库中已有这把位。';
      } else if (status.status === 'derived') {
        hint.classList.add('editor-hint--warn');
        hint.textContent = '库中无此指法，将不显示指法图（渲染器会按符号自动推算备用把位）。';
      } else {
        hint.classList.add('editor-hint--warn');
        hint.textContent = '指法库未加载，无法校验。';
      }
      const ok = value !== '' && isChordLike(value);
      confirm.disabled = !ok;
      return ok;
    };
    validate();

    input.addEventListener('input', validate);
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        event.stopPropagation();
        if (validate()) this.confirmChord();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        this.cancel();
      }
    });
    quick.addEventListener('click', (event) => {
      const btn = event.target.closest('[data-chord]');
      if (!btn) return;
      input.value = btn.dataset.chord;
      validate();
      this.confirmChord();
    });
    beatRow.addEventListener('click', (event) => {
      const btn = event.target.closest('[data-beat]');
      if (!btn) return;
      pendingBeat = Number(btn.dataset.beat);
      beatRow.querySelectorAll('[data-beat]').forEach((node) => {
        node.classList.toggle('is-active', Number(node.dataset.beat) === pendingBeat);
      });
      this.openState.beatTo = pendingBeat;
    });
    actions.addEventListener('click', (event) => {
      const btn = event.target.closest('[data-editor-act]');
      if (!btn) return;
      const act = btn.dataset.editorAct;
      if (act === 'cancel') this.cancel();
      else if (act === 'delete') this.deleteChord();
      else if (act === 'confirm') {
        if (validate()) this.confirmChord();
      }
    });

    this.positionNear(pop, info.target);
    requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
    this.renderer?.setActiveBar?.(Number(bar.index) - 1);
    return pop;
  }

  confirmChord() {
    const state = this.openState;
    if (!state || state.kind !== 'chord' || !this.pop) return null;
    const input = this.pop.querySelector('.editor-input');
    const value = normalizeChordInput(input?.value);
    const beatTo = state.beatTo;
    const result = this.commit('改和弦', (draft) =>
      applyChordEdit(
        draft,
        { bar: state.bar, chordIdx: state.chordIdx, beat: state.beat },
        value,
        {
          displayShift: this.getShift(),
          rawPreferFlats: preferFlatsForKey(draft.meta?.key),
          viewPreferFlats: preferFlatsForKey(this.getView()?.meta?.key),
          moveToBeat: beatTo,
          resolveVoicing: (symbol) => this.resolveVoicing(symbol),
        }
      )
    );
    if (result?.ok && !result.noop) {
      const where = this.getShift() ? '（按当前移调反算）' : '';
      this.toast(`已改为 ${value}${where}`, { kind: 'ok' });
    }
    return result;
  }

  deleteChord() {
    const state = this.openState;
    if (!state || state.kind !== 'chord') return null;
    return this.commit('删除和弦', (draft) =>
      applyChordDelete(draft, {
        bar: state.bar,
        chordIdx: state.chordIdx,
        beat: state.beat,
        displayShift: this.getShift(),
        viewPreferFlats: preferFlatsForKey(this.getView()?.meta?.key),
      })
    );
  }

  /* ── 歌词编辑（原位 input） ── */

  openLyric(barIndex, info = {}) {
    const raw = this.getRaw();
    if (!raw) return null;
    const bar = findRawBar(raw, barIndex);
    const noteIdx = Number(info.noteIdx);
    const note = asArray(bar?.vocal)[noteIdx];
    if (!note) return null;
    this.close();
    const input = el('input', 'editor-lyric-input');
    input.type = 'text';
    input.setAttribute('aria-label', '编辑歌词');
    input.setAttribute('spellcheck', 'false');
    input.setAttribute('autocomplete', 'off');
    input.value = note.lyric == null ? '' : String(note.lyric);
    this.layer.appendChild(input);
    this.lyricInput = input;
    this.openState = {
      kind: 'lyric',
      bar: Number(bar.index),
      noteIdx,
      original: note.lyric ?? null,
      target: info.target || null,
    };

    const commitLyric = () => {
      if (this._committing) return;
      const state = this.openState;
      if (!state || state.kind !== 'lyric') return;
      this._committing = true;
      const value = input.value;
      const target = { bar: state.bar, note: state.noteIdx };
      this.hideLyricInput();
      const result = this.commit('改歌词', (draft) => applyLyricEdit(draft, target, value));
      this._committing = false;
      if (result?.ok && !result.noop) {
        this.toast(value.trim() ? `歌词已改为「${value.trim()}」` : '已删除歌词（该音变延音）', { kind: 'ok' });
      }
      return result;
    };

    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        event.stopPropagation();
        commitLyric();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        this._committing = true;
        this.hideLyricInput();
        this.openState = null;
        this._committing = false;
      }
    });
    input.addEventListener('blur', () => {
      if (this._committing) return;
      commitLyric();
    });

    this.placeLyricInput(input, info.target);
    requestAnimationFrame(() => {
      input.focus();
      input.select();
    });
    this.renderer?.setActiveBar?.(Number(bar.index) - 1);
    return input;
  }

  hideLyricInput() {
    if (this.lyricInput) {
      this.lyricInput.hidden = true;
      this.lyricInput.remove();
      this.lyricInput = null;
    }
  }

  placeLyricInput(input, target) {
    const rect = target?.getBoundingClientRect?.();
    const vw = window.innerWidth;
    if (!rect || !rect.width) {
      input.style.left = '8px';
      input.style.top = '8px';
      input.style.width = '120px';
      return;
    }
    const width = Math.max(64, Math.min(220, rect.width + 36));
    const left = Math.max(6, Math.min(rect.left + rect.width / 2 - width / 2, vw - width - 6));
    input.style.width = `${Math.round(width)}px`;
    input.style.left = `${Math.round(left)}px`;
    input.style.top = `${Math.round(rect.top - 4)}px`;
    input.style.height = `${Math.max(22, Math.round(rect.height + 6))}px`;
  }

  /* ── 小节号菜单 ── */

  openBarMenu(barIndex, info = {}) {
    const raw = this.getRaw();
    if (!raw) return null;
    const bar = findRawBar(raw, barIndex);
    if (!bar) return null;
    this.close();
    const menu = el('div', 'editor-pop editor-menu');
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', `第 ${bar.index} 小节操作`);
    menu.dataset.kind = 'bar';

    const head = el('div', 'editor-pop__head');
    head.append(
      el('span', 'editor-pop__title', `第 ${bar.index} 小节`),
      el('span', 'editor-pop__meta', `拍号 ${fmtTimeSignature(bar.time_signature || raw.meta?.time_signature || [4, 4])}`)
    );
    menu.appendChild(head);

    const list = el('div', 'editor-menu__list');
    const add = (label, act, extra = '') => {
      const btn = el('button', 'editor-menu__item');
      btn.type = 'button';
      btn.dataset.editorAct = act;
      btn.append(el('span', 'editor-menu__label', label));
      if (extra) btn.append(el('span', 'editor-menu__extra', extra));
      list.appendChild(btn);
      return btn;
    };
    add('在下方插入 1 小节', 'insert', '空拍');
    add('删除本小节', 'delete', '后续小节重编号');
    const tsRow = el('div', 'editor-menu__row');
    tsRow.append(el('span', 'editor-menu__rowlabel', '本小节拍号'));
    for (const ts of TIME_SIGNATURES) {
      const text = fmtTimeSignature(ts);
      const btn = el('button', 'editor-menu__chip', text);
      btn.type = 'button';
      btn.dataset.ts = text;
      const active = fmtTimeSignature(bar.time_signature || raw.meta?.time_signature || [4, 4]) === text;
      if (active) btn.classList.add('is-active');
      tsRow.appendChild(btn);
    }
    list.appendChild(tsRow);
    menu.appendChild(list);

    this.layer.appendChild(menu);
    this.pop = menu;
    this.openState = { kind: 'bar', bar: Number(bar.index), target: info.target || null };

    list.addEventListener('click', (event) => {
      const chip = event.target.closest('[data-ts]');
      if (chip) {
        const [num, den] = chip.dataset.ts.split('/').map(Number);
        const result = this.commit('改拍号', (draft) => applyTimeSignature(draft, barIndex, [num, den]));
        if (result?.ok) {
          this.toast(result.noop ? '拍号没有变化' : `第 ${barIndex} 小节拍号改为 ${chip.dataset.ts}`, {
            kind: result.noop ? 'info' : 'ok',
          });
        }
        return;
      }
      const btn = event.target.closest('[data-editor-act]');
      if (!btn) return;
      if (btn.dataset.editorAct === 'insert') {
        const result = this.commit('插入小节', (draft) => applyBarInsert(draft, barIndex));
        if (result?.ok) this.toast(`已在第 ${barIndex} 小节后插入 1 小节`, { kind: 'ok' });
      } else if (btn.dataset.editorAct === 'delete') {
        const result = this.commit('删除小节', (draft) => applyBarDelete(draft, barIndex));
        if (result?.ok) this.toast(`已删除第 ${barIndex} 小节`, { kind: 'ok' });
      }
    });

    this.positionNear(menu, info.target);
    this.renderer?.setActiveBar?.(Number(bar.index) - 1);
    return menu;
  }

  /* ── 面板定位 / 关闭 ── */

  positionNear(node, target) {
    const rect = target?.getBoundingClientRect?.() || {
      left: window.innerWidth / 2,
      right: window.innerWidth / 2,
      top: 120,
      bottom: 140,
      width: 0,
      height: 0,
    };
    node.style.visibility = 'hidden';
    node.style.left = '0px';
    node.style.top = '0px';
    const width = node.offsetWidth || 240;
    const height = node.offsetHeight || 160;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(8, Math.min(left, vw - width - 8));
    let top = rect.bottom + 8;
    if (top + height > vh - 8) top = Math.max(8, rect.top - height - 8);
    node.style.left = `${Math.round(left)}px`;
    node.style.top = `${Math.round(top)}px`;
    node.style.visibility = '';
  }

  close() {
    if (this.pop) {
      this.pop.remove();
      this.pop = null;
    }
    this.hideLyricInput();
    this.openState = null;
    this.renderer?.setActiveBar?.(null);
  }

  cancel() {
    this.close();
  }

  handleOutsidePointer(event) {
    if (!this.openState) return;
    const node = this.pop || this.lyricInput;
    if (!node) return;
    if (node.contains(event.target)) return;
    // 点在另一个可编辑对象上时不要先关再开：pointerdown 早于 click，
    // 提前 cancel 会让紧随其后的 openChord/openLyric 拿到空的 openState，
    // 表现为"第一次点歌词没反应，第二次才出来"。交给 click 里的 open* 自己重置。
    if (event.target?.closest?.('[data-chord], [data-lyric], [data-barnum]')) return;
    // 歌词 input 的失焦提交由 blur 负责；点外面先关面板
    this.cancel();
  }

  handleKeyDown(event) {
    if (event.key === 'Escape') {
      if (this.openState) {
        event.preventDefault();
        this.cancel();
      }
      return;
    }
    const typing = isTypingTarget(event.target);
    if (!typing && (event.ctrlKey || event.metaKey) && !event.altKey) {
      const key = String(event.key || '').toLowerCase();
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        this.undo();
        return;
      }
      if ((key === 'z' && event.shiftKey) || key === 'y') {
        event.preventDefault();
        this.redo();
      }
    }
  }

  /* ── 质量面板里的「已手工修正 N 处」 ── */

  afterRender() {
    if (typeof document === 'undefined') return;
    const state = this.getState();
    const content = document.getElementById('rail-content');
    if (!content) return;
    content.querySelector('.panel--editor')?.remove();
    const raw = state.raw;
    if (!raw) return;
    const total = countEdits(raw);

    const panel = el('section', 'panel panel--editor');
    const head = el('div', 'panel__head');
    head.append(
      el('h2', 'panel__title', '手工修正'),
      el('span', 'panel__aside', `${total} 处`)
    );
    panel.appendChild(head);

    const count = el('p', 'editor-count');
    count.dataset.role = 'edit-count';
    count.textContent = `已手工修正 ${total} 处`;
    panel.appendChild(count);

    const row = el('div', 'editor-panel__row');
    const undo = el('button', 'editor-btn editor-btn--small', '撤销');
    undo.type = 'button';
    undo.dataset.editorAct = 'undo';
    undo.disabled = !this.canUndo;
    undo.title = '撤销 (Ctrl+Z)';
    const redo = el('button', 'editor-btn editor-btn--small', '重做');
    redo.type = 'button';
    redo.dataset.editorAct = 'redo';
    redo.disabled = !this.canRedo;
    redo.title = '重做 (Ctrl+Shift+Z / Ctrl+Y)';
    row.append(undo, redo);
    panel.appendChild(row);

    const lines = recentEdits(raw, 4);
    if (lines.length) {
      const ul = el('ul', 'editor-log');
      for (const line of lines) ul.appendChild(el('li', 'editor-log__item', line));
      panel.appendChild(ul);
    }

    const hint = el(
      'p',
      'editor-panel__hint',
      '点和弦符号可改和弦 / 删和弦；点歌词可改词；点小节号可增删小节、改拍号。'
    );
    panel.appendChild(hint);

    row.addEventListener('click', (event) => {
      const btn = event.target.closest('[data-editor-act]');
      if (!btn || btn.disabled) return;
      if (btn.dataset.editorAct === 'undo') this.undo();
      else if (btn.dataset.editorAct === 'redo') this.redo();
    });

    content.appendChild(panel);
  }

  /** 导出（供 main.js 的导出菜单调用） */
  buildExport({ clean = false } = {}) {
    const raw = this.getRaw();
    if (!raw) return null;
    return buildExport(raw, { clean });
  }
}

/** 浏览器入口 */
export function createEditor(deps) {
  const editor = new ScoreEditor(deps);
  editor.init();
  return editor;
}
