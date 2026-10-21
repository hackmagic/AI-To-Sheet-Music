/**
 * music.js —— 乐理与记谱转换工具（纯函数，无 DOM 依赖）
 *
 * 约定（与 schemas/score.schema.json 一致）：
 *  - 谱面时间单位是四分音符拍，dur 可以是 number 或 "1/3" 这样的分数字符串；
 *  - midi 为记谱音高（capo 已扣除）；capo 品数在 meta.capo 里；
 *  - "调号升降数" num_sharps_flats > 0 为升号，< 0 为降号。
 */

export const SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export const FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

/** 五度圈：fifths -> 大调主音名（-7..7） */
const MAJOR_BY_FIFTHS = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#'];
/** pc（0=C）-> 大调调号 fifths */
const MAJOR_FIFTHS_BY_PC = { 0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 5: -1, 10: -2, 3: -3, 8: -4 };
/** pc（0=C）-> 小调调号 fifths（相对大调） */
const MINOR_FIFTHS_BY_PC = { 9: 0, 4: 1, 11: 2, 6: 3, 1: 4, 8: 5, 3: 6, 2: -1, 7: -2, 0: -3, 5: -4, 10: -5 };

/** 段落 kind -> 中文默认名 */
export const SECTION_LABELS = {
  intro: '前奏',
  verse: '主歌',
  prechorus: '预副歌',
  chorus: '副歌',
  bridge: '桥段',
  interlude: '间奏',
  solo: '独奏',
  outro: '尾奏',
  other: '其他',
};

export const STAGE_LABELS = {
  beats: '节拍',
  tempo: '速度',
  key: '调性',
  chords: '和弦',
  structure: '段落',
  lyrics: '歌词',
  melody: '旋律',
  lyric_match: '歌词对位',
  bars: '小节',
  tempo_changes: '速度变化',
  voicing: '指法',
  separate: '分轨',
  asr: '歌词识别',
  overall: '总体',
};

/* ───────────────────────────── 音高 / 时值 ───────────────────────────── */

/** 把 dur 规整为“四分音符拍”数 */
export function durToBeats(dur, fallback = 1) {
  if (typeof dur === 'number' && isFinite(dur) && dur > 0) return dur;
  if (typeof dur === 'string') {
    const m = dur.trim().match(/^(\d+)\s*\/\s*(\d+)$/);
    if (m) {
      const v = Number(m[1]) / Number(m[2]);
      if (v > 0) return v;
    }
    const n = Number(dur);
    if (isFinite(n) && n > 0) return n;
  }
  return fallback;
}

/** 拍号 -> 每小节拍数（四分音符拍） */
export function barBeats(timeSignature) {
  const [num, den] = Array.isArray(timeSignature) && timeSignature.length === 2 ? timeSignature : [4, 4];
  if (!num || !den) return 4;
  return (num * 4) / den;
}

export function pcName(pc, preferFlats = false) {
  const norm = ((pc % 12) + 12) % 12;
  return (preferFlats ? FLAT_NAMES : SHARP_NAMES)[norm];
}

export function nameToPc(name) {
  if (!name) return 0;
  const raw = String(name).trim();
  const m = raw.match(/^([A-Ga-g])([#b♯♭]{0,2})/);
  if (!m) return 0;
  const base = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1].toUpperCase()];
  let acc = 0;
  for (const ch of m[2]) {
    if (ch === '#' || ch === '♯') acc += 1;
    if (ch === 'b' || ch === '♭') acc -= 1;
  }
  return (((base + acc) % 12) + 12) % 12;
}

const MAJOR_FIFTHS_BY_NAME = new Map(MAJOR_BY_FIFTHS.map((name, i) => [name, i - 7]));

/** 调号 fifths：返回 -7..7；输入可以是 "C"/"F#"/"Bb" 与 "major"/"minor"
 *  注意必须按“拼写”而不是按 pc 取，否则 C# 与 Db 会得到同一个调号。 */
export function fifthsFor(tonic, mode = 'major') {
  const raw = String(tonic || 'C').trim().replace(/m$/, '');
  if (mode !== 'minor') {
    const direct = MAJOR_FIFTHS_BY_NAME.get(raw);
    if (direct !== undefined) return direct;
    return MAJOR_FIFTHS_BY_PC[nameToPc(raw)] ?? 0;
  }
  // 小调：换算到关系大调（主音 + 3 个半音），升降号方向跟随小调主音的写法
  const preferFlats = /(b|♭)/.test(raw);
  const relPc = (nameToPc(raw) + 3) % 12;
  const relName = pcName(relPc, preferFlats);
  const rel = MAJOR_FIFTHS_BY_NAME.get(relName);
  return rel ?? MAJOR_FIFTHS_BY_PC[relPc] ?? 0;
}

/** 由 fifths 得到 VexFlow 的调号名（大调名；小调由调用方自行加 m） */
export function keyNameFromFifths(fifths) {
  const v = Math.max(-7, Math.min(7, Math.round(fifths || 0)));
  return MAJOR_BY_FIFTHS[v + 7];
}

/** 由 meta.key 得到 VexFlow 调号字符串（小调给 "Am" 这类形式） */
export function vexKeySignature(keyMeta) {
  if (!keyMeta) return 'C';
  const fifths =
    typeof keyMeta.num_sharps_flats === 'number'
      ? keyMeta.num_sharps_flats
      : fifthsFor(keyMeta.play || keyMeta.sounding || 'C', keyMeta.mode || 'major');
  const major = keyNameFromFifths(fifths);
  return keyMeta.mode === 'minor' ? `${major}m` : major;
}

/** 记谱偏好：降号调用降号拼写 */
export function prefersFlats(fifths) {
  return (fifths || 0) < 0;
}

/** midi -> VexFlow key，如 "c#/4"（fifths 决定升/降拼写） */
export function midiToVexKey(midi, fifths = 0) {
  const m = Math.round(midi);
  const pc = ((m % 12) + 12) % 12;
  const octave = Math.floor(m / 12) - 1;
  return `${pcName(pc, prefersFlats(fifths)).toLowerCase()}/${octave}`;
}

/** midi -> 频率 */
export function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/** midi -> 音名（显示用，如 "C4"） */
export function midiToName(midi, fifths = 0) {
  const m = Math.round(midi);
  return `${pcName(((m % 12) + 12) % 12, prefersFlats(fifths))}${Math.floor(m / 12) - 1}`;
}

/* ───────────────────────────── VexFlow 时值 ───────────────────────────── */

/**
 * 拍数 -> VexFlow 记谱原子。
 * 返回 { duration, tickMul:[n,d]|null, triplet:boolean }；
 * 无法用单个音符表达时返回 null（调用方拆分为连音）。
 */
export function beatsToVex(beats) {
  // [四分音符拍数, VexFlow 时值字符串]；VexFlow 4 用后缀 "d" 表示附点（8d = 附点八分）
  const table = [
    [4, 'w'],
    [3, 'hd'],
    [2, 'h'],
    [1.5, 'qd'],
    [1, 'q'],
    [0.75, '8d'],
    [0.5, '8'],
    [0.375, '16d'],
    [0.25, '16'],
    [0.125, '32'],
  ];
  const precise = Math.round(beats * 96) / 96;
  for (const [value, duration] of table) {
    if (Math.abs(precise - value) < 1e-6) {
      return { duration, tickMul: null, triplet: false, value };
    }
  }
  return null;
}

/** 三连音：dur 形如 "1/3" -> 8 分音符 ×2/3 */
export function tripletBeatsToVex(beats) {
  // 1/3 拍 = 八分三连（8 × 2/3），2/3 拍 = 四分三连（q × 2/3）
  const map = [
    [1 / 3, '8'],
    [2 / 3, 'q'],
    [4 / 3, 'h'],
  ];
  for (const [value, base] of map) {
    if (Math.abs(beats - value) < 1e-6) return { duration: base, tickMul: [2, 3], triplet: true };
  }
  return null;
}

/** 把一个拍数拆成若干标准时值（用于连音拆分） */
export function splitBeats(beats, maxParts = 4) {
  const out = [];
  let rest = Math.round(beats * 96) / 96;
  const pieces = [4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25, 1 / 3, 1 / 6];
  let guard = 0;
  while (rest > 1e-6 && out.length < maxParts && guard++ < 8) {
    let hit = null;
    for (const p of pieces) {
      if (p <= rest + 1e-9) {
        hit = p;
        break;
      }
    }
    if (!hit) break;
    out.push(hit);
    rest = Math.round((rest - hit) * 96) / 96;
  }
  if (out.length === 0) out.push(Math.max(0.125, beats));
  return out;
}

/** 判断 dur 是否为三连音记号 */
export function isTripletDur(dur) {
  if (typeof dur !== 'string') return false;
  const m = dur.match(/^(\d+)\s*\/\s*(\d+)$/);
  return !!m && Number(m[2]) === 3;
}

/* ───────────────────────────── 和弦符号 ───────────────────────────── */

const QUALITY_INTERVALS = [
  // 先匹配更长的后缀
  ['maj13', [0, 4, 7, 11, 14, 21]],
  ['maj9', [0, 4, 7, 11, 14]],
  ['maj7', [0, 4, 7, 11]],
  ['mmaj7', [0, 3, 7, 11]],
  ['m(maj7)', [0, 3, 7, 11]],
  ['m7b5', [0, 3, 6, 10]],
  ['dim7', [0, 3, 6, 9]],
  ['dim', [0, 3, 6]],
  ['aug', [0, 4, 8]],
  ['7sus4', [0, 5, 7, 10]],
  ['sus4', [0, 5, 7]],
  ['sus2', [0, 2, 7]],
  ['sus', [0, 5, 7]],
  ['m13', [0, 3, 7, 10, 14, 21]],
  ['m11', [0, 3, 7, 10, 14, 17]],
  ['m9', [0, 3, 7, 10, 14]],
  ['m7', [0, 3, 7, 10]],
  ['m6', [0, 3, 7, 9]],
  ['madd9', [0, 3, 7, 14]],
  ['m', [0, 3, 7]],
  ['13', [0, 4, 7, 10, 14, 21]],
  ['11', [0, 4, 7, 10, 14, 17]],
  ['9', [0, 4, 7, 10, 14]],
  ['add9', [0, 4, 7, 14]],
  ['7', [0, 4, 7, 10]],
  ['6', [0, 4, 7, 9]],
  ['5', [0, 7]],
  ['', [0, 4, 7]],
];

/** 解析和弦符号 -> { root, quality, bass, intervals } */
export function parseChord(symbol) {
  const raw = String(symbol || '').trim();
  if (!raw) return { root: 'C', quality: '', bass: null, intervals: [0, 4, 7] };
  const slash = raw.split('/');
  const head = slash[0].trim();
  const bassRaw = slash[1] ? slash[1].trim() : null;
  const m = head.match(/^([A-Ga-g])([#b♯♭]{0,2})(.*)$/);
  if (!m) return { root: 'C', quality: head, bass: null, intervals: [0, 4, 7] };
  let quality = (m[3] || '').trim();
  quality = quality.replace(/^M7$/, 'maj7').replace(/^Δ7?$/, 'maj7').replace(/^min/, 'm').replace(/^mi/, 'm');
  let intervals = [0, 4, 7];
  for (const [suffix, iv] of QUALITY_INTERVALS) {
    if (quality === suffix) {
      intervals = iv;
      break;
    }
    if (suffix && quality.startsWith(suffix) && suffix.length >= 1) {
      // 处理 7b9、9sus4 等未知扩展：取已知前缀
      intervals = iv;
      break;
    }
  }
  const rootName = m[1].toUpperCase() + (m[2] || '').replace(/♯/g, '#').replace(/♭/g, 'b');
  return { root: rootName, quality, bass: bassRaw, intervals };
}

/** 和弦的 pitch class 集合（根音在前） */
export function chordPcs(symbol) {
  const { root, intervals } = parseChord(symbol);
  const rootPc = nameToPc(root);
  const pcs = [];
  for (const iv of intervals) {
    const pc = (((rootPc + iv) % 12) + 12) % 12;
    if (!pcs.includes(pc)) pcs.push(pc);
  }
  return pcs;
}

/** 和弦低音 pitch class（斜杠和弦用斜杠音，否则根音） */
export function chordBassPc(symbol) {
  const { bass, root } = parseChord(symbol);
  return bass ? nameToPc(bass) : nameToPc(root);
}

/** 和弦移调 */
export function transposeChord(symbol, semitones, preferFlats = false) {
  if (!semitones) return symbol;
  const raw = String(symbol || '').trim();
  if (!raw) return symbol;
  const m = raw.match(/^([A-Ga-g])([#b♯♭]{0,2})(.*)$/);
  if (!m) return symbol;
  const rest = m[3] || '';
  const slashIdx = rest.lastIndexOf('/');
  const quality = slashIdx >= 0 ? rest.slice(0, slashIdx) : rest;
  const bass = slashIdx >= 0 ? rest.slice(slashIdx + 1) : null;
  const rootPc = nameToPc(m[1] + (m[2] || ''));
  const newRoot = pcName(rootPc + semitones, preferFlats);
  let out = newRoot + quality;
  if (bass) {
    const bassPc = nameToPc(bass);
    out += `/${pcName(bassPc + semitones, preferFlats)}`;
  }
  return out;
}

/** 移调调性对象（保持 mode，重算 fifths / sounding / play） */
export function transposeKey(keyMeta, semitones, opts = {}) {
  const mode = keyMeta?.mode || 'major';
  const displayName = keyMeta?.play || keyMeta?.sounding || 'C';
  const preferFlatsOut = opts.preferFlats ?? prefersFlats(keyMeta?.num_sharps_flats || 0);
  const newTonic = pcName(nameToPc(displayName) + semitones, preferFlatsOut);
  const newFifths = fifthsFor(newTonic, mode);
  const play = newTonic;
  const sounding =
    typeof semitones === 'number'
      ? pcName(nameToPc(keyMeta?.sounding || displayName) + semitones, preferFlatsOut)
      : keyMeta?.sounding;
  return {
    ...keyMeta,
    play,
    sounding,
    detected: keyMeta?.detected,
    num_sharps_flats: newFifths,
    mode,
  };
}

/** 小节内拍位置是否落在强拍（节拍器重音） */
export function isStrongBeat(beat) {
  return Math.abs(beat - Math.round(beat)) < 1e-6 && Math.round(beat) % 2 === 0;
}

/* ───────────────────────────── 时间格式化 ───────────────────────────── */

export function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds + 0.5);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function formatTimePrecise(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00.0';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

/* ───────────────────────────── 视图变换 ───────────────────────────── */

/**
 * 根据移调/变调夹参数计算“显示位移”。
 * displayShift = transpose - (capo - baseCapo)，发声音高位移 = transpose。
 */
export function computeDisplayShift(transposeSemis, capo, baseCapo) {
  return (transposeSemis || 0) - ((capo || 0) - (baseCapo || 0));
}

/** 拍号 -> 显示字符串 */
export function timeSignatureLabel(ts) {
  const [num, den] = Array.isArray(ts) && ts.length === 2 ? ts : [4, 4];
  return `${num}/${den}`;
}

/** 段落中文名（优先 score 里的 label） */
export function sectionLabel(section) {
  if (!section) return '';
  if (section.label) return section.label;
  return SECTION_LABELS[section.kind] || '段落';
}
