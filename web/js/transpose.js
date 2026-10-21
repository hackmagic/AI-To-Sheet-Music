/**
 * transpose.js —— 视图变换：移调 + 变调夹
 *
 * 语义（与 docs/02-architecture.md 一致）
 *  - 「移调」：发声音高与显示调一起移动 n 个半音；
 *  - 「变调夹」：发声音高不变，显示调与和弦把位向下移动 (capo - baseCapo) 个半音，
 *    也就是用更简单的把位弹同样的音高。
 *  - 显示位移 displayShift = transpose - (capo - baseCapo)；
 *  - 发声音高位移 soundShift   = transpose。
 *
 * 所有函数都返回新的 score 对象，绝不修改传入的原始数据。
 */

import {
  transposeChord,
  transposeKey,
  nameToPc,
  pcName,
  fifthsFor,
  durToBeats,
  barBeats,
} from './music.js';

export const DEFAULT_TUNING = [40, 45, 50, 55, 59, 64]; // 低音 E 到高音 e（midi）

function clone(score) {
  if (typeof structuredClone === 'function') return structuredClone(score);
  return JSON.parse(JSON.stringify(score));
}

/** 为一个音级选升/降拼写：选调号更少的那个 */
export function spellTonic(pc, mode = 'major') {
  const sharpName = pcName(pc, false);
  const flatName = pcName(pc, true);
  if (sharpName === flatName) return sharpName;
  const sFifths = Math.abs(fifthsFor(sharpName, mode));
  const fFifths = Math.abs(fifthsFor(flatName, mode));
  return fFifths < sFifths ? flatName : sharpName;
}

/**
 * 把 [string, fret] 数组按半音移动后重新落指（保持拨弦顺序与弦序）
 * strings: [[str, fret], ...]；str=1 为最细的 1 弦（高音 e）
 */
export function refingerStrings(strings, semitones, tuning = DEFAULT_TUNING) {
  if (!Array.isArray(strings) || !semitones) return strings;
  const out = [];
  for (const pair of strings) {
    if (!Array.isArray(pair) || pair.length < 2) continue;
    const [str, fret] = pair;
    if (typeof str !== 'number' || typeof fret !== 'number' || fret < 0) {
      out.push(pair);
      continue;
    }
    const tuningIndex = Math.max(0, Math.min(5, 6 - str));
    const midi = tuning[tuningIndex] + fret + semitones;
    const sameStringFret = midi - tuning[tuningIndex];
    if (sameStringFret >= 0 && sameStringFret <= 15) {
      out.push([str, sameStringFret]);
      continue;
    }
    // 同一根弦放不下，就近换弦
    let placed = null;
    for (let i = 0; i < tuning.length && !placed; i++) {
      const f = midi - tuning[i];
      if (f >= 0 && f <= 12) {
        const s = 6 - i;
        placed = [s, f];
      }
    }
    out.push(placed || [str, Math.max(0, Math.min(15, sameStringFret))]);
  }
  return out;
}

/**
 * 生成显示用 score。
 * @param {object} score 原始 score.json
 * @param {number} displayShift 显示位移（半音）
 * @param {{ soundingShift?: number, capo?: number, baseCapo?: number, preferFlats?: boolean }} opts
 */
export function transposeScore(score, displayShift = 0, opts = {}) {
  const shift = Math.round(displayShift || 0);
  if (!shift && opts.force !== true) return score;
  const out = clone(score);
  const meta = out.meta || (out.meta = {});
  const tuning = meta.guitar?.tuning || DEFAULT_TUNING;

  const mode = meta.key?.mode || 'major';
  const origTonic = meta.key?.play || meta.key?.sounding || 'C';
  const origSounding = meta.key?.sounding || origTonic;
  // 显示调：按调号最少的拼写；和弦拼写跟随显示调的调号方向（降号调才用降号，
  // 吉他谱里 F#/C# 比 Gb/Db 常见得多）。
  const displayTonic = spellTonic(nameToPc(origTonic) + shift, mode);
  const displayFifths = fifthsFor(displayTonic, mode);
  const preferFlatsOut = opts.preferFlats ?? displayFifths < 0;

  if (meta.key) {
    meta.key = transposeKey(meta.key, shift, { preferFlats: preferFlatsOut });
    meta.key.play = displayTonic;
    meta.key.num_sharps_flats = displayFifths;
    if (typeof opts.soundingShift === 'number') {
      const soundTonic = spellTonic(nameToPc(origSounding) + opts.soundingShift, mode);
      meta.key.sounding = soundTonic;
      meta.key.sounding_fifths = fifthsFor(soundTonic, mode);
    }
  }
  if (typeof opts.capo === 'number') meta.capo = opts.capo;
  meta.display_shift = shift;

  const bars = Array.isArray(out.bars) ? out.bars : [];
  for (const bar of bars) {
    const beatsTotal = barBeats(bar.time_signature || meta.time_signature || [4, 4]);

    if (Array.isArray(bar.vocal)) {
      for (const note of bar.vocal) {
        if (typeof note.midi === 'number') note.midi += shift;
        if (typeof note.midi_sounding === 'number') note.midi_sounding += opts.soundingShift ?? shift;
      }
    }

    if (Array.isArray(bar.chords)) {
      for (const chord of bar.chords) {
        if (!chord.symbol) continue;
        chord.symbol = transposeChord(chord.symbol, shift, preferFlatsOut);
        chord.voicing_id = null;
      }
    }

    const guitar = bar.guitar;
    if (guitar) {
      if (Array.isArray(guitar.hits)) {
        for (const hit of guitar.hits) {
          if (Array.isArray(hit.strings) && hit.strings.length) {
            hit.strings = refingerStrings(hit.strings, shift, tuning);
          }
        }
      }
      if (Array.isArray(guitar.riff)) {
        for (const r of guitar.riff) {
          if (typeof r.midi === 'number') r.midi += shift;
          if (Array.isArray(r.tab)) delete r.tab; // 交给渲染器按 midi 重新落指
        }
      }
    }

    // 段落内和弦与 beat 关系不变；保留 bar.index 等定位信息
    void beatsTotal;
  }

  out.meta = meta;
  return out;
}

/**
 * 供 UI 使用的调性描述，如 "C 大调 · Capo 2"。
 */
export function describeKey(score) {
  const key = score?.meta?.key || {};
  const modeText = key.mode === 'minor' ? '小调' : '大调';
  const tonic = key.play || key.sounding || 'C';
  const capo = score?.meta?.capo || 0;
  return `${tonic} ${modeText}${capo ? ` · Capo ${capo}` : ''}`;
}

/** 依据显示调推荐 capo（0~5）：优先让主和弦落在 C/G/D/A/E 这类开放把位 */
const OPEN_KEYS = ['C', 'G', 'D', 'A', 'E', 'F'];
export function suggestCapo(score, capo = 0) {
  const tonic = score?.meta?.key?.play || 'C';
  const baseCapo = score?.meta?.capo || 0;
  const pc = nameToPc(tonic);
  let best = { capo, offset: Infinity };
  for (let c = 0; c <= 5; c++) {
    const displayPc = pc - (c - baseCapo);
    const name = spellTonic(displayPc, score?.meta?.key?.mode || 'major');
    if (!OPEN_KEYS.includes(name)) continue;
    const offset = Math.abs(c - capo);
    if (offset < best.offset) best = { capo: c, offset };
  }
  return best.capo;
}
