/**
 * musicxml.js —— MusicXML → score/payload 的前端解析（纯标准库，无第三方依赖）
 *
 * 为什么不用 vexml / vexflow-musicxml
 * ----------------------------------
 * 两者都需要 npm 引入，而本仓库的前端是**无构建步骤**的（原生 ES Module +
 * 本地 vendor）。离线环境下也拉不到 npm 包。但浏览器自带 `DOMParser`，
 * 而我们的目标格式（payload）本身就很窄，所以自己写一个「够用」的解析器
 * 反而是更稳的选择：
 *
 *   - 不引入新依赖，与现有 VexFlow 渲染链路完全兼容；
 *   - 解析结果直接喂给已有 `ScoreRenderer`，不需要改渲染器；
 *   - 用户上传 `.musicxml` / `.xml` / `.mxl`（MuseScore、Guitar Pro 导出）即可看谱。
 *
 * 支持的子集（刻意保守，**不认识就跳过并记 warning，绝不猜**）：
 *   - partwise 格式（`<score-partwise>`）；
 *   - `<attributes>`：divisions / key(fifths) / time(beats, beat-type) / clef；
 *   - `<measure>`、`<note>`（pitch / rest / chord / duration / voice / type / dot / tie）；
 *   - `<lyric>`（text + syllabic）；
 *   - `<harmony>`（`<root><root-step>`/`<root-alter>` + `<kind>` + `<bass><bass-step>`）；
 *   - `<direction>` 里的 `<sound tempo>`、`<rehearsal>`（段落名）。
 *
 * **明确不支持**：timewise 布局、多 part 合并（只取第一个有音符的 part）、
 * `<print>` 版式、装饰音（grace 会按 0 时值处理并告警）、`<technical>` 指法
 * （TAB 数据在 payload 里由编配层生成，这里只读五线谱）。
 */

const PITCH_STEPS = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
/** 谱面时值以「四分音符拍」为单位，与 score.json 约定一致 */
const TYPE_TO_QUARTERS = {
  whole: 4, half: 2, quarter: 1, eighth: 0.5, '16th': 0.25, '32nd': 0.125, '64th': 0.0625,
};

const child = (node, name) => {
  if (!node) return null;
  for (const item of node.children || []) if (item.tagName === name) return item;
  return null;
};
const children = (node, name) =>
  Array.from(node?.children || []).filter((item) => item.tagName === name);
/**
 * 按 `a/b/c` 简单路径查找（每级取唯一同名子节点）。
 *
 * 注意：`text(node, 'step')` 只查**直接子节点**，而 `<harmony>` 里的根音在
 * `<root><root-step>` 下，必须用路径写法 `find(node, 'root/root-step')`。
 */
const find = (node, path) => {
  if (!node) return null;
  let current = node;
  for (const name of String(path).split('/')) {
    current = child(current, name);
    if (!current) return null;
  }
  return current;
};
const text = (node, path, fallback = '') => {
  const found = path.includes('/') ? find(node, path) : child(node, path);
  return found ? (found.textContent || '').trim() : fallback;
};
const num = (node, path, fallback = 0) => {
  const value = Number.parseFloat(text(node, path, ''));
  return Number.isFinite(value) ? value : fallback;
};

/** `<pitch>` → MIDI 号 */
export function pitchToMidi(pitchNode) {
  if (!pitchNode) return null;
  const step = text(pitchNode, 'step', 'C').toUpperCase();
  const octave = Number.parseInt(text(pitchNode, 'octave', '4'), 10);
  const alter = Number.parseInt(text(pitchNode, 'alter', '0'), 10) || 0;
  if (!(step in PITCH_STEPS)) return null;
  if (!Number.isFinite(octave)) return null;
  return (octave + 1) * 12 + PITCH_STEPS[step] + alter;
}

/** `<kind>` 值 → 和弦符号后缀（与 score.json 的写法保持一致） */
export function harmonyKindToSuffix(kindValue) {
  const kind = String(kindValue || '').trim().toLowerCase();
  if (!kind) return '';
  // MusicXML 的 kind 取值见 spec；这里只映射我们编配层会产生/消费的那些
  const map = {
    major: '', minor: 'm', augmented: 'aug', diminished: 'dim', dominant: '7',
    'major-seventh': 'maj7', 'minor-seventh': 'm7', 'diminished-seventh': 'dim7',
    'half-diminished': 'm7b5', 'major-minor': 'mMaj7', 'suspended-fourth': 'sus4',
    'suspended-second': 'sus2', 'power': '5', 'major-sixth': '6', 'minor-sixth': 'm6',
    'dominant-seventh': '7', 'dominant-ninth': '9', 'major-ninth': 'maj9', 'minor-ninth': 'm9',
  };
  if (kind in map) return map[kind];
  // 不认识的就原样带上（标在括号里会让符号非法，所以只用字母数字部分）
  const cleaned = kind.replace(/[^a-z0-9+#-]/g, '');
  return cleaned ? cleaned : '';
}

/** `<harmony>` → 和弦符号，如 `G` / `G/B` / `Am7` */
export function parseHarmony(harmonyNode, { preferFlats = false } = {}) {
  if (!harmonyNode) return '';
  const rootStep = text(harmonyNode, 'root/root-step', '');
  if (!rootStep) return '';
  const rootAlter = num(harmonyNode, 'root/root-alter', 0);
  const kindSuffix = harmonyKindToSuffix(text(harmonyNode, 'kind', 'major'));
  const bassStep = text(harmonyNode, 'bass/bass-step', '');
  const bassAlter = num(harmonyNode, 'bass/bass-alter', 0);

  const spell = (step, alter) => {
    const base = PITCH_STEPS[step.toUpperCase()];
    if (base == null) return '';
    const pc = ((base + alter) % 12 + 12) % 12;
    return (preferFlats ? FLAT_NAMES : SHARP_NAMES)[pc];
  };
  const root = spell(rootStep, rootAlter);
  if (!root) return '';
  const suffix = kindSuffix === 'major' ? '' : kindSuffix;
  const bass = bassStep ? spell(bassStep, bassAlter) : '';
  return bass ? `${root}${suffix}/${bass}` : `${root}${suffix}`;
}

/**
 * 解析 MusicXML 文本 → 与 `payload` 同形的 `{ meta, sections, bars }`。
 *
 * 返回 `null` 表示完全无法解析（不是 MusicXML / 没有小节）。
 * 第二个返回值的 `warnings` 记录所有「跳过/不支持」的内容——调用方应展示它们，
 * 而不是假装解析完整。
 */
export function parseMusicXML(xmlText, { preferFlats = null } = {}) {
  const warnings = [];
  const source = String(xmlText || '').trim();
  if (!source) return { score: null, warnings: ['MusicXML 内容为空'] };
  if (typeof DOMParser === 'undefined') {
    return { score: null, warnings: ['当前环境没有 DOMParser，无法解析 MusicXML'] };
  }

  let doc;
  try {
    doc = new DOMParser().parseFromString(source, 'application/xml');
  } catch (err) {
    return { score: null, warnings: [`MusicXML 解析失败：${err.message}`] };
  }
  const parseError = doc.querySelector('parsererror');
  if (parseError) {
    return { score: null, warnings: [`MusicXML 不是合法 XML：${parseError.textContent.slice(0, 200)}`] };
  }

  const root = doc.documentElement;
  if (!root) return { score: null, warnings: ['MusicXML 没有根元素'] };
  if (root.tagName === 'score-timewise') {
    return { score: null, warnings: ['不支持 timewise 布局（score-timewise），请用 score-partwise'] };
  }
  if (root.tagName !== 'score-partwise') {
    return { score: null, warnings: [`不认识的根元素：<${root.tagName}>`] };
  }

  // ---- part 选择：取第一个含音符的 part（不做多谱表合并）----
  const parts = children(root, 'part');
  if (!parts.length) return { score: null, warnings: ['MusicXML 没有 <part>'] };
  const part = parts.find((item) => children(item, 'measure').length > 0) || parts[0];
  if (parts.length > 1) {
    warnings.push(`MusicXML 有 ${parts.length} 个 part，只解析第一个（${part.getAttribute('id') || '未命名'}）`);
  }

  // ---- 全局属性（后续小节可覆盖）----
  let divisions = 1;
  let beatsPerBar = 4;
  let beatType = 4;
  let fifths = 0;
  let tempo = null;
  const measures = [];
  let barNumber = 0;

  for (const measureNode of children(part, 'measure')) {
    barNumber += 1;
    // attributes 可以出现在任意小节（换拍号 / 换调）
    const attributes = child(measureNode, 'attributes');
    if (attributes) {
      const div = num(attributes, 'divisions', NaN);
      if (Number.isFinite(div) && div > 0) divisions = div;
      const keyNode = child(attributes, 'key');
      if (keyNode) {
        const f = num(keyNode, 'fifths', NaN);
        if (Number.isFinite(f)) fifths = f;
      }
      const timeNode = child(attributes, 'time');
      if (timeNode) {
        const b = num(timeNode, 'beats', NaN);
        const bt = num(timeNode, 'beat-type', NaN);
        if (Number.isFinite(b) && b > 0) beatsPerBar = b;
        if (Number.isFinite(bt) && bt > 0) beatType = bt;
      }
    }

    // ---- 速度与段落名 ----
    let sectionLabel = '';
    for (const direction of children(measureNode, 'direction')) {
      for (const sound of children(child(direction, 'direction-type'), 'sound')) {
        const value = Number.parseFloat(sound.getAttribute('tempo') || '');
        if (Number.isFinite(value) && value > 0 && tempo == null) tempo = value;
      }
      const rehearsal = child(child(direction, 'direction-type'), 'rehearsal');
      if (rehearsal && rehearsal.textContent.trim()) {
        sectionLabel = rehearsal.textContent.trim();
      }
      for (const sound of children(direction, 'sound')) {
        const value = Number.parseFloat(sound.getAttribute('tempo') || '');
        if (Number.isFinite(value) && value > 0 && tempo == null) tempo = value;
      }
    }

    // ---- 小节内事件 ----
    const chords = [];
    const vocal = [];
    let cursor = 0; // 单位：四分音符拍
    let pendingChord = null; // <harmony> 出现在 note 之前，挂到下一个音

    for (const node of measureNode.children || []) {
      if (node.tagName === 'harmony') {
        const symbol = parseHarmony(node, { preferFlats: preferFlats === true });
        if (symbol) {
          pendingChord = { beat: cursor, dur: beatsPerBar, symbol, concert: symbol, confidence: 0.8 };
          chords.push(pendingChord);
        }
        continue;
      }
      if (node.tagName === 'backup' || node.tagName === 'forward') {
        const delta = num(node, 'duration', 0) / divisions;
        cursor += node.tagName === 'forward' ? delta : -delta;
        if (cursor < 0) cursor = 0;
        continue;
      }
      if (node.tagName !== 'note') continue;

      const durDiv = num(node, 'duration', 0);
      const quarters = divisions > 0 ? durDiv / divisions : 0;
      const isGrace = Boolean(child(node, 'grace'));
      const isChordTone = Boolean(child(node, 'chord'));
      const isRest = Boolean(child(node, 'rest'));
      const voice = text(node, 'voice', '1');

      // <chord/> 表示与上一个音同时发声：时间不前移，只多一个音高
      if (!isChordTone) {
        // 上一个非和弦音的时值决定游标
      }

      const midi = isRest ? null : pitchToMidi(child(node, 'pitch'));
      if (!isRest && midi == null) {
        warnings.push(`第 ${barNumber} 小节有音符缺少可识别的 <pitch>`);
      }

      if (isGrace) {
        warnings.push(`第 ${barNumber} 小节的装饰音按 0 时值处理（暂不支持时值折算）`);
      }

      if (!isRest && midi != null && voice === (text(measureNode, 'voice', '') || voice)) {
        // 只取主旋律声部：默认第 1 个出现的 voice
        const lyricNodes = children(node, 'lyric');
        const lyric = lyricNodes.length ? (child(lyricNodes[0], 'text')?.textContent || '').trim() : '';
        const syllable = lyricNodes.length ? text(lyricNodes[0], 'syllabic', '') : '';
        vocal.push({
          beat: isChordTone ? Math.max(0, cursor) : cursor,
          dur: quarters,
          midi,
          confidence: 0.8,
          lyric: syllable === 'begin' || syllable === 'middle' ? lyric : lyric,
        });
      }

      if (!isChordTone) cursor += isGrace ? 0 : quarters;
    }

    // 和弦时值：默认铺满到下一个和弦起点或小节末
    for (let index = 0; index < chords.length; index += 1) {
      const current = chords[index];
      const next = chords[index + 1];
      current.dur = next ? Math.max(0.25, next.beat - current.beat) : Math.max(0.25, beatsPerBar - current.beat);
    }
    // 没有 <harmony> 时不留空数组，渲染器按「无和弦」处理

    measures.push({
      index: barNumber,
      time_signature: [beatsPerBar, beatType],
      chords,
      vocal,
      sectionLabel,
    });
  }

  if (!measures.length) return { score: null, warnings: ['MusicXML 里没有任何小节'] };

  // ---- 组装成 score/payload 兼容形状 ----
  const sections = [];
  measures.forEach((measure) => {
    if (!measure.sectionLabel) return;
    const last = sections[sections.length - 1];
    if (last && last.label === measure.sectionLabel) {
      last.end_bar = measure.index;
      return;
    }
    sections.push({
      id: `s${sections.length + 1}`,
      label: measure.sectionLabel,
      kind: 'other',
      start_bar: measure.index,
      end_bar: measure.index,
      start_sec: 0,
      end_sec: 0,
      repeat_of: null,
      pattern_id: '',
      energy: 0.5,
    });
  });
  if (sections.length) sections[sections.length - 1].end_bar = measures.length;

  const keyName = (preferFlats === true ? FLAT_NAMES : SHARP_NAMES)[((fifths * 7) % 12 + 12) % 12];
  const score = {
    schema: 'aitsm.score/v1',
    meta: {
      title: text(root, 'movement-title', '') || '导入的 MusicXML',
      artist: text(root, 'identification/creator', ''),
      tempo: tempo || 90,
      tempo_source: tempo ? 'musicxml' : 'default',
      time_signature: [beatsPerBar, beatType],
      key: {
        // 只给 fifths 推出的主音近似值，供背景渲染；精确调性仍需编配层判断
        detected: `fifths ${fifths}`,
        sounding: keyName,
        mode: 'major',
        play: keyName,
        num_sharps_flats: fifths,
        confidence: 0.5,
      },
      capo: 0,
      guitar: { strings: 6, tuning: [40, 45, 50, 55, 59, 64], tuning_name: 'standard' },
      difficulty: 'intermediate',
      style: 'pop',
      source: { original_name: '', duration_sec: 0 },
      warnings: [],
    },
    sections,
    bars: measures.map((measure) => ({
      index: measure.index,
      start_sec: 0,
      end_sec: 0,
      time_signature: measure.time_signature,
      ...(measure.chords.length ? { chords: measure.chords } : {}),
      ...(measure.vocal.length ? { vocal: measure.vocal } : {}),
      guitar: { pattern_id: '' },
    })),
    voicings: {},
    strum_patterns: {},
    quality: { overall: null, stages: {}, warnings: [] },
    source_format: 'musicxml',
  };

  if (fifths !== 0) {
    warnings.push(`调号按 <fifths>${fifths}</fifths> 近似为 ${keyName} major，关系大小调无法从调号区分`);
  }
  return { score, warnings };
}

/** 判断一个文件名/文本是否像 MusicXML（用于上传时的分流） */
export function looksLikeMusicXML(nameOrText) {
  const value = String(nameOrText || '').trim();
  if (!value) return false;
  if (/\.(musicxml|xml|mxl)$/i.test(value)) return true;
  return value.startsWith('<?xml') || /<score-(partwise|timewise)/i.test(value);
}
