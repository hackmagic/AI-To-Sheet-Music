/**
 * payload.js —— 渲染 payload 的纯数据层
 *
 * 后端在 `payload.json`（`aitsm.payload/v1`）里给出一份「渲染契约」：
 * MusicXML、ABC、绝对时间轴、指法图。前端优先消费这份契约，拿不到时
 * 再退回 `score.json` 自带的字段——两条路径都必须可用。
 *
 * 本模块只做解析与归一化，不碰 DOM、不发请求，可直接被 node 测试。
 */

export const PAYLOAD_SCHEMA = 'aitsm.payload/v1';

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const finite = (value) => {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
};

/**
 * 把后端 payload 归一化成一个稳定形状；结构不对时返回 null（调用方退回 score.json）。
 *
 * @param {unknown} raw 后端 payload.json 解析结果
 * @returns {{
 *   schema: string, title: string, abc: string, musicxml: string,
 *   audioUrl: string, timeline: Array<{index:number,start:number,end:number}>,
 *   diagrams: Map<string, object>, timing: object|null
 * } | null}
 */
export function normalizePayload(raw) {
  if (!isObject(raw)) return null;
  const timing = isObject(raw.timing) ? raw.timing : null;
  const timeline = [];
  if (timing && Array.isArray(timing.bars)) {
    for (const bar of timing.bars) {
      if (!isObject(bar)) continue;
      const index = finite(bar.index);
      const start = finite(bar.start);
      const end = finite(bar.end);
      if (index == null || start == null || end == null || end < start) continue;
      timeline.push({ index: Math.trunc(index), start, end });
    }
    timeline.sort((a, b) => a.index - b.index || a.start - b.start);
  }

  const diagrams = new Map();
  if (Array.isArray(raw.chord_diagrams?.bars)) {
    for (const bar of raw.chord_diagrams.bars) {
      if (!isObject(bar) || !Array.isArray(bar.chords)) continue;
      const index = finite(bar.index);
      if (index == null) continue;
      for (const chord of bar.chords) {
        if (!isObject(chord) || !isObject(chord.diagram)) continue;
        const symbol = typeof chord.symbol === 'string' ? chord.symbol.trim() : '';
        if (!symbol) continue;
        diagrams.set(diagramKey(Math.trunc(index), symbol), chord.diagram);
      }
    }
  }

  return {
    schema: typeof raw.schema === 'string' ? raw.schema : '',
    title: typeof raw.title === 'string' ? raw.title : '',
    abc: typeof raw.abc === 'string' ? raw.abc : '',
    musicxml: typeof raw.musicxml === 'string' ? raw.musicxml : '',
    audioUrl: typeof raw.audio_url === 'string' ? raw.audio_url : '',
    timeline,
    diagrams,
    timing,
  };
}

/** 指法图索引键：小节号 + 和弦符号（与 payload 里 chord_diagrams 的顺序一致） */
export function diagramKey(barIndex, symbol) {
  return `${Number(barIndex)}:${String(symbol || '').trim()}`;
}

/**
 * 用 payload 的绝对时间轴覆盖谱面里的小节起止秒。
 *
 * `score.json` 的 `start_sec` 来自转录层，弱起/变速时有误差；payload 的
 * timing 是渲染层按拍号重新推导并做过单调修正的版本。只覆盖能对上的小节，
 * 绝不改写原对象（返回浅拷贝）。
 *
 * @param {object|null} score
 * @param {object|null} payload 已 normalizePayload 的 payload
 * @returns {object|null}
 */
export function attachTimeline(score, payload) {
  if (!isObject(score) || !Array.isArray(score.bars)) return score;
  const timeline = payload?.timeline;
  if (!Array.isArray(timeline) || timeline.length === 0) return score;

  const byIndex = new Map(timeline.map((item) => [item.index, item]));
  let changed = false;
  const bars = score.bars.map((bar, position) => {
    if (!isObject(bar)) return bar;
    const index = finite(bar.index) ?? position + 1;
    const hit = byIndex.get(Math.trunc(index));
    if (!hit) return bar;
    if (bar.start_sec === hit.start && bar.end_sec === hit.end) return bar;
    changed = true;
    return { ...bar, start_sec: hit.start, end_sec: hit.end };
  });
  if (!changed) return score;
  return { ...score, bars };
}

/**
 * 选一份可用的 ABC：payload 的版本由后端统一生成，优先于 score.json 里的残留字段。
 * @returns {string}
 */
export function pickAbc(payload, score) {
  if (typeof payload?.abc === 'string' && payload.abc.trim()) return payload.abc;
  if (typeof score?.abc === 'string' && score.abc.trim()) return score.abc;
  return '';
}

/** 取某个小节某个和弦的指法图；没有就返回 null（调用方自行降级）。 */
export function diagramFor(payload, barIndex, symbol) {
  if (!payload?.diagrams) return null;
  return payload.diagrams.get(diagramKey(barIndex, symbol)) || null;
}

/** payload 时间轴的总时长（秒）；没有时间轴返回 0。 */
export function timelineDuration(payload) {
  const bars = payload?.timeline;
  if (!Array.isArray(bars) || bars.length === 0) return 0;
  return Math.max(0, bars[bars.length - 1].end || 0);
}
