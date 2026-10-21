/**
 * renderer.js —— 把 score.json 渲染成排版乐谱（VexFlow 4 + 自绘 SVG 叠加层）
 *
 * 设计要点
 *  - 每个「系统」（一行 2~4 小节）是一张独立的 VexFlow SVG；和弦符号、歌词、段落标题、
 *    节奏型、指法图、小节高亮、点击热区都是同一张 SVG 里的元素，坐标直接复用，
 *    打印时按系统分页即可，不需要重排。
 *  - 旋律声部按「四分音符拍」组织；dur 支持 "1/3" 这类分数字符串（三连音）。
 *  - 播放高亮按事件 id（如 n-3-0）定位；小节定位按 bar.index。
 *
 * 对外 API
 *   const r = new ScoreRenderer(hostEl, { onBarClick, onNoteClick, showTabs, ... });
 *   r.setVoicingData(json); r.render(score, {width});
 *   r.setPlayhead(barIndex, beat)  r.clearPlayhead()
 *   r.setActiveBar(barIndex|null)  r.scrollBarIntoView(barIndex)
 *   r.getBarInfo(barIndex)  r.dispose()
 */

import {
  beatsToVex,
  splitBeats,
  barBeats,
  durToBeats,
  midiToVexKey,
  vexKeySignature,
  fifthsFor,
  sectionLabel,
  timeSignatureLabel,
} from './music.js';
import { VoicingLibrary, renderDiagram } from './chorddiagram.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** VexFlow 4：全音符 = 16384 ticks */
export const RESOLUTION = 16384; // VexFlow 4 默认分辨：全音符 = 16384 ticks，四分 = 4096

const LAYOUT = {
  padTop: 8,
  padBottom: 10,
  marginLeft: 46,
  marginRight: 20,
  titleH: 24,
  diagramH: 86,
  diagramSize: 44,
  diagramGap: 12,
  chordRowH: 22,
  staveYOffset: 14,
  // VexFlow 4：new Stave(x, y) 的 y 不是首线，五线从 y + 4*spacing(10) = y+40 开始
  staveTopPad: 40,
  staveLineSpan: 40,
  lyricGap: 22, // 歌词基线到最下方一线（或最低下加线）的距离
  tabGap: 10,
  tabHeight: 78, // TabStave 6 线间距 13px → 5×13=65，再留出数字与下伸空间
};

function vex() {
  const V = globalThis.Vex;
  return V ? V.Flow || V : null;
}

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

function clearEl(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function round(n) {
  return Math.round(n * 1000) / 1000;
}

/** 三连音时值（拍数为 1/3 的整数倍且小于 1 拍） */
function isTripletBeat(beats) {
  const x = beats * 3;
  // API 可能输出 0.3333 这类近似值，容差放宽到 2‰ 拍
  return Math.abs(x - Math.round(x)) < 2e-3 && beats < 1 - 1e-6;
}

/** 拍数 -> ticks */
function ticks(beats) {
  return Math.round(beats * RESOLUTION);
}

/** 线性插值：按采样点把 beat 映射为 x */
function xForBeat(samples, beat) {
  if (!samples.length) return 0;
  if (beat <= samples[0].beat) return samples[0].x;
  const last = samples[samples.length - 1];
  if (beat >= last.beat) return last.x;
  for (let i = 0; i < samples.length - 1; i++) {
    const a = samples[i];
    const b = samples[i + 1];
    if (beat >= a.beat && beat <= b.beat) {
      const t = b.beat === a.beat ? 0 : (beat - a.beat) / (b.beat - a.beat);
      return a.x + (b.x - a.x) * t;
    }
  }
  return last.x;
}

export class ScoreRenderer {
  constructor(host, options = {}) {
    this.host = host;
    this.opts = {
      allowDerivedTab: true,
      showLyrics: true,
      showChords: true,
      showDiagrams: true,
      showTabs: true,
      selectable: true,
      showSectionLabels: true,
      onBarClick: null,
      onNoteClick: null,
      voicings: null,
      ...options,
    };
    this.lib = new VoicingLibrary(this.opts.voicings || {});
    this.score = null;
    this.width = 0;
    this.systems = [];
    this.barMap = new Map();
    this.noteMap = new Map();
    this.sectionStartBars = new Map();
    this._clickHandler = (ev) => this._handleClick(ev);
    this.host.addEventListener('click', this._clickHandler);
  }

  setVoicingData(json) {
    this.lib.addAll(json || {});
    return this;
  }

  setOptions(patch = {}) {
    Object.assign(this.opts, patch);
    if (this.score) this.render(this.score, { width: this.width });
    return this;
  }

  dispose() {
    this.host.removeEventListener('click', this._clickHandler);
    clearEl(this.host);
    this.systems = [];
    this.barMap.clear();
    this.noteMap.clear();
  }

  /* ─────────────────────────── 主渲染 ─────────────────────────── */

  render(score, { width } = {}) {
    const V = vex();
    if (!V) throw new Error('VexFlow 未加载：请确认 web/vendor/vexflow.js 已加载');
    if (!score || !Array.isArray(score.bars)) throw new Error('score.json 缺少 bars');

    this.score = score;
    this.width = Math.max(320, Math.floor(width || this.host.clientWidth || 900));
    clearEl(this.host);
    this.systems = [];
    this.barMap.clear();
    this.noteMap.clear();
    this.sectionStartBars.clear();

    const meta = score.meta || {};
    const timeSig = meta.time_signature || [4, 4];
    const keySig = vexKeySignature(meta.key || {});
    const fifths =
      typeof meta.key?.num_sharps_flats === 'number'
        ? meta.key.num_sharps_flats
        : fifthsFor(meta.key?.play || 'C', meta.key?.mode || 'major');
    const perRow = this.opts.barsPerRow || (this.width < 620 ? 2 : this.width < 980 ? 3 : 4);

    (score.sections || []).forEach((s) => {
      if (s.start_bar && !this.sectionStartBars.has(s.start_bar)) this.sectionStartBars.set(s.start_bar, s);
    });

    const groups = this._groupBars(score.bars, perRow);
    const ctxInfo = { V, timeSig, keySig, fifths, isFirst: true };
    groups.forEach((group, idx) => {
      this._renderSystem(group, idx, groups.length, ctxInfo);
      ctxInfo.isFirst = false;
    });

    this.layout = { width: this.width, systems: this.systems };
    return this.layout;
  }

  /** 按 4 小节/行切行；段落边界强制换行，`new_system` / `new_page` 生效 */
  _groupBars(bars, perRow) {
    const groups = [];
    let cur = null;
    for (const bar of bars) {
      let startNew = !cur;
      if (cur) {
        const sectionChanged = String(bar.section) !== String(cur[0].section);
        startNew = startNew || sectionChanged || bar.new_system || bar.new_page;
        // 段尾只剩 1 小节且当前行已排满 2/3，尽量并入避免孤行
        if (!startNew && cur.length >= perRow) startNew = true;
      }
      if (startNew) {
        cur = [];
        groups.push(cur);
      }
      cur.push(bar);
    }
    // 收尾防孤行：最后一行只剩 1 小节时，从上一行挪 1 小节下来凑成 2 小节
    if (groups.length > 1) {
      const last = groups[groups.length - 1];
      const prev = groups[groups.length - 2];
      if (last.length === 1 && prev.length >= 3) last.unshift(prev.pop());
      else if (last.length < perRow) {
        // 上一行还有空间时直接并进去，别让末尾留一行 1-2 小节的碎片
        if (prev.length + last.length <= perRow) {
          prev.push(...last);
          groups.pop();
        }
      }
    }
    return groups;
  }

  _renderSystem(bars, systemIndex, systemCount, ctxInfo) {
    const { V, timeSig, keySig, isFirst } = ctxInfo;
    const W = this.width;
    const meta = this.score.meta || {};
    const beatsPerBar = barBeats(bars[0]?.time_signature || timeSig);
    const firstBar = bars[0];
    const section =
      this.sectionStartBars.get(firstBar.index) ||
      (this.score.sections || []).find((s) => s.start_bar === firstBar.index || s.id === firstBar.section) ||
      null;
    const isSectionStart = !!section && section.start_bar === firstBar.index;
    const titleText = this.opts.showSectionLabels && isSectionStart ? sectionLabel(section) : '';

    const hasLyrics = this.opts.showLyrics && bars.some((b) => (b.vocal || []).some((n) => n.lyric && n.midi != null && !n.rest));
    const hasTabData =
      this.opts.showTabs &&
      bars.some((b) => {
        if (!this.opts.allowDerivedTab && !this._hasExplicitTab(b)) return false;
        return this._tabPositionsFor(b).length > 0;
      });

    // 指法图：本行出现的和弦（按首次出现顺序去重）
    const chordSymbols = [];
    const chordVoicing = new Map();
    for (const bar of bars) {
      for (const c of bar.chords || []) {
        if (!c.symbol) continue;
        if (!chordSymbols.includes(c.symbol)) chordSymbols.push(c.symbol);
        if (!chordVoicing.has(c.symbol)) chordVoicing.set(c.symbol, c.voicing_id || null);
      }
    }
    const showDiagrams = this.opts.showDiagrams && chordSymbols.length > 0;
    const dgCount = Math.min(chordSymbols.length, 9);
    const titleH = titleText ? LAYOUT.titleH : 0;
    const dgH = showDiagrams ? LAYOUT.diagramH : 0;
    const chordH = this.opts.showChords ? LAYOUT.chordRowH : 0;
    // staveY 在这里指「可见的第一条五线」，传给 VexFlow 时再减掉内边距
    const staveY = LAYOUT.padTop + titleH + dgH + chordH + LAYOUT.staveYOffset;
    const staveBottom = staveY + LAYOUT.staveLineSpan;
    const chordY = staveY - 10;
    // 歌词基线：既要落在谱表下方，也要避开谱表以下加线/低音音符。
    // 人声谱里低于谱表的音（如 A3、G3）会带下加线，直接把歌词写在
    // staveBottom 之下仍会压到它们，所以按本行最低音再往下让。
    const lowestVocalMidi = bars.reduce((min, b) => {
      for (const n of b.vocal || []) {
        if (n.midi != null && !n.rest && !(n.tie === 'stop' || n.tie === 'continue')) {
          if (min == null || n.midi < min) min = n.midi;
        }
      }
      return min;
    }, null);
    // 五线谱最下方一线 = E4(64)，相邻「线-间」距离是线距的一半（5px），
    // 低于谱表的音每 2 个半音下移一格，再加半个符头高度和歌词间距。
    const ledgerDepth = lowestVocalMidi != null && lowestVocalMidi < 64
      ? Math.ceil((64 - lowestVocalMidi) / 2) * 5 + 5
      : 0;
    const lyricY = staveBottom + ledgerDepth + LAYOUT.lyricGap;
    // TAB 首线必须让开：①歌词整块（基线 + 下伸部 + 呼吸）②谱表下加线区
    // ③TAB 第一个数字的顶部（数字约在首线上方 8px）。无人声时也要留足，否则 TAB 会像贴在谱表下。
    const lyricBlockBottom = hasLyrics ? lyricY + 14 : 0;
    const tabAnchor = Math.max(lyricBlockBottom + 18, staveBottom + 40);
    const tabTop = hasTabData ? tabAnchor + LAYOUT.tabGap : 0;
    // 歌词行可能因为一行太挤而折成两行（见下方歌词排版），非 TAB 场景要预留第二行高度
    const systemH = hasTabData
      ? tabTop + LAYOUT.tabHeight + LAYOUT.padBottom
      : (hasLyrics ? lyricY + 26 : staveBottom + 18) + LAYOUT.padBottom;

    const wrapper = document.createElement('div');
    wrapper.className = 'system__canvas';
    wrapper.style.width = `${W}px`;
    wrapper.style.height = `${systemH}px`;
    const systemEl = document.createElement('div');
    systemEl.className = 'system';
    if (isSectionStart) systemEl.classList.add('system--section-start');
    systemEl.dataset.system = String(systemIndex);
    systemEl.dataset.firstBar = String(firstBar.index);
    systemEl.appendChild(wrapper);
    // 先入文档再画：SVG 文本宽度（getComputedTextLength）和 CSS 字体都依赖真实排版，
    // 否则和弦避让、歌词定位都会拿到 0 宽度算错。
    this.host.appendChild(systemEl);

    const renderer = new V.Renderer(wrapper, V.Renderer.Backends.SVG);
    renderer.resize(W, systemH);
    const ctx = renderer.getContext();
    const svg = wrapper.querySelector('svg');
    svg.classList.add('system__svg');
    svg.setAttribute('role', 'group');
    svg.setAttribute('aria-label', `第 ${systemIndex + 1} 行：第 ${firstBar.index} 至 ${bars[bars.length - 1].index} 小节`);
    svg.setAttribute('viewBox', `0 0 ${W} ${systemH}`);

    const contentLeft = LAYOUT.marginLeft;
    const contentW = Math.max(120, W - LAYOUT.marginLeft - LAYOUT.marginRight);
    const barW = contentW / bars.length;

    const systemInfo = {
      index: systemIndex,
      el: systemEl,
      svg,
      width: W,
      height: systemH,
      staveY,
      bars: [],
    };

    /* ── 段落标题 ── */
    if (titleText) {
      const t = svgEl('text', {
        x: 2,
        y: LAYOUT.padTop + 15,
        class: 'ov-section',
        'font-size': 13,
        'font-weight': 700,
        'letter-spacing': 1.5,
        fill: 'currentColor',
      }, svg);
      t.textContent = titleText;
      systemInfo.titleEl = t;
    } else if (isFirst) {
      systemInfo.titleEl = null;
    }

    /* ── 指法图 ── */
    if (showDiagrams) {
      const dgTop = LAYOUT.padTop + titleH;
      const dgGap = LAYOUT.diagramSize + LAYOUT.diagramGap;
      chordSymbols.slice(0, dgCount).forEach((symbol, i) => {
        const voicing = this._voicingForSymbol(symbol, chordVoicing.get(symbol));
        if (!voicing) return;
        const g = svgEl('g', {
          class: 'diagram-group',
          transform: `translate(${round(contentLeft + i * dgGap)}, ${round(dgTop)})`,
          'data-symbol': symbol,
        }, svg);
        try {
          const dg = renderDiagram(voicing, { size: LAYOUT.diagramSize });
          g.appendChild(dg);
        } catch (err) {
          console.warn('和弦图渲染失败', symbol, err);
          return;
        }
        const label = svgEl('text', {
          x: LAYOUT.diagramSize / 2,
          y: LAYOUT.diagramSize + 15,
          'text-anchor': 'middle',
          class: 'diagram__label',
          'font-size': 10,
          'font-weight': 600,
          fill: 'currentColor',
        }, g);
        label.textContent = symbol;
      });
      if (chordSymbols.length > dgCount) {
        const more = svgEl('text', {
          x: round(contentLeft + dgCount * dgGap) + 4,
          y: dgTop + LAYOUT.diagramSize - 14,
          class: 'diagram__label',
          'font-size': 10,
          fill: 'currentColor',
          opacity: 0.6,
        }, svg);
        more.textContent = `+${chordSymbols.length - dgCount}`;
      }
    }

    /* ── 谱表 + 音符 ── */
    const isLastBarOfPiece = bars[bars.length - 1].index === (this.score.bars[this.score.bars.length - 1]?.index ?? -1);
    bars.forEach((bar, i) => {
      const x = contentLeft + i * barW;
      const stave = new V.Stave(x, staveY - LAYOUT.staveTopPad, barW);
      if (i === 0) {
        stave.addClef('treble');
        stave.addKeySignature(keySig);
        if (isFirst) stave.addTimeSignature(`${timeSig[0]}/${timeSig[1]}`);
      } else {
        stave.setBegBarType(V.BarlineType.NONE);
      }
      stave.setEndBarType(isLastBarOfPiece && i === bars.length - 1 ? V.BarlineType.END : V.BarlineType.SINGLE);
      stave.setContext(ctx);
      stave.draw();

      const barInfo = {
        index: bar.index,
        system: systemIndex,
        stave,
        x,
        width: barW,
        noteStartX: 0,
        noteEndX: 0,
        samples: [],
        wash: null,
        hit: null,
        chords: [],
      };
      this.barMap.set(bar.index, barInfo);
      systemInfo.bars.push(barInfo);

      this._drawBarContent(V, ctx, svg, bar, barInfo, {
        staveY, stave, barW, x, chordY, chordY0: chordY, chordH, lyricY,
      });
    });

    /* ── TAB 谱表 ── */
    if (hasTabData) {
      const tabStave = new V.TabStave(contentLeft, tabTop, contentW);
      tabStave.setNumLines(6);
      // TabStave 6 线（spacing 13）的首线也不在 y 上，构造后按实际首线校正一次
      try {
        tabStave.setY(tabStave.getY() + (tabTop - tabStave.getYForLine(0)));
      } catch (err) {
        /* 老版本没有 getYForLine 就保持原样 */
      }
      tabStave.setBegBarType(V.BarlineType.NONE);
      tabStave.setEndBarType(V.BarlineType.SINGLE);
      tabStave.setContext(ctx);
      tabStave.draw();
      systemInfo.tabStave = tabStave;
      const derived = bars.some((b) => this._isDerivedTab(b));
      const tabLabel = svgEl('text', {
        x: LAYOUT.marginLeft + 2,
        y: tabTop + 10,
        class: 'ov-tab-label',
        'font-size': 9,
        'letter-spacing': 0.6,
        fill: 'currentColor',
        'data-derived': derived ? '1' : '0',
      }, svg);
      tabLabel.textContent = derived ? 'TAB · 按节奏型推得' : 'TAB';
      tabLabel.setAttribute('opacity', '0.55');
      systemInfo.tabLabel = tabLabel;

      const tabNotesByBar = [];
      bars.forEach((bar, i) => {
        const info = this.barMap.get(bar.index);
        const notes = this._buildTabNotes(V, bar, info, tabStave);
        tabNotesByBar.push(notes);
      });
      // 统一格式化，让 TAB 与主谱表共用宽度
      const allTabNotes = tabNotesByBar.flat();
      if (allTabNotes.length) {
        const voice = new V.Voice({ num_beats: beatsPerBar * bars.length, beat_value: 4 });
        voice.setMode(V.Voice.Mode.SOFT);
        voice.addTickables(allTabNotes);
        const formatter = new V.Formatter();
        try {
          formatter.joinVoices([voice]).format([voice], contentW - 40);
        } catch (err) {
          console.warn('TAB 格式化失败', err);
        }
        try {
          voice.setStave(tabStave);
          allTabNotes.forEach((n) => n.setStave(tabStave));
          voice.draw(ctx, tabStave);
        } catch (err) {
          console.warn('TAB 绘制失败', err);
        }
        try {
          const beams = V.Beam.generateBeams(allTabNotes, { beam_rests: false });
          beams.forEach((b) => b.setContext(ctx).draw());
        } catch (err) {
          /* 连梁失败不影响可读性 */
        }
      }
    }

    /* ── 和弦符号 ── */
    if (this.opts.showChords) {
      // 和弦按 beat 定位，右边界限制在本小节内；再按「上一个和弦右边界」统一避让。
      // 小节最后一拍的和弦会被自然推到小节线内侧（记谱惯例），不会压到下一小节。
      let prevRight = -Infinity;
      bars.forEach((bar) => {
        const info = this.barMap.get(bar.index);
        const samples = info.samples.length ? info.samples : [{ beat: 0, x: info.noteStartX }];
        const barStart = info.x + 4;
        const barEnd = info.x + info.width - 4;
        for (const chord of bar.chords || []) {
          if (!chord.symbol) continue;
          const chordIdx = (bar.chords || []).indexOf(chord);
          const t = svgEl('text', {
            x: 0,
            y: chordY,
            class: 'ov-chord',
            'font-size': 12.5,
            'font-weight': 600,
            fill: 'currentColor',
            'data-bar': bar.index,
            'data-beat': chord.beat ?? 0,
            'data-symbol': chord.symbol,
            'data-chord': chordIdx,
            // 编辑器写入的痕迹标记：CSS 画虚线下划线，导出时剥掉
            'data-edited': chord._edited ? '1' : null,
          }, svg);
          t.textContent = chord.symbol;
          const textLen = chord.symbol.length;
          let width = this._textWidth(t, textLen * 7.4);
          let left = Math.min(Math.max(barStart, xForBeat(samples, chord.beat ?? 0)), Math.max(barStart, barEnd - width));
          const minGap = 6;
          if (left < prevRight + minGap) {
            left = prevRight + minGap;
            if (left + width > barEnd) {
              // 空间实在不够才压字号（12.5 → 11.5 → 11 → 10.5）
              for (const fs of [11.5, 11, 10.5]) {
                t.setAttribute('font-size', String(fs));
                const w2 = this._textWidth(t, textLen * fs * 0.6);
                if (prevRight + minGap + w2 <= barEnd) {
                  width = w2;
                  break;
                }
              }
              left = Math.min(prevRight + minGap, Math.max(barStart, barEnd - width));
            }
          }
          t.setAttribute('x', String(round(left)));
          prevRight = left + width;
          info.chords.push(t);
        }
      });
    }

    /* ── 歌词 ── */
    if (this.opts.showLyrics) {
      // ① 每个字先按音符中心摆好；② 整行按“字距 ≥ 5px”重排；③ 一行实在放不下时先缩字号，
      // 仍放不下就折成两行（前半 / 后半），而不是让字叠在一起。
      const LYRIC_BASE_FONT = 13;
      const LYRIC_MIN_FONT = 10;
      const LYRIC_LINE_H = 17;
      const LYRIC_GAP = 6;
      const avail = Math.max(160, W - LAYOUT.marginLeft - LAYOUT.marginRight - 8);

      const items = [];
      for (const info of systemInfo.bars) {
        for (const l of info.lyrics || []) {
          const center = Math.max(info.x + 4, l.x);
          const t = svgEl('text', {
            x: round(center),
            y: lyricY,
            class: 'ov-lyric',
            'text-anchor': 'middle',
            'font-size': LYRIC_BASE_FONT,
            fill: 'currentColor',
            'data-bar': info.index,
            'data-note': l.id || '',
            'data-lyric': l.id || '',
            'data-note-idx': l.event?.noteIdx ?? '',
            'data-edited': l.note?._edited ? '1' : null,
          }, svg);
          t.textContent = l.text;
          items.push({ node: t, center, bar: info, w: 0 });
          info.lyricEls = info.lyricEls || [];
          info.lyricEls.push(t);
        }
      }
      items.sort((a, b) => a.center - b.center);

      const measureRow = (row) => {
        row.forEach((it) => {
          const fallback = String(it.node.textContent || '').length * LYRIC_BASE_FONT;
          it.w = this._textWidth(it.node, fallback);
        });
      };

      const sysLeft = LAYOUT.marginLeft + 2;
      const sysRight = W - 4;
      const spanAvail = Math.max(160, sysRight - sysLeft);

      // 折行判断：按“首字左缘 → 末字右缘”的*实际跨度*判断，而不是整行宽度和。
      // 后半小节歌词天然挤在行尾时，整行宽度可能远大于 stave 宽度，但跨度才是能否排下的关键。
      let rows = [items];
      if (items.length > 3) {
        measureRow(items);
        const span = (items[items.length - 1].center + items[items.length - 1].w / 2)
          - (items[0].center - items[0].w / 2);
        const needScale = spanAvail / Math.max(1, span);
        if (needScale < 0.82) {
          const mid = Math.ceil(items.length / 2);
          rows = [items.slice(0, mid), items.slice(mid)];
        }
      }

      const placeRow = (row, y) => {
        let fontSize = LYRIC_BASE_FONT;
        const applyFont = (fs) => {
          if (Math.abs(fs - fontSize) < 0.05) return;
          fontSize = fs;
          row.forEach((it) => {
            it.node.setAttribute('font-size', String(fs));
            it.w = this._textWidth(it.node, String(it.node.textContent || '').length * fs);
          });
        };
        const place = () => {
          let prevRight = -Infinity;
          for (const it of row) {
            let left = it.center - it.w / 2;
            if (left < prevRight + LYRIC_GAP) left = prevRight + LYRIC_GAP;
            it.left = left;
            prevRight = Math.max(prevRight, left + it.w);
          }
        };
        const bounds = () => ({
          first: row[0].left,
          last: row[row.length - 1].left + row[row.length - 1].w,
        });
        const shiftIntoBounds = () => {
          const { first, last } = bounds();
          if (last <= sysRight) return;
          // 整行一起左移，而不是把末尾单字往回塞（那正是过去“nownow”叠字的成因）
          const shift = Math.min(last - sysRight, Math.max(0, first - sysLeft));
          if (shift > 0) row.forEach((it) => { it.left -= shift; });
        };

        place();
        shiftIntoBounds();
        if (bounds().last > sysRight) {
          const { first, last } = bounds();
          const width = Math.max(1, last - first);
          const scale = Math.min(1, Math.max(LYRIC_MIN_FONT / LYRIC_BASE_FONT, (sysRight - sysLeft) / width));
          applyFont(Math.max(LYRIC_MIN_FONT, Math.round(fontSize * scale * 10) / 10));
          place();
          shiftIntoBounds();
        }
        row.forEach((it) => {
          it.node.setAttribute('y', String(round(y)));
          it.node.setAttribute('x', String(round(it.left + it.w / 2)));
          it.node.setAttribute('data-shift', Math.abs(it.left + it.w / 2 - it.center) > 0.5 ? '1' : '0');
        });
      };

      rows.forEach((row, rowIndex) => {
        if (!row.length) return;
        placeRow(row, lyricY + rowIndex * LYRIC_LINE_H);
      });
    }

    /* ── 节奏型 / 段落标记 ── */
    const patternId = firstBar.pattern_id || section?.pattern_id;
    const pattern = patternId ? this.score.strum_patterns?.[patternId] : null;
    const patternText = pattern?.display || (pattern ? pattern.name : '');
    if (patternText && (titleText || isFirst)) {
      const label = svgEl('text', {
        x: W - LAYOUT.marginRight,
        y: LAYOUT.padTop + 15,
        'text-anchor': 'end',
        class: 'ov-strum',
        'font-size': 10.5,
        fill: 'currentColor',
      }, svg);
      label.textContent = `节奏型 ${patternText}`;
    }

    /* ── 速度标记（首行 + 速度变化处） ── */
    const tempo = firstBar.tempo || meta.tempo;
    if (isFirst || firstBar.tempo) {
      const t = svgEl('text', {
        x: W - LAYOUT.marginRight,
        y: chordH ? chordY : staveY - 6,
        'text-anchor': 'end',
        class: 'ov-tempo',
        'font-size': 11,
        fill: 'currentColor',
      }, svg);
      t.textContent = `♩ = ${Math.round(tempo || 84)}`;
    }

    /* ── 小节号 ── */
    const bn = svgEl('text', {
      x: LAYOUT.marginLeft - 9,
      y: staveY + 13,
      'text-anchor': 'end',
      class: 'ov-barnum',
      'font-size': 10,
      'font-weight': 600,
      fill: 'currentColor',
      'data-bar': firstBar.index,
      'data-barnum': '1',
    }, svg);
    bn.textContent = String(firstBar.index);

    /* ── 高亮层 / 热区 ── */
    for (const info of systemInfo.bars) {
      const wash = svgEl('rect', {
        x: round(info.x + 1),
        y: round(staveY - 14),
        width: round(info.width - 2),
        height: round(systemH - (staveY - 14) - LAYOUT.padBottom + 4),
        class: 'bar-wash',
        fill: 'transparent',
        'data-bar': info.index,
        // 高亮层同样覆盖文字，编辑模式下一并不吃事件
        'pointer-events': this.opts.editing ? 'none' : null,
      }, svg);
      info.wash = wash;
      const hit = svgEl('rect', {
        x: round(info.x),
        y: round(staveY - 26),
        width: round(info.width),
        height: round(systemH - (staveY - 26) - 2),
        class: 'bar-hit',
        fill: 'transparent',
        'data-bar': info.index,
        // 编辑模式下必须让位：这块透明矩形盖在和弦/歌词/tab 之上，
        // 留着会吃掉对文字的直接点击（表现为"点歌词没反应"）。
        'pointer-events': this.opts.selectable && !this.opts.editing ? 'all' : 'none',
      }, svg);
      info.hit = hit;
    }

    this.systems.push(systemInfo);
  }

  /**
   * 文本实际渲染宽度。
   * 注意：system 还没插进文档时 getComputedTextLength() 返回 0（未排版），
   * 所以先试 SVG 度量，失败再用 canvas measureText（同一字体栈），最后才用估算值。
   */
  _textWidth(node, fallback) {
    try {
      const w = node.getComputedTextLength();
      if (w > 0) return w;
    } catch (err) {
      /* 离屏文档可能没实现 */
    }
    try {
      if (!this._measureCtx) {
        this._measureCtx = document.createElement('canvas').getContext('2d');
      }
      const fontSize = parseFloat(node.getAttribute('font-size') || '12.5') || 12.5;
      const weight = node.getAttribute('font-weight') || '600';
      this._measureCtx.font = `${weight} ${fontSize}px "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif`;
      const w = this._measureCtx.measureText(node.textContent || '').width;
      if (w > 0) return w;
    } catch (err) {
      /* canvas 不可用时退回估算 */
    }
    return fallback;
  }

  /* ─────────────────────── 小节内容 ─────────────────────── */

  _drawBarContent(V, ctx, svg, bar, info, geom) {
    const ts = bar.time_signature || this.score.meta?.time_signature || [4, 4];
    const beatsTotal = barBeats(ts);
    const events = this._vocalEvents(bar, beatsTotal);

    info.lyrics = [];
    info.samples = [];

    if (!events.length) {
      // 空小节：画全休止符
      const rest = new V.StaveNote({
        keys: ['b/4'],
        duration: 'wr',
        clef: 'treble',
        align_center: true,
      });
      rest.setStave(info.stave);
      rest.setContext(ctx);
      try {
        const formatter = new V.Formatter();
        const voice = new V.Voice({ num_beats: Math.round(beatsTotal), beat_value: 4 });
        voice.setMode(V.Voice.Mode.SOFT);
        voice.addTickables([rest]);
        formatter.joinVoices([voice]).formatToStave([voice], info.stave);
        rest.setStave(info.stave);
        rest.draw();
      } catch (err) {
        console.warn('空小节休止符绘制失败', err);
      }
      info.noteStartX = info.stave.getNoteStartX();
      info.noteEndX = info.stave.getNoteEndX();
      info.samples = [
        { beat: 0, x: info.noteStartX },
        { beat: beatsTotal, x: info.noteEndX },
      ];
      return;
    }

    // ── 组织记谱单元（普通音符 / 三连音组 / 休止） ──
    // 给每个事件标注它在 bar.vocal 里的下标：编辑器改歌词要按数组下标定位，
    // 靠 id 字符串反解既脆弱又慢。index 在 transposeScore 之后仍对应同一位置。
    events.forEach((ev, idx) => {
      if (ev && typeof ev === 'object') ev.noteIdx = idx;
    });
    const units = [];
    let i = 0;
    while (i < events.length) {
      const ev = events[i];
      if (!ev.rest && isTripletBeat(ev.beats)) {
        let sum = 0;
        const group = [];
        let j = i;
        while (j < events.length && !events[j].rest && isTripletBeat(events[j].beats) && sum < 1 - 1e-6) {
          group.push(events[j]);
          sum += events[j].beats;
          j++;
        }
        if (Math.abs(sum - 1) < 5e-3) {
          units.push({ type: 'tuplet', events: group });
          i = j;
          continue;
        }
      }
      units.push({ type: 'single', events: [ev] });
      i++;
    }

    const tickables = [];
    const allNotes = [];
    const ties = [];
    const tuplets = [];
    const fifths = this._fifths();

    const makeNote = (ev, pieceBeats, { isRest = false, first = false } = {}) => {
      let spec = beatsToVex(pieceBeats);
      if (!spec) {
        // 兜底：拆到最近的标准时值
        const approx = splitBeats(pieceBeats)[0];
        spec = beatsToVex(approx) || beatsToVex(0.25);
        console.warn(`小节 ${bar.index} 时值无法精确记谱：${pieceBeats} 拍，用 ${spec.duration}`);
      }
      const duration = isRest ? `${spec.duration}r` : spec.duration;
      const keys = isRest || ev.midi == null ? ['b/4'] : [midiToVexKey(ev.midi, fifths)];
      // 带歌词的声部一律符干朝上：朝下的符干会和下方的歌词行重叠，
      // 这也是人声谱的通行记谱惯例（歌词永远写在符干一侧的反方向）。
      const stemDirection = this.opts.showLyrics ? 1 : null;
      const note = new V.StaveNote({
        keys,
        duration,
        clef: 'treble',
        auto_stem: stemDirection == null,
      });
      if (stemDirection != null && !isRest) note.setStemDirection(stemDirection);
      if (ev.id && first) note._aitsmId = ev.id;
      if (ev.confidence != null && ev.confidence < 0.5) note.addClass?.('is-low-confidence');
      note.setStave(info.stave);
      return note;
    };

    for (const unit of units) {
      if (unit.type === 'tuplet') {
        const groupNotes = [];
        for (const ev of unit.events) {
          const natural = ev.beats * 1.5;
          const spec = beatsToVex(natural) || beatsToVex(0.5);
          const note = new V.StaveNote({
            keys: [midiToVexKey(ev.midi, fifths)],
            duration: spec.duration,
            clef: 'treble',
            auto_stem: !this.opts.showLyrics,
          });
          if (this.opts.showLyrics) note.setStemDirection(1);
          if (ev.id) note._aitsmId = ev.id;
          note.setStave(info.stave);
          groupNotes.push(note);
          tickables.push(note);
          allNotes.push(note);
        }
        try {
          tuplets.push(new V.Tuplet(groupNotes, { num_notes: 3, notes_occupied: 2 }));
        } catch (err) {
          console.warn('三连音创建失败', err);
        }
        // 三连音的每个音各自可能带歌词（一字一音），不能只取组内第一个。
        unit.events.forEach((ev, idx) => {
          if (ev.lyric) info.lyrics.push({ id: ev.id, text: ev.lyric, event: ev, note: groupNotes[idx] });
        });
        continue;
      }

      const ev = unit.events[0];
      if (ev.rest) {
        for (const piece of splitBeats(ev.beats)) {
          const note = makeNote(ev, piece, { isRest: true, first: false });
          note.setContext(ctx);
          tickables.push(note);
          allNotes.push(note);
        }
        continue;
      }

      const pieces = splitBeats(ev.beats);
      const pieceNotes = [];
      pieces.forEach((piece, idx) => {
        const note = makeNote(ev, piece, { first: idx === 0 });
        note.setContext(ctx);
        tickables.push(note);
        allNotes.push(note);
        pieceNotes.push(note);
        if (idx > 0) {
          try {
            ties.push(
              new V.StaveTie({
                first_note: pieceNotes[idx - 1],
                last_note: note,
                first_indices: [0],
                last_indices: [0],
              })
            );
          } catch (err) {
            /* ignore */
          }
        }
      });
      if (ev.lyric) info.lyrics.push({ id: ev.id, text: ev.lyric, event: ev, note: pieceNotes[0] });
      if (ev.tie === 'stop' && pieceNotes.length) {
        pieceNotes.forEach((n) => {
          n._aitsmTieStop = true;
        });
      }
    }

    if (!tickables.length) return;

    // ── 格式化 + 绘制 ──
    const totalTicks = tickables.reduce((sum, n) => sum + (n.getTicks()?.value?.() || n.getIntrinsicTicks() || 0), 0);
    const expectTicks = ticks(beatsTotal);
    const voiceMode = Math.abs(totalTicks - expectTicks) < expectTicks * 0.02 ? V.Voice.Mode.STRICT : V.Voice.Mode.SOFT;
    const [num, den] = Array.isArray(ts) ? ts : [4, 4];
    const voice = new V.Voice({ num_beats: num, beat_value: den, resolution: V.RESOLUTION || 16384 });
    voice.setMode(voiceMode);
    voice.addTickables(tickables);

    try {
      const formatter = new V.Formatter();
      formatter.joinVoices([voice]).formatToStave([voice], info.stave);
    } catch (err) {
      console.warn('小节格式化失败，降级为软格式化', bar.index, err);
      try {
        const formatter = new V.Formatter();
        formatter.joinVoices([voice]).format([voice], info.stave.getWidth() - 20, { align_rests: false });
      } catch (err2) {
        console.warn('退回到默认格式化也失败', err2);
      }
    }

    tickables.forEach((n) => n.setStave(info.stave));
    try {
      voice.draw(ctx, info.stave);
    } catch (err) {
      console.warn('声部绘制失败', bar.index, err);
    }
    ties.forEach((tie) => {
      try {
        tie.setContext(ctx).draw();
      } catch (err) {
        /* ignore */
      }
    });

    tuplets.forEach((tuplet) => {
      try {
        tuplet.setContext(ctx).draw();
      } catch (err) {
        console.warn('三连音括号绘制失败', err);
      }
    });

    // 连梁（八分及更短）
    try {
      const beamables = tickables.filter((n) => !n.isRest || n.isRest());
      const beams = V.Beam.generateBeams(beamables, {
        beam_rests: false,
        // 有歌词时连梁也不能反转符干，否则歌词行会被符干压住
        maintain_stem_directions: this.opts.showLyrics,
      });
      beams.forEach((b) => b.setContext(ctx).draw());
    } catch (err) {
      /* 连梁失败可接受 */
    }

    // ── 采样 x（供和弦符号插值） ──
    info.noteStartX = info.stave.getNoteStartX();
    info.noteEndX = info.stave.getNoteEndX();
    const samples = [];
    const beatsSoFar = [];
    let acc = 0;
    for (const n of tickables) {
      const beat = acc / RESOLUTION;
      acc += n.getIntrinsicTicks() || 0;
      const x = typeof n.getNoteHeadBeginX === 'function' ? (n.getNoteHeadBeginX() + n.getNoteHeadEndX()) / 2 : n.getAbsoluteX();
      samples.push({ beat: round(beat), x });
      beatsSoFar.push({ beat, x });
    }
    info.samples = [{ beat: 0, x: info.noteStartX }, ...samples, { beat: beatsTotal, x: info.noteEndX }];
    info.samples.sort((a, b) => a.beat - b.beat);

    // 歌词 x
    for (const l of info.lyrics) {
      const note = l.note;
      if (!note) continue;
      const x = typeof note.getNoteHeadBeginX === 'function'
        ? (note.getNoteHeadBeginX() + note.getNoteHeadEndX()) / 2
        : note.getAbsoluteX();
      l.x = x;
      this.noteMap.set(l.id, { note, bar: bar.index, system: info.system });
    }
    // 记录所有音符（供播放高亮）
    for (const n of tickables) {
      if (!n._aitsmId) continue;
      if (!this.noteMap.has(n._aitsmId)) {
        const x = typeof n.getNoteHeadBeginX === 'function'
          ? (n.getNoteHeadBeginX() + n.getNoteHeadEndX()) / 2
          : n.getAbsoluteX();
        this.noteMap.set(n._aitsmId, { note: n, bar: bar.index, system: info.system, x });
      } else {
        const rec = this.noteMap.get(n._aitsmId);
        rec.note = rec.note || n;
        if (rec.x == null) {
          rec.x = typeof n.getNoteHeadBeginX === 'function'
            ? (n.getNoteHeadBeginX() + n.getNoteHeadEndX()) / 2
            : n.getAbsoluteX();
        }
      }
    }
  }

  /* ─────────────────────── 人声事件 ─────────────────────── */

  _vocalEvents(bar, beatsTotal) {
    const raw = Array.isArray(bar.vocal) ? bar.vocal.slice() : [];
    raw.sort((a, b) => (a.beat || 0) - (b.beat || 0));

    const events = [];
    const seq = raw.filter((n) => n && !n.rest && n.midi != null);
    for (let i = 0; i < seq.length; i++) {
      const n = seq[i];
      if (typeof n.beat !== 'number') continue;
      let beats = durToBeats(n.dur, 0);
      const next = seq[i + 1];
      if (!beats) beats = next ? Math.max(0.125, next.beat - n.beat) : Math.max(0.125, beatsTotal - n.beat);

      const prev = events[events.length - 1];
      const tieJoin = (n.tie === 'stop' || n.tie === 'continue') && prev && prev.midi === n.midi && !n.lyric;
      if (tieJoin) {
        prev.beats = Math.max(prev.beats, n.beat + beats - prev.beat);
        prev.tieStop = true;
        prev.id = prev.id; // 保留首音 id
        continue;
      }
      // 与后一音重叠时裁剪
      if (next && n.beat + beats > next.beat + 1e-6) beats = Math.max(0.125, next.beat - n.beat);
      if (n.beat + beats > beatsTotal + 1e-6) beats = Math.max(0.125, beatsTotal - n.beat);
      if (beats <= 0) continue;
      events.push({
        beat: round(n.beat),
        beats: round(beats),
        midi: n.midi,
        rest: false,
        lyric: n.lyric || null,
        tie: n.tie || null,
        confidence: n.confidence,
        id: `n-${bar.index}-${events.length}`,
        index: i,
      });
    }

    // 补休止，保证小节时值完整
    const filled = [];
    const TOL = 5e-3; // 三连音/浮点误差容忍（1/3 化成 0.333 时会出现毫秒级空隙）
    let pos = 0;
    for (const ev of events) {
      if (ev.beat > pos + TOL) {
        filled.push({ beat: round(pos), beats: round(ev.beat - pos), rest: true, id: null });
      } else if (Math.abs(ev.beat - pos) <= TOL) {
        ev.beat = round(pos);
      }
      filled.push(ev);
      pos = Math.max(pos, ev.beat + ev.beats);
    }
    if (pos < beatsTotal - 1e-6) {
      filled.push({ beat: round(pos), beats: round(beatsTotal - pos), rest: true, id: null });
    }
    return filled;
  }

  /* ─────────────────────── TAB ─────────────────────── */

  _positionsOf(strings) {
    const out = [];
    for (const pair of strings || []) {
      if (!Array.isArray(pair) || pair.length < 2) continue;
      const [str, fret] = pair;
      if (typeof str !== 'number' || typeof fret !== 'number') continue;
      out.push({ str: Math.max(1, Math.min(6, str)), fret: Math.max(0, Math.min(24, fret)) });
    }
    return out;
  }

  /**
   * 一个谱表的 TAB 事件序列：{positions, beats}[]
   *  priority: guitar.hits → guitar.riff → 由和弦把位 + 节奏型推导（弹唱谱的右手指法）
   */
  _tabPositionsFor(bar) {
    const guitar = bar.guitar || {};
    if (Array.isArray(guitar.hits) && guitar.hits.length) {
      return guitar.hits
        .map((hit) => ({ positions: this._positionsOf(hit.strings), beats: durToBeats(hit.dur, 0.5) || 0.5 }))
        .filter((x) => x.positions.length);
    }
    if (Array.isArray(guitar.riff) && guitar.riff.length) {
      return guitar.riff
        .map((r) => {
          let positions = Array.isArray(r.tab) ? this._positionsOf([r.tab]) : [];
          if (!positions.length && typeof r.midi === 'number') {
            const found = this._midiToTab(r.midi);
            if (found) positions = [found];
          }
          return { positions, beats: durToBeats(r.dur, 0.5) || 0.5 };
        })
        .filter((x) => x.positions.length);
    }
    // 没有显式 TAB 数据：用和弦把位 + 节奏型推一份右手指法，避免“开着 TAB 却没有内容”
    const chord = (bar.chords || [])[0];
    const patternId = guitar.pattern_id || bar.pattern_id;
    const pattern = patternId ? this.score.strum_patterns?.[patternId] : null;
    if (!chord || !pattern || !Array.isArray(pattern.events)) return [];
    const voicing = this._voicingForSymbol(chord.symbol, chord.voicing_id);
    if (!voicing || !Array.isArray(voicing.frets)) return [];
    return pattern.events
      .map((ev) => {
        const beats = durToBeats(ev.dur, 0.5) || 0.5;
        const strings = Array.isArray(ev.strings) && ev.strings.length ? ev.strings : [6, 5, 4, 3, 2, 1];
        const positions = [];
        for (const str of strings) {
          const fret = voicing.frets[6 - str];
          if (typeof fret === 'number' && fret >= 0) positions.push({ str, fret });
        }
        return { positions, beats };
      })
      .filter((x) => x.positions.length);
  }

  _hasExplicitTab(bar) {
    const g = bar.guitar || {};
    return (Array.isArray(g.hits) && g.hits.length > 0) || (Array.isArray(g.riff) && g.riff.length > 0);
  }

  _isDerivedTab(bar) {
    const g = bar.guitar || {};
    return !((Array.isArray(g.hits) && g.hits.length) || (Array.isArray(g.riff) && g.riff.length));
  }

  _buildTabNotes(V, bar, info, tabStave) {
    const items = this._tabPositionsFor(bar);
    const notes = [];
    for (const item of items) {
      const spec = beatsToVex(item.beats) || beatsToVex(0.5);
      try {
        const note = new V.TabNote({ positions: item.positions, duration: spec.duration });
        note.setStave(tabStave);
        notes.push(note);
      } catch (err) {
        console.warn('TAB 音符失败', bar.index, err);
      }
    }
    return notes;
  }

  _midiToTab(midi) {
    const tuningMidi = this.score?.meta?.guitar?.tuning || [40, 45, 50, 55, 59, 64];
    for (let s = 5; s >= 0; s--) {
      const fret = midi - tuningMidi[s];
      if (fret >= 0 && fret <= 16) return { str: 6 - s, fret };
    }
    return null;
  }

  /* ─────────────────────── 指法图 ─────────────────────── */

  _fifths() {
    const key = this.score?.meta?.key || {};
    return typeof key.num_sharps_flats === 'number'
      ? key.num_sharps_flats
      : fifthsFor(key.play || 'C', key.mode || 'major');
  }

  _voicingForSymbol(symbol, voicingId) {
    if (voicingId) {
      const v = this.lib.getById(voicingId);
      if (v && v.symbol === symbol) return v;
    }
    return this.lib.select(symbol, { tuning: this.score?.meta?.guitar?.tuning });
  }

  /* ─────────────────────── 交互 / 高亮 ─────────────────────── */

  _handleClick(ev) {
    const target = ev.target.closest?.('[data-bar]');
    if (!target) return;
    const barIndex = Number(target.getAttribute('data-bar'));
    if (!Number.isFinite(barIndex)) return;
    const noteEl = ev.target.closest?.('[data-note]');
    const noteId = noteEl?.getAttribute('data-note') || null;

    // 编辑器的细粒度命中目标：和弦符号 / 歌词 / 小节号
    const chordEl = ev.target.closest?.('[data-chord]');
    if (chordEl) {
      const info = {
        index: Number(chordEl.getAttribute('data-chord')),
        symbol: chordEl.getAttribute('data-symbol') || '',
        beat: Number(chordEl.getAttribute('data-beat') || 0),
        target: chordEl,
      };
      // 编辑器接管时不再触发小节跳转，避免"点和弦却跳走播放位置"
      if (this.opts.onChordClick) {
        this.opts.onChordClick(barIndex, info);
        return;
      }
    }
    const lyricEl = ev.target.closest?.('[data-lyric]');
    if (lyricEl) {
      const info = {
        id: lyricEl.getAttribute('data-lyric'),
        noteId: lyricEl.getAttribute('data-note') || null,
        // 编辑器按数组下标定位音符；渲染器写进 data-note-idx
        noteIdx: (() => {
          const raw = lyricEl.getAttribute('data-note-idx');
          const n = Number(raw);
          return Number.isFinite(n) ? n : NaN;
        })(),
        text: lyricEl.textContent || '',
        target: lyricEl,
      };
      if (this.opts.onLyricClick) {
        this.opts.onLyricClick(barIndex, info);
        return;
      }
    }
    const barNumEl = ev.target.closest?.('[data-barnum]');
    if (barNumEl && this.opts.onBarNumberClick) {
      this.opts.onBarNumberClick(barIndex, { target: barNumEl });
      return;
    }

    this.opts.onBarClick?.(barIndex, { noteId, target });
    if (noteId) this.opts.onNoteClick?.(noteId, barIndex);
  }

  /**
   * barMap 以 score 里的 bar.index（1 起）为键，调用方一律传 0 起的数组下标；
   * 这里做一次兼容查找，两种写法都能命中。
   */
  _barInfo(barIndex) {
    if (barIndex == null) return null;
    return this.barMap.get(barIndex + 1) || this.barMap.get(barIndex) || null;
  }

  getBarInfo(barIndex) {
    return this._barInfo(barIndex);
  }

  setActiveBar(barIndex) {
    const target = barIndex == null ? null : this._barInfo(barIndex);
    for (const [, info] of this.barMap) {
      if (!info.wash) continue;
      const isCurrent = info === target;
      info.wash.classList.toggle('is-current', isCurrent);
      const sysEl = this.systems[info.system]?.el;
      if (sysEl) sysEl.classList.toggle('is-current', isCurrent);
    }
  }

  /** 高亮当前发声音符（按 score 的 note.id 定位到 SVG） */
  setPlayingNote(noteId) {
    if (this._playingNoteId === (noteId || null)) return;
    if (this._playingNoteId) {
      const prev = this.noteMap.get(this._playingNoteId);
      try {
        prev?.note?.getSVGElement?.()?.classList.remove('playing-note');
      } catch {
        /* ignore */
      }
    }
    this._playingNoteId = noteId || null;
    if (!noteId) return;
    const rec = this.noteMap.get(noteId);
    try {
      rec?.note?.getSVGElement?.()?.classList.add('playing-note');
    } catch {
      /* ignore */
    }
  }

  setPlayhead(barIndex, beat = 0) {
    const info = this._barInfo(barIndex);
    if (!info) return;
    this.setActiveBar(barIndex);
    for (const sys of this.systems) {
      if (!sys._playhead) {
        const rect = svgEl('rect', {
          x: -8,
          y: round(sys.staveY - 16),
          width: 16,
          height: round(sys.height - (sys.staveY - 16) - 8),
          class: 'playhead',
          fill: 'rgba(180, 83, 9, 0.10)',
          'pointer-events': 'none',
        }, sys.svg);
        sys._playhead = rect;
        sys._playheadLine = svgEl('line', {
          x1: -1,
          y1: round(sys.staveY - 8),
          x2: -1,
          y2: round(sys.height - 10),
          stroke: 'rgba(180, 83, 9, 0.75)',
          'stroke-width': 1.2,
          'pointer-events': 'none',
          class: 'playhead__line',
        }, sys.svg);
        rect.style.display = 'none';
        sys._playheadLine.style.display = 'none';
      }
    }
    // 显示当前系统的光标，隐藏其它
    for (const sys of this.systems) {
      const show = sys.index === info.system;
      if (sys._playhead) sys._playhead.style.display = show ? '' : 'none';
      if (sys._playheadLine) sys._playheadLine.style.display = show ? '' : 'none';
    }
    const sys = this.systems[info.system];
    if (!sys || !sys._playhead) return;
    const x = Math.max(info.x + 2, xForBeat(info.samples, beat));
    sys._playhead.setAttribute('x', String(round(x - 8)));
    sys._playheadLine.setAttribute('x1', String(round(x)));
    sys._playheadLine.setAttribute('x2', String(round(x)));

    // 和弦高亮：当前 bar 中 beat 之前最近的那个和弦
    for (const [, bi] of this.barMap) {
      const active = bi === info;
      for (const [k, el] of (bi.chords || []).entries()) {
        const elBeat = Number(el.getAttribute('data-beat'));
        const nextBeat = bi.chords[k + 1] ? Number(bi.chords[k + 1].getAttribute('data-beat')) : Infinity;
        const isCurrent = active && elBeat <= beat + 1e-6 && nextBeat > beat;
        el.classList.toggle('is-current', isCurrent);
      }
    }
  }

  clearPlayhead() {
    for (const sys of this.systems) {
      if (sys._playhead) sys._playhead.style.display = 'none';
      if (sys._playheadLine) sys._playheadLine.style.display = 'none';
    }
    this.setActiveBar(null);
    for (const [, info] of this.barMap) {
      for (const el of info.chords || []) el.classList.remove('is-current');
      for (const el of info.lyricEls || []) el.classList.remove('is-current');
    }
  }

  scrollBarIntoView(barIndex, { smooth = true, margin = 120 } = {}) {
    const info = this._barInfo(barIndex);
    if (!info) return;
    const sys = this.systems[info.system];
    if (!sys) return;
    const rect = sys.el.getBoundingClientRect();
    const viewTop = margin;
    const viewBottom = window.innerHeight - margin;
    if (rect.top >= viewTop && rect.bottom <= viewBottom) return;
    const y = window.scrollY + rect.top - Math.max(80, window.innerHeight * 0.18);
    window.scrollTo({ top: y, behavior: smooth ? 'smooth' : 'auto' });
  }
}
