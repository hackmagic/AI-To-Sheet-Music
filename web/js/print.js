/**
 * print.js —— 打印用分页谱面生成
 *
 * 做法：用 ScoreRenderer 在离屏容器里以 A4 可印宽度（688px ≈ 182mm）重排一次，
 * 再按系统高度把 .system 节点搬进 .print-page 分页容器。屏幕上这些节点是隐藏的
 * （.print-root{display:none}），打印时由 body.is-printing 切换显示。
 */

import { ScoreRenderer } from './renderer.js';

const PAGE_W = 688; // 182mm @96dpi
const PAGE_BUDGET_FIRST = 800;
const PAGE_BUDGET = 900; // 每页可容纳的系统高度（px），保守取值避免溢出
const FIRST_PAGE_EXTRA = 0;

function el(tag, cls = '', text = '') {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

function fmtKey(meta) {
  const key = meta?.key || {};
  const mode = key.mode === 'minor' ? '小调' : '大调';
  return `${key.play || key.sounding || 'C'} ${mode}`;
}

function fmtMeta(meta) {
  const bits = [fmtKey(meta)];
  if (meta?.capo) bits.push(`Capo ${meta.capo}`);
  if (meta?.tempo) bits.push(`♩ = ${Math.round(meta.tempo)}`);
  const ts = meta?.time_signature;
  if (Array.isArray(ts)) bits.push(`${ts[0]}/${ts[1]}`);
  return bits.join(' · ');
}

export class PrintController {
  constructor({ printRoot, host = document.body, getView, getVoicings, getOptions }) {
    this.printRoot = printRoot;
    this.host = host;
    this.getView = getView;
    this.getVoicings = getVoicings;
    this.getOptions = getOptions;
    this.pages = [];
    this.ready = false;
    this.offscreen = null;
    this.renderer = null;
  }

  invalidate() {
    this.ready = false;
    this.pages = [];
    if (this.printRoot) this.printRoot.replaceChildren();
    if (this.offscreen) {
      this.offscreen.remove();
      this.offscreen = null;
    }
    this.renderer?.dispose?.();
    this.renderer = null;
  }

  /** 同步构建分页文档（window.print 前必须同步完成） */
  build() {
    const view = this.getView?.();
    if (!this.printRoot || !view) return false;
    this.invalidate();

    this.offscreen = el('div', 'print-measure');
    this.offscreen.style.cssText = `position:absolute;left:-99999px;top:0;width:${PAGE_W}px;visibility:hidden;`;
    this.host.appendChild(this.offscreen);

    const options = this.getOptions?.() || {};
    this.renderer = new ScoreRenderer(this.offscreen, {
      showLyrics: options.lyrics !== false,
      showChords: options.chords !== false,
      showDiagrams: options.diagrams !== false,
      showTabs: options.tab !== false,
      // 打印版只保留真实转写的 TAB；由节奏型推导的右手指法属于练习提示，不进成谱
      allowDerivedTab: false,
      selectable: false,
      barsPerRow: 4,
      showSectionLabels: true,
      voicings: this.getVoicings?.(),
    });
    this.renderer.render(view, { width: PAGE_W });

    const systems = this.renderer.systems.map((sys) => sys.el);
    const heights = systems.map((node) => Math.max(40, Math.ceil(node.getBoundingClientRect().height || 120)));

    // 分页
    const groups = [];
    let current = { nodes: [], height: 0, budget: PAGE_BUDGET_FIRST + FIRST_PAGE_EXTRA };
    for (let i = 0; i < systems.length; i++) {
      const h = heights[i];
      if (current.nodes.length && current.height + h > current.budget) {
        groups.push(current);
        current = { nodes: [], height: 0, budget: PAGE_BUDGET };
      }
      current.nodes.push(systems[i]);
      current.height += h;
    }
    if (current.nodes.length) groups.push(current);

    const meta = view.meta || {};
    const title = meta.title || meta.source?.original_name || '未命名乐谱';
    const total = groups.length || 1;

    groups.forEach((group, pageIndex) => {
      const page = el('div', 'print-page');
      const head = el('div', 'print-page__head');
      head.append(
        el('div', 'print-page__title', title),
        el('div', 'print-page__meta', fmtMeta(meta))
      );
      const body = el('div', 'print-page__body');
      group.nodes.forEach((node) => body.appendChild(node));
      const foot = el('div', 'print-page__foot');
      foot.append(
        el('span', '', meta.artist ? `${meta.artist} · AI-To-Sheet-Music` : 'AI-To-Sheet-Music 弹唱谱'),
        el('span', '', `第 ${pageIndex + 1} / ${total} 页`)
      );
      page.append(head, body, foot);
      this.printRoot.appendChild(page);
    });

    this.pages = groups;
    this.ready = true;
    return true;
  }

  /** 打印：确保先构建分页文档，再呼出浏览器打印 */
  print() {
    if (!this.getView?.()) return false;
    if (!this.ready) this.build();
    document.body.classList.add('is-printing');
    const cleanup = () => {
      window.removeEventListener('afterprint', cleanup);
    };
    window.addEventListener('afterprint', cleanup);
    // 等一帧，确保打印样式生效
    requestAnimationFrame(() => window.print());
    return true;
  }

  /** 供"打印预览"截图/调试使用：在屏幕上也显示分页文档 */
  preview(on = true) {
    document.body.classList.toggle('is-print-preview', !!on);
    if (on && !this.ready) this.build();
  }
}
