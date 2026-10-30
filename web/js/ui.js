/**
 * ui.js —— 质量面板、信息条、上传进度、Toast 等纯展示逻辑
 * 不直接发请求，交互通过回调交给 main.js。
 */

import { STAGE_LABELS, formatTime, sectionLabel, SECTION_LABELS, timeSignatureLabel } from './music.js';

const STAGE_ORDER = ['queued', 'ingesting', 'separating', 'analyzing', 'transcribing', 'arranging'];

function el(tag, cls = '', text = '') {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text) node.textContent = text;
  return node;
}

function levelClass(level) {
  if (level === 'error') return 'warning--error';
  if (level === 'warn' || level === 'warning') return 'warning--warn';
  return 'warning--info';
}

function stageName(stage) {
  return STAGE_LABELS[stage] || stage || '处理中';
}

function confidenceTone(value) {
  if (value < 0.5) return 'low';
  if (value < 0.75) return 'mid';
  return 'high';
}

function parseBarTarget(target) {
  if (typeof target === 'number') return target;
  if (typeof target !== 'string') return null;
  const m = target.match(/(\d+)/);
  return m ? Number(m[1]) : null;
}

/* ───────────────────────── Toast ───────────────────────── */

export function toast(message, { kind = 'info', timeout = 3600 } = {}) {
  const host = document.getElementById('toasts');
  if (!host) return;
  const node = el('div', `toast${kind === 'error' ? ' toast--err' : kind === 'ok' ? ' toast--ok' : ''}`, message);
  node.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  host.appendChild(node);
  setTimeout(() => {
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), 260);
  }, timeout);
}

/* ───────────────────────── 上传进度 ───────────────────────── */

export function resetStages() {
  const list = document.getElementById('stages');
  if (!list) return;
  list.querySelectorAll('li').forEach((li) => li.classList.remove('is-active', 'is-done', 'is-failed'));
  const first = list.querySelector('li[data-stage="queued"]');
  first?.classList.add('is-active');
  setUploadMeter(0, '准备中…');
}

export function setUploadMeter(progress, message) {
  const meter = document.getElementById('upload-meter');
  const msg = document.getElementById('upload-msg');
  const pct = Math.max(0, Math.min(1, Number(progress) || 0));
  if (meter) meter.style.width = `${Math.round(pct * 100)}%`;
  if (msg && message) msg.textContent = message;
}

export function updateStages(job = {}) {
  const list = document.getElementById('stages');
  if (!list) return;
  const status = String(job.status || job.state || '').toLowerCase();
  const failed = ['failed', 'error', 'cancelled', 'canceled', 'aborted'].includes(status);
  let stage = String(job.stage || job.phase || '').toLowerCase();
  // 后端可能给 "separating" / "separate" / "stems" 等写法，做一次模糊匹配
  if (!STAGE_ORDER.includes(stage)) {
    if (/stem|separat|人声/.test(stage)) stage = 'separating';
    else if (/beat|tempo|key|chord|analy/.test(stage)) stage = 'analyzing';
    else if (/lyric|melody|transcri|asr/.test(stage)) stage = 'transcribing';
    else if (/arrang|score|render|validat|check|qa|export/.test(stage)) stage = 'arranging';
    else if (/ingest|decode|load|upload/.test(stage)) stage = 'ingesting';
  }
  const currentIndex = STAGE_ORDER.indexOf(stage);
  list.querySelectorAll('li').forEach((li) => {
    const idx = STAGE_ORDER.indexOf(li.dataset.stage);
    li.classList.toggle('is-done', !failed && currentIndex >= 0 && idx < currentIndex);
    li.classList.toggle('is-active', !failed && idx === currentIndex);
    li.classList.toggle('is-failed', failed && idx === currentIndex);
  });
  if (currentIndex >= 0 && !failed) {
    list.querySelectorAll('li').forEach((li) => {
      if (STAGE_ORDER.indexOf(li.dataset.stage) < currentIndex) li.classList.add('is-done');
    });
  }

  // 后端契约用 percent(0-100)，早期接口用 progress(0-1)，两者都兼容
  let progress = typeof job.percent === 'number' ? job.percent : job.progress;
  if (typeof progress === 'number' && progress > 1) progress /= 100;
  const parts = [];
  if (status === 'failed' || failed) {
    parts.push(job.error || job.message || '处理失败');
    setUploadMeter(1, `失败：${parts.join(' · ')}`);
    return;
  }
  if (job.stage_label) parts.push(job.stage_label);
  else if (job.message) parts.push(job.message);
  else if (stage) parts.push(stageName(stage));
  if (typeof progress === 'number') parts.push(`${Math.round(progress * 100)}%`);
  if (status === 'done') {
    list.querySelectorAll('li').forEach((li) => {
      li.classList.remove('is-active', 'is-failed');
      li.classList.add('is-done');
    });
    setUploadMeter(1, '完成');
    return;
  }
  setUploadMeter(progress ?? (currentIndex >= 0 ? (currentIndex + 1) / STAGE_ORDER.length : 0.05), parts.join(' · '));
}

/* ───────────────────────── 和弦图加载占位 ───────────────────────── */

/**
 * 指法库是 577KB，懒加载期间给谱面一个安静的占位提示（null 表示隐藏）。
 */
export function setDiagramStatus(text) {
  const head = document.getElementById('paper-head');
  if (!head) return;
  let node = head.querySelector('.diagram-status');
  if (!text) {
    node?.remove();
    return;
  }
  if (!node) {
    node = document.createElement('p');
    node.className = 'diagram-status';
    node.setAttribute('role', 'status');
    head.appendChild(node);
  }
  node.textContent = text;
}

/* ───────────────────────── 信息条 / 标题 ───────────────────────── */

export function renderDocHeader(score, view = null) {
  const meta = (view || score)?.meta || {};
  const title = meta.title || meta.source?.original_name || '未命名乐谱';
  const artist = meta.artist || '';
  setText('tb-title', title);
  setText('score-title', title);
  setText('score-artist', artist ? `原唱 / 演唱：${artist}` : '');

  const chips = [];
  chips.push(`第 ${score?.bars?.length || 0} 小节`);
  chips.push(`Capo ${meta.capo || 0}`);
  setText('tb-sub', chips.join(' · '));
  return { title, artist };
}

export function renderFactline(score, view = null) {
  const list = document.getElementById('factline');
  if (!list) return;
  const meta = (view || score)?.meta || {};
  const key = meta.key || {};
  const modeText = key.mode === 'minor' ? '小调' : '大调';
  const duration = meta.source?.duration_sec;
  const items = [
    {
      cls: 'fact--key',
      html: `调性 <b>${key.play || key.sounding || 'C'} ${modeText}</b>`,
      title: key.detected ? `检测结果：${key.detected}` : '',
    },
    {
      html: `变调夹 <b>${meta.capo ? `Capo ${meta.capo}` : '不用'}</b>`,
      title: '变调夹只改变显示调与指法，不改变发声音高',
    },
    { html: `速度 <b>♩ = ${Math.round(meta.tempo || 84)}</b>`, title: meta.tempo_source ? `来源：${meta.tempo_source}` : '' },
    { html: `拍号 <b>${timeSignatureLabel(meta.time_signature)}</b>` },
    { html: `小节 <b>${score?.bars?.length || 0}</b>` },
  ];
  if (duration) items.push({ html: `原曲时长 <b>${formatTime(duration)}</b>` });
  if (meta.difficulty) items.push({ html: `难度 <b>${difficultyLabel(meta.difficulty)}</b>` });

  list.replaceChildren(
    ...items.map((item) => {
      const li = el('li', item.cls || '');
      li.innerHTML = item.html;
      if (item.title) li.title = item.title;
      return li;
    })
  );
}

export function updateToolbar(score, view, { transpose = 0, capo = 0 } = {}) {
  const label = document.getElementById('transpose-label');
  if (label) {
    if (!transpose) label.textContent = '原调';
    else label.textContent = `${transpose > 0 ? '+' : ''}${transpose} 半音`;
    const key = (view || score)?.meta?.key;
    label.title = key ? `当前显示调：${key.play || key.sounding || 'C'}` : '相对原调的移调量';
  }
  const select = document.getElementById('capo-select');
  if (select) {
    const base = score?.meta?.capo || 0;
    if (select.options.length === 0) {
      for (let i = 0; i <= 7; i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = i === 0 ? '不用' : `Capo ${i}`;
        select.appendChild(opt);
      }
    }
    select.value = String(capo);
    select.title = base ? `原谱 Capo ${base}` : '变调夹品数';
  }
}

export function updatePaperFoot(score) {
  const foot = document.getElementById('paper-foot-meta');
  if (!foot) return;
  const q = score?.quality || {};
  const warnings = (q.warnings || []).length + (score?.meta?.warnings || []).length;
  const bits = [];
  if (typeof q.overall === 'number') bits.push(`识别置信度 ${Math.round(q.overall * 100)}%`);
  if (warnings) bits.push(`${warnings} 条提示`);
  bits.push(score?.meta?.generator?.version ? `谱面引擎 v${score.meta.generator.version}` : 'AI 扒谱生成');
  foot.textContent = bits.join(' · ');
}

function difficultyLabel(value) {
  return { beginner: '入门', elementary: '基础', intermediate: '进阶', advanced: '高阶' }[value] || value;
}

function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

/* ───────────────────────── 质量面板 ───────────────────────── */

/** 同类告警合并；level 取组内最严重的那个 */
function groupWarnings(warnings) {
  const map = new Map();
  for (const w of warnings) {
    if (!w) continue;
    const code = w.code || '';
    const key = code || String(w.message || '').replace(/第\s*\d+\s*/g, '#');
    let g = map.get(key);
    if (!g) {
      g = { key, code, level: w.level || 'info', items: [] };
      map.set(key, g);
    }
    g.items.push(w);
    if (severity(w.level) > severity(g.level)) g.level = w.level;
  }
  return [...map.values()].sort((a, b) => severity(b.level) - severity(a.level) || b.items.length - a.items.length);
}

function severity(level) {
  if (level === 'error') return 3;
  if (level === 'warn' || level === 'warning') return 2;
  return 1;
}

function isHardWarning(g) {
  return severity(g.level) >= 2;
}

function renderWarningGroup(group, onJumpBar) {
  const first = group.items.find((w) => w.target != null || w.bar != null || w.start_bar != null) || group.items[0];
  const barTarget = parseBarTarget(first?.target ?? first?.bar ?? first?.start_bar);
  const item = el(barTarget != null && onJumpBar ? 'button' : 'div', `warning ${levelClass(group.level)}`);
  if (barTarget != null && onJumpBar) {
    item.type = 'button';
    item.dataset.bar = String(barTarget);
    item.title = `跳转到第 ${barTarget} 小节`;
    item.addEventListener('click', () => onJumpBar(barTarget));
  }
  item.appendChild(el('span', 'warning__dot'));
  const textWrap = el('span', 'warning__text');
  textWrap.append(document.createTextNode(first?.message || group.code || ''));
  if (group.items.length > 1) {
    textWrap.appendChild(el('span', 'warning__count', `（共 ${group.items.length} 处）`));
  }
  if (first?.stage) textWrap.appendChild(el('span', 'warning__stage', stageName(first.stage)));
  item.appendChild(textWrap);
  return item;
}

/* ───────────────────────── 质量面板 ───────────────────────── */

export function renderRail(score, { onJumpBar = null } = {}) {
  const content = document.getElementById('rail-content');
  const empty = document.getElementById('rail-empty');
  if (!content) return;
  if (!score) {
    content.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  content.hidden = false;
  content.replaceChildren();

  const q = score.quality || {};
  const overall = typeof q.overall === 'number' ? Math.max(0, Math.min(1, q.overall)) : null;
  const stages = q.stages || {};
  const warnings = [...(q.warnings || []), ...(score.meta?.warnings || [])];

  /* 总体 */
  const panel = el('section', 'panel');
  const head = el('div', 'panel__head');
  head.append(el('h2', 'panel__title', '识别质量'), el('span', 'panel__aside', overall != null ? `${Math.round(overall * 100)} / 100` : '—'));
  panel.appendChild(head);

  const grade = el('div', 'grade');
  const ringSize = 62;
  const r = 27;
  const c = 2 * Math.PI * r;
  const ring = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  ring.setAttribute('class', 'grade__ring');
  ring.setAttribute('viewBox', `0 0 ${ringSize} ${ringSize}`);
  ring.setAttribute('aria-hidden', 'true');
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  bg.setAttribute('class', 'grade__ring-bg');
  bg.setAttribute('cx', '31');
  bg.setAttribute('cy', '31');
  bg.setAttribute('r', String(r));
  const fg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  fg.setAttribute('class', 'grade__ring-fg');
  fg.setAttribute('cx', '31');
  fg.setAttribute('cy', '31');
  fg.setAttribute('r', String(r));
  fg.setAttribute('stroke-dasharray', c.toFixed(2));
  fg.setAttribute('stroke-dashoffset', (c * (1 - (overall ?? 0))).toFixed(2));
  ring.append(bg, fg);

  const gradeText = el('div');
  gradeText.append(el('div', 'grade__num', overall != null ? String(Math.round(overall * 100)) : '—'), el('div', 'grade__label', '总体置信度'));
  grade.append(ring, gradeText);
  panel.appendChild(grade);

  const note = overall == null ? '尚无质量数据' : qualityNote(overall, stages);
  panel.appendChild(el('p', 'grade__note', note));
  content.appendChild(panel);

  /* 分阶段 */
  const stageKeys = Object.keys(stages || {});
  if (stageKeys.length) {
    const stagePanel = el('section', 'panel');
    const stageHead = el('div', 'panel__head');
    stageHead.append(el('h2', 'panel__title', '分阶段置信度'));
    stagePanel.appendChild(stageHead);
    const meters = el('div', 'meters');
    for (const key of stageKeys) {
      const value = Math.max(0, Math.min(1, Number(stages[key]) || 0));
      const row = el('div', `meter meter--${confidenceTone(value)}`);
      const track = el('div', 'meter__track');
      const fill = el('div', 'meter__fill');
      fill.style.width = `${Math.round(value * 100)}%`;
      track.appendChild(fill);
      row.append(el('span', 'meter__label', stageName(key)), track, el('span', 'meter__value', `${Math.round(value * 100)}%`));
      meters.appendChild(row);
    }
    stagePanel.appendChild(meters);
    content.appendChild(stagePanel);
  }

  /* 告警（同类折叠：相同 code/文案合并成一条，显示“共 N 处”） */
  if (warnings.length) {
    const groups = groupWarnings(warnings);
    const softGroups = groups.filter((g) => !isHardWarning(g));
    const hardGroups = groups.filter(isHardWarning);

    const warnPanel = el('section', 'panel');
    const warnHead = el('div', 'panel__head');
    const asideText = groups.length === warnings.length
      ? `${warnings.length} 条`
      : `${groups.length} 组 · ${warnings.length} 条`;
    warnHead.append(el('h2', 'panel__title', '需要留意'), el('span', 'panel__aside', asideText));
    warnPanel.appendChild(warnHead);
    const list = el('div', 'warnings');
    for (const g of hardGroups) list.appendChild(renderWarningGroup(g, onJumpBar));
    warnPanel.appendChild(list);

    if (softGroups.length) {
      const more = el('details', 'warnings__more');
      const summary = el('summary', 'warnings__summary');
      const softCount = softGroups.reduce((n, g) => n + g.items.length, 0);
      summary.textContent = `另有 ${softCount} 条处理提示（${softGroups.length} 类，默认折叠）`;
      more.appendChild(summary);
      const softList = el('div', 'warnings warnings--soft');
      for (const g of softGroups) softList.appendChild(renderWarningGroup(g, onJumpBar));
      more.appendChild(softList);
      warnPanel.appendChild(more);
    }
    content.appendChild(warnPanel);
  }

  /* 曲式 */
  const sections = score.sections || [];
  if (sections.length) {
    const secPanel = el('section', 'panel');
    const secHead = el('div', 'panel__head');
    secHead.append(el('h2', 'panel__title', '曲式结构'));
    secPanel.appendChild(secHead);
    const list = el('div', 'sections');
    sections.forEach((section, i) => {
      const row = el(onJumpBar ? 'button' : 'div', 'section-row');
      if (onJumpBar) {
        row.type = 'button';
        row.dataset.bar = String(section.start_bar ?? 1);
        row.title = `跳转到第 ${section.start_bar ?? 1} 小节`;
        row.addEventListener('click', () => onJumpBar(section.start_bar ?? 1));
      }
      const label = el('span', 'section-row__label');
      label.append(
        el('span', 'section-row__kind', String(section.kind || '').toUpperCase()),
        document.createTextNode(sectionLabel(section) || SECTION_LABELS[section.kind] || `段落 ${i + 1}`)
      );
      const bars = section.start_bar != null && section.end_bar != null ? `第 ${section.start_bar}–${section.end_bar} 小节` : '';
      row.append(label, el('span', 'section-row__bars', bars));
      list.appendChild(row);
    });
    secPanel.appendChild(list);
    content.appendChild(secPanel);
  }
}

function qualityNote(overall, stages) {
  const entries = Object.entries(stages || {}).sort((a, b) => a[1] - b[1]);
  const weak = entries.filter(([, v]) => Number(v) < 0.6).map(([k]) => stageName(k));
  if (overall >= 0.85) return '识别置信度很高，可以直接照着弹；个别和弦可按听觉习惯微调。';
  if (weak.length) return `${weak.join('、')}置信度偏低，建议对照原曲核对后再使用。`;
  return '整体可用；标黄的部分建议人工确认。';
}

/* ───────────────────────── 播放器 UI ───────────────────────── */

export function renderProgressMarks(sections, totalBars) {
  const host = document.getElementById('progress-marks');
  if (!host) return;
  host.replaceChildren();
  for (const section of sections || []) {
    if (section.start_bar == null || !totalBars) continue;
    const mark = document.createElement('i');
    mark.className = 'is-section';
    mark.style.left = `${((section.start_bar - 1) / totalBars) * 100}%`;
    mark.title = sectionLabel(section);
    host.appendChild(mark);
  }
}

export function setProgress(fraction, { barIndex, elapsed, duration, totalBars }) {
  const fill = document.getElementById('progress-fill');
  if (fill) fill.style.width = `${Math.max(0, Math.min(1, fraction || 0)) * 100}%`;
  const bar = document.getElementById('pos-bar');
  if (bar && barIndex != null) bar.textContent = `第 ${barIndex} 小节`;
  const time = document.getElementById('pos-time');
  if (time && duration != null) time.textContent = `${formatTime(elapsed || 0)} / ${formatTime(duration)}`;
  // role="slider" 的 aria 值必须跟着走，否则读屏播报停在「第 1 小节」永远不变
  const track = document.getElementById('progress-track');
  if (track && barIndex != null) {
    track.setAttribute('aria-valuemin', '1');
    if (totalBars != null) track.setAttribute('aria-valuemax', String(totalBars));
    track.setAttribute('aria-valuenow', String(barIndex));
    track.setAttribute('aria-valuetext', `第 ${barIndex} 小节`);
  }
  void totalBars;
}

/* ───────────────────────── Modal ───────────────────────── */

/**
 * 打开前把焦点来源记下来，关闭时还回去。
 *
 * 实测问题：点「上传」打开弹窗、按 Escape 关闭后，焦点落在 `<body>`（不是触发按钮），
 * 键盘用户的 Tab 位置被清空，得从头再走一遍。
 */
let modalReturnFocus = null;

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]),' +
  ' textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function focusableWithin(node) {
  return [...node.querySelectorAll(FOCUSABLE_SELECTOR)].filter(
    (el) => el.offsetParent !== null || el === document.activeElement
  );
}

export function openModal(node) {
  if (!node) return;
  const trigger = document.activeElement;
  if (trigger instanceof HTMLElement && !node.contains(trigger)) modalReturnFocus = trigger;
  node.hidden = false;
  document.body.classList.add('has-modal');
  const focusable = node.querySelector(FOCUSABLE_SELECTOR);
  focusable?.focus?.({ preventScroll: true });
}

export function closeModal(node) {
  if (!node) return;
  node.hidden = true;
  document.body.classList.remove('has-modal');
  const back = modalReturnFocus;
  modalReturnFocus = null;
  // 触发按钮可能已被移除/隐藏，focus() 失败就安静放弃，不要抛错打断关闭流程
  if (back instanceof HTMLElement && back.isConnected) {
    back.focus?.({ preventScroll: true });
  }
}

/**
 * 模态框内的 Tab 循环 + Escape 关闭。
 *
 * 只做一件事：把 Tab 关在弹窗里。原来 Tab 会跑到工具栏上，
 * 视觉上弹窗还盖着页面，键盘却已经在操作背后的按钮。
 *
 * @returns 是否消费了这次按键（调用方据此决定要不要继续处理）
 */
export function trapModalKey(event, node) {
  if (!node || node.hidden) return false;
  if (event.key === 'Escape') {
    event.preventDefault();
    closeModal(node);
    return true;
  }
  if (event.key !== 'Tab') return false;
  const items = focusableWithin(node);
  if (items.length === 0) return false;
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  // 焦点在弹窗之外（例如刚打开还没落位）时，先收回到第一个
  if (!node.contains(active)) {
    event.preventDefault();
    first.focus?.();
    return true;
  }
  if (event.shiftKey && active === first) {
    event.preventDefault();
    last.focus?.();
    return true;
  }
  if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus?.();
    return true;
  }
  return false;
}
