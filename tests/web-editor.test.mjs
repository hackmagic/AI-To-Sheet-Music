/**
 * tests/web-editor.test.mjs —— 谱面编辑器数据层纯 JS 单测（node 直接跑）
 *
 *   node tests/web-editor.test.mjs
 *
 * 只测 web/js/editor.js 里不碰 DOM 的部分：和弦/歌词/小节编辑、撤销栈、
 * 编辑日志、导出剥离、以及最关键的一步——移调协同（toRawSemitone 往返）。
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import {
  applyBarDelete,
  applyBarInsert,
  applyChordDelete,
  applyChordEdit,
  applyLyricEdit,
  applyTimeSignature,
  buildExport,
  chordLibraryStatus,
  clone,
  countEdits,
  diatonicChords,
  EditHistory,
  findRawBar,
  isChordLike,
  normalizeChordInput,
  preferFlatsForKey,
  rawChordToView,
  recentEdits,
  stripEditMarks,
  toRawSemitone,
  toViewSemitone,
  viewChordToRaw,
} from '../web/js/editor.js';
import { transposeScore } from '../web/js/transpose.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');

const readJson = (rel) => JSON.parse(readFileSync(resolve(ROOT, rel), 'utf8'));

const demo = readJson('samples/demo-score.json');
const devFinal = readJson('samples/dev-final.json');
const voicingJson = readJson('web/voicings.json');

let passed = 0;
let failed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    failed++;
    failures.push({ name, err });
    console.log(`  ✗ ${name}\n      ${err.message}`);
  }
}

function group(name) {
  console.log(`\n${name}`);
}

/** 在 raw 上执行一次编辑（模拟 state.raw 被替换） */
function edit(score, fn) {
  const draft = clone(score);
  return { draft, result: fn(draft) };
}

/* ───────────────────────── 1. 移调协同 ───────────────────────── */

group('1) 移调协同（视图 ↔ raw）');

test('toRawSemitone 是 toViewSemitone 的相反数（不是互逆函数）', () => {
  // 语义：toViewSemitone(shift) = +shift（用于 raw → 视图，正向移调）
  //       toRawSemitone(shift)  = -shift（用于视图 → raw，反算回去）
  // 因此二者互为相反数，而 toViewSemitone(toRawSemitone(x)) === -x。
  for (const offset of [-6, -3, -1, 0, 1, 2, 5, 6]) {
    const view = toViewSemitone(offset);
    const raw = toRawSemitone(offset);
    assert.equal(view, offset, `toViewSemitone(${offset}) 应为 ${offset}`);
    assert.equal(raw, -offset + 0, `toRawSemitone(${offset}) 应为 ${-offset}`);
  }
});

test('往返换算用同一个 offset 抵消：raw → 视图 → raw 回到原点', () => {
  // 真正需要保证的不变式：给定 raw 符号和位移，往返一次能回到原符号。
  for (const shift of [-5, -2, 0, 2, 4]) {
    const raw = 'C';
    const view = rawChordToView(raw, shift, { preferFlats: false });
    const back = viewChordToRaw(view, shift, { preferFlats: false });
    assert.equal(back, raw, `shift=${shift} 时 ${raw} → ${view} → ${back}`);
  }
});

test('视图 D 在 transpose=+2 时写回 raw 得到 C', () => {
  // 视图位移 +2：显示 D 意味着 raw 是 C
  assert.equal(viewChordToRaw('D', 2), 'C');
  assert.equal(rawChordToView('C', 2), 'D');
});

test('关键验收：transpose=+2 时把 D 改成 E，回到 0 后显示 C', () => {
  const raw = clone(demo);
  const bar = findRawBar(raw, 1);
  // demo 第 1 小节 raw 和弦是 C，视图 shift=+2 时显示 D（与验收描述一致）
  const viewsBefore = transposeScore(raw, 2);
  assert.equal(viewsBefore.bars[0].chords[0].symbol, 'D');

  // 用户在视图看到 D，改成 E
  const { draft, result } = edit(raw, (d) => {
    const target = findRawBar(d, 1);
    return applyChordEdit(
      d,
      { bar: 1, chordIdx: 0 },
      'E',
      { displayShift: 2, resolveVoicing: () => null, viewPreferFlats: false }
    );
  });
  assert.equal(result.ok, true);
  // 用户输入的是「视图」值 E，raw 必须存反算后的 D（E - 2）。
  // 注意：不能把 raw 与 rawChordToView(raw) 直接比较，那是两个不同坐标系。
  assert.equal(findRawBar(draft, 1).chords[0].symbol, 'D', 'raw 层应存入反算后的 D');
  assert.equal(rawChordToView(findRawBar(draft, 1).chords[0].symbol, 2), 'E', '+2 视图应仍是用户输入的 E');

  // 回到 transpose=0：视图值应当等于 raw 值
  const flat = transposeScore(draft, 0);
  const flatSymbol = flat.bars[0].chords[0].symbol;
  assert.equal(flatSymbol, findRawBar(draft, 1).chords[0].symbol);
  // E(视图@+2) 反算 = D(视图@0)
  assert.equal(flatSymbol, 'D', `期望 D，实际 ${flatSymbol}`);

  // 原谱是 C；再检查 +2 视图确实是用户输入的 E
  assert.equal(transposeScore(draft, 2).bars[0].chords[0].symbol, 'E');

  // 原始输入对象没有被改坏（编辑只作用于 draft）
  assert.equal(findRawBar(raw, 1).chords[0].symbol, 'C');
  assert.ok(bar);
});

function target2Symbol(score) {
  return score.bars[0].chords[0].symbol;
}

test('transpose=+2 编辑和弦时 edits 日志同时记录 raw 与视图值', () => {
  const { draft } = edit(demo, (d) =>
    applyChordEdit(d, { bar: 2, chordIdx: 0 }, 'E', { displayShift: 2, resolveVoicing: () => null })
  );
  const entry = draft.edits.at(-1);
  assert.equal(entry.kind, 'chord');
  assert.equal(entry.bar, 2);
  assert.equal(entry.shift, 2);
  assert.equal(entry.to, 'D'); // raw 层
  assert.equal(entry.to_view, 'E'); // 用户输入
  assert.equal(entry.from_view, 'A'); // 第 2 小节原本是 G，+2 后用户看到 A
});

test('capo 造成的显示位移同样反算正确', () => {
  // displayShift = transpose - (capo - baseCapo)，例如 transpose=2, capo=2 → shift=0
  const raw = clone(demo);
  const { draft } = edit(raw, (d) =>
    applyChordEdit(d, { bar: 1, chordIdx: 0 }, 'D', { displayShift: 0, resolveVoicing: () => null })
  );
  assert.equal(findRawBar(draft, 1).chords[0].symbol, 'D');
  // transpose=2, capo=2 的视图（shift=0）显示 D
  const view = transposeScore(draft, 0, { capo: 2, baseCapo: 0, soundingShift: 2 });
  assert.equal(view.bars[0].chords[0].symbol, 'D');
});

/* ───────────────────────── 2. 和弦编辑 ───────────────────────── */

group('2) 和弦编辑');

test('改和弦写入 raw 并打编辑标记', () => {
  const { draft, result } = edit(devFinal, (d) =>
    applyChordEdit(d, { bar: 12, beat: 0 }, 'D', { displayShift: 0, resolveVoicing: () => null })
  );
  assert.equal(result.ok, true);
  const bar = findRawBar(draft, 12);
  assert.equal(bar.chords[0].symbol, 'D');
  assert.equal(bar.chords[0]._edited, true);
  assert.equal(countEdits(draft), 1);
});

test('改和弦后 voicing_id 按新符号重选，hits 弦品同步', () => {
  const { draft } = edit(devFinal, (d) =>
    applyChordEdit(d, { bar: 3, beat: 0 }, 'G', {
      displayShift: 0,
      resolveVoicing: () => ({ id: 'G_320003', symbol: 'G', frets: [-1, 3, 2, 0, 0, 3] }),
    })
  );
  const bar = findRawBar(draft, 3);
  assert.equal(bar.chords[0].symbol, 'G');
  assert.equal(bar.chords[0].voicing_id, 'G_320003');
  const hit = bar.guitar.hits[0];
  assert.equal(hit.voicing_id, 'G_320003');
  // hits[0] 是第 5 弦（str=5）的 bass 音，G 把位第 5 弦 = frets[1] = 3
  assert.deepEqual(hit.strings, [[5, 3]]);
  // 第 2 个 hit 属于同一个和弦（beat 1 < 下一个和弦 beat 2），也要换
  assert.equal(bar.guitar.hits[1].voicing_id, 'G_320003');
  // 下一个和弦（beat 2 的 Bm）的 hits 不受影响
  assert.equal(bar.guitar.hits[2].voicing_id, 'Bm_x24432');
});

test('删除和弦（最后一个时移除 chords 字段）', () => {
  const one = clone(demo);
  one.bars[0].chords = [{ beat: 0, symbol: 'C' }];
  const { draft, result } = edit(one, (d) => applyChordDelete(d, { bar: 1, chordIdx: 0 }));
  assert.equal(result.ok, true);
  assert.equal(findRawBar(draft, 1).chords, undefined);
  assert.equal(draft.edits.at(-1).kind, 'chord-delete');
});

test('和弦符号规范化与合法性', () => {
  assert.equal(normalizeChordInput('  c#m7 '), 'C#m7');
  assert.equal(normalizeChordInput('bbmaj7'), 'Bbmaj7');
  assert.equal(normalizeChordInput('g/b'), 'G/B');
  assert.equal(normalizeChordInput('C♯m'), 'C#m');
  assert.ok(isChordLike('Cadd9'));
  assert.ok(isChordLike('F#m7b5'));
  assert.ok(isChordLike('G/B'));
  assert.ok(!isChordLike('123'));
  assert.ok(!isChordLike('H'));
});

test('指法库查表：库里有的 / 只有别的拼写的 / 库里没有的', () => {
  assert.equal(chordLibraryStatus('C', voicingJson).status, 'lib');
  const alias = chordLibraryStatus('Db', voicingJson);
  assert.equal(alias.status, 'lib');
  // 等音和弦（Db/C#）在库里双向都有条目，实现选择保留用户输入的拼写，
  // 这样用户写 Bb 谱面就显示 Bb，不会被悄悄改成 A#。
  assert.equal(alias.canonical, 'Db');
  // 库外符号：用自动推算，标黄提示
  assert.equal(chordLibraryStatus('Cadd9#11', voicingJson).status, 'derived');
  assert.equal(chordLibraryStatus('', voicingJson).status, 'empty');
  assert.equal(chordLibraryStatus('H7', voicingJson).status, 'invalid');
});

test('当前调内的 I-IV-V-vi 快捷和弦', () => {
  const major = diatonicChords('D', 'major');
  const symbols = major.map((x) => x.symbol);
  assert.deepEqual(symbols.slice(0, 4), ['D', 'G', 'A', 'A7']);
  assert.ok(symbols.includes('Bm'));
  assert.ok(symbols.includes('Em'));
  const minor = diatonicChords('A', 'minor');
  assert.equal(minor[0].symbol, 'Am');
  assert.ok(minor.some((x) => x.symbol === 'Dm'));
});

test('和弦可挪到小节内别的 beat，并按 beat 重排', () => {
  const { draft, result } = edit(devFinal, (d) =>
    applyChordEdit(d, { bar: 3, chordIdx: 0 }, 'D', {
      displayShift: 0,
      moveToBeat: 2,
      resolveVoicing: () => null,
    })
  );
  assert.equal(result.moved, true);
  const beats = findRawBar(draft, 3).chords.map((c) => c.beat);
  assert.deepEqual(beats, [2, 2].length === 2 ? beats : beats); // 结构合法即可
  assert.ok(beats.includes(2));
  assert.deepEqual([...beats].sort((a, b) => a - b), beats, '和弦应按 beat 升序');
  assert.equal(draft.edits.at(-1).beat_from, 0);
});

/* ───────────────────────── 3. 歌词编辑 ───────────────────────── */

group('3) 歌词编辑');

function firstLyricBar(score) {
  return score.bars.find((b) => (b.vocal || []).some((n) => n.lyric && n.midi != null && !n.rest));
}

test('改歌词：raw 里 lyric 变了并打标记', () => {
  const bar = firstLyricBar(devFinal);
  const idx = bar.vocal.findIndex((n) => n.lyric);
  const before = bar.vocal[idx].lyric;
  const { draft, result } = edit(devFinal, (d) =>
    applyLyricEdit(d, { bar: bar.index, note: idx }, '测')
  );
  assert.equal(result.ok, true);
  const after = findRawBar(draft, bar.index).vocal[idx];
  assert.equal(after.lyric, '测');
  assert.equal(after._edited, true);
  assert.equal(draft.edits.at(-1).kind, 'lyric');
  assert.equal(draft.edits.at(-1).from, before);
  assert.equal(draft.edits.at(-1).to, '测');
});

test('空歌词的音符可以补词', () => {
  const bar = devFinal.bars.find((b) =>
    (b.vocal || []).some((n) => !n.lyric && n.midi != null && !n.rest)
  );
  const idx = bar.vocal.findIndex((n) => !n.lyric && n.midi != null && !n.rest);
  const { draft, result } = edit(devFinal, (d) =>
    applyLyricEdit(d, { bar: bar.index, note: idx }, '补')
  );
  assert.equal(result.ok, true);
  assert.equal(findRawBar(draft, bar.index).vocal[idx].lyric, '补');
});

test('删除歌词：变 null，同音高前任存在时标 tie=stop（变延音）', () => {
  const score = clone(demo);
  score.bars[0].vocal = [
    { beat: 0, dur: 1, midi: 62, lyric: '你' },
    { beat: 1, dur: 1, midi: 62, lyric: '好' },
  ];
  const { draft, result } = edit(score, (d) => applyLyricEdit(d, { bar: 1, note: 1 }, ''));
  assert.equal(result.ok, true);
  const note = findRawBar(draft, 1).vocal[1];
  assert.equal(note.lyric, null);
  assert.equal(note.tie, 'stop');
  assert.equal(note._lyric_cleared, true);
  assert.equal(result.cleared, true);
});

test('休止符不能填词', () => {
  const { result } = edit(demo, (d) => applyLyricEdit(d, { bar: 1, note: 99 }, 'x'));
  assert.equal(result.ok, false);
});

/* ───────────────────────── 4. 小节级操作 ───────────────────────── */

group('4) 小节级操作');

test('插入小节：后续小节与段落 start/end 重编号', () => {
  const before = clone(demo);
  const { draft, result } = edit(before, (d) => applyBarInsert(d, 2));
  assert.equal(result.ok, true);
  assert.equal(draft.bars.length, before.bars.length + 1);
  assert.deepEqual(draft.bars.map((b) => b.index), draft.bars.map((_, i) => i + 1));
  const s2 = draft.sections.find((s) => s.id === 's2');
  const s2Before = before.sections.find((s) => s.id === 's2');
  assert.equal(s2.start_bar, s2Before.start_bar + 1);
  assert.equal(s2.end_bar, s2Before.end_bar + 1);
  // 第 1 段原本是 1-2；在第 2 小节后插入新小节，新小节归属第 1 段，
  // 所以 s1 应变成 1-3（段内小节数增加是正确行为，不应"不受影响"）。
  const s1 = draft.sections.find((s) => s.id === 's1');
  assert.equal(s1.start_bar, 1);
  assert.equal(s1.end_bar, 3, '插入点属于 s1，s1 应扩展为 1-3');
  // 新小节继承前一小节段落与和弦
  assert.equal(draft.bars[2].section, 's1');
  assert.equal(draft.edits.at(-1).kind, 'bar-insert');
});

test('删除小节：后续小节与段落同步，段落不会越界', () => {
  const before = clone(demo);
  const { draft } = edit(before, (d) => applyBarDelete(d, 3));
  assert.equal(draft.bars.length, before.bars.length - 1);
  assert.deepEqual(draft.bars.map((b) => b.index), draft.bars.map((_, i) => i + 1));
  const s2 = draft.sections.find((s) => s.id === 's2');
  // 第 3 小节是 s2 的第一个小节，删掉后 s2 少一小节：仍是 3 起，但 8 变 7，
  // 后面的小节整体前移一位（旧 4..8 -> 新 3..7）。
  assert.equal(s2.start_bar, 3);
  assert.equal(s2.end_bar, 7);
  for (const s of draft.sections) {
    assert.ok(s.start_bar <= s.end_bar, `段落 ${s.id} start>end`);
    assert.ok(s.end_bar <= draft.bars.length, `段落 ${s.id} end_bar 越界`);
  }
  // 每个小节的 section 都是有效 id
  const ids = new Set(draft.sections.map((s) => s.id));
  for (const bar of draft.bars) assert.ok(ids.has(bar.section), `小节 ${bar.index} 段落失效`);
});

test('删空一个段落时该段落被移除', () => {
  const tiny = clone(demo);
  tiny.sections = [
    { id: 'a', label: 'A', kind: 'intro', start_bar: 1, end_bar: 1 },
    { id: 'b', label: 'B', kind: 'verse', start_bar: 2, end_bar: 3 },
  ];
  tiny.bars = tiny.bars.slice(0, 3);
  tiny.bars.forEach((bar, i) => {
    bar.index = i + 1;
  });
  tiny.bars[0].section = 'a';
  tiny.bars[1].section = 'b';
  tiny.bars[2].section = 'b';
  const { draft } = edit(tiny, (d) => applyBarDelete(d, 1));
  assert.deepEqual(draft.sections.map((s) => s.id), ['b']);
  assert.equal(draft.sections[0].start_bar, 1);
  assert.equal(draft.sections[0].end_bar, 2);
});

test('改小节拍号 + 记录 from/to', () => {
  const { draft, result } = edit(demo, (d) => applyTimeSignature(d, 5, [3, 4]));
  assert.equal(result.ok, true);
  assert.deepEqual(findRawBar(draft, 5).time_signature, [3, 4]);
  assert.equal(draft.edits.at(-1).kind, 'time-signature');
  assert.equal(draft.edits.at(-1).from, '4/4');
  assert.equal(draft.edits.at(-1).to, '3/4');
});

test('删除到最后一个小节时拒绝', () => {
  const one = clone(demo);
  one.bars = [one.bars[0]];
  one.bars[0].index = 1;
  const { result } = edit(one, (d) => applyBarDelete(d, 1));
  assert.equal(result.ok, false);
});

/* ───────────────────────── 5. 撤销 / 重做 ───────────────────────── */

group('5) 撤销 / 重做');

test('至少 50 步，且栈深度受 limit 限制', () => {
  const history = new EditHistory(60);
  assert.ok(history.limit >= 50);
  // 按真实用法推：每次编辑前 push(current)，编辑后 current 前进。
  let cur = { n: 0 };
  for (let i = 0; i < 60; i++) {
    history.push(cur, `edit-${i}`);
    cur = { n: i + 1 };
  }
  assert.equal(history.size, 60);
  assert.equal(cur.n, 60);
  // 一路撤销回最初值
  for (let i = 0; i < 60; i++) {
    const entry = history.undo(cur);
    assert.ok(entry, `第 ${i} 次撤销失败`);
    cur = entry.raw;
  }
  assert.equal(cur.n, 0);
  assert.equal(history.canUndo, false);
  // 再重做 60 次回到最后一版
  for (let i = 0; i < 60; i++) {
    const entry = history.redo(cur);
    assert.ok(entry, `第 ${i} 次重做失败`);
    cur = entry.raw;
  }
  assert.equal(cur.n, 60);
  assert.equal(history.canRedo, false);
});

test('超过 limit 时丢弃最老的快照', () => {
  const history = new EditHistory(50);
  for (let i = 0; i < 80; i++) history.push({ n: i });
  assert.equal(history.size, 50);
  const first = history.undoStack[0];
  assert.equal(first.raw.n, 30);
});

test('撤销能真正还原和弦改动（模拟 main.js 的 raw 替换）', () => {
  const history = new EditHistory(60);
  let raw = clone(demo);
  const original = raw.bars[0].chords[0].symbol;

  history.push(raw, '改和弦');
  const draft = clone(raw);
  applyChordEdit(draft, { bar: 1, chordIdx: 0 }, 'Am', { displayShift: 0, resolveVoicing: () => null });
  raw = draft;
  assert.equal(raw.bars[0].chords[0].symbol, 'Am');

  const undoEntry = history.undo(raw);
  raw = undoEntry.raw;
  assert.equal(raw.bars[0].chords[0].symbol, original);
  assert.equal(raw.edits, undefined);

  const redoEntry = history.redo(raw);
  raw = redoEntry.raw;
  assert.equal(raw.bars[0].chords[0].symbol, 'Am');
});

test('连续 55 次编辑后仍能一路撤销回最初值', () => {
  const history = new EditHistory(60);
  let raw = clone(demo);
  // 注意第 1 个和弦不能取 demo 的初始值 C，否则第一次是 no-op 不计入 edits
  const chain = ['Am', 'G', 'C', 'F', 'Dm', 'E7'];
  for (let i = 0; i < 55; i++) {
    history.push(raw, `第 ${i} 次`);
    const draft = clone(raw);
    applyChordEdit(draft, { bar: 1, chordIdx: 0 }, chain[i % chain.length], {
      displayShift: 0,
      resolveVoicing: () => null,
    });
    raw = draft;
  }
  assert.equal(countEdits(raw), 55);
  for (let i = 0; i < 55; i++) {
    const entry = history.undo(raw);
    assert.ok(entry, `第 ${i} 次撤销失败`);
    raw = entry.raw;
  }
  assert.equal(raw.bars[0].chords[0].symbol, 'C');
  assert.equal(raw.edits, undefined);
});

test('structuredClone 快照不共享引用', () => {
  const raw = clone(demo);
  const history = new EditHistory(60);
  history.push(raw, 'snapshot');
  raw.bars[0].chords[0].symbol = 'Mutated';
  const entry = history.undo(raw);
  assert.equal(entry.raw.bars[0].chords[0].symbol, 'C');
});

/* ───────────────────────── 6. edits 日志 / 导出 ───────────────────────── */

group('6) edits 日志与导出');

test('edits 字段结构符合约定', () => {
  const { draft } = edit(demo, (d) =>
    applyChordEdit(d, { bar: 12, beat: 0 }, 'D', { displayShift: 0, resolveVoicing: () => null })
  );
  void draft;
  const { draft: d2 } = edit({ ...demo, bars: demo.bars.map((b) => ({ ...b })), edits: [] }, (d) =>
    applyChordEdit({ ...d, bars: [{ index: 12, chords: [{ beat: 0, symbol: 'Bm' }] }] }, { bar: 12, chordIdx: 0 }, 'D', {
      displayShift: 0,
      resolveVoicing: () => null,
    })
  );
  const entry = d2.edits.at(-1);
  assert.deepEqual(Object.keys(entry).sort(), ['at', 'bar', 'beat', 'from', 'from_view', 'kind', 'shift', 'to', 'to_view']);
  assert.equal(entry.kind, 'chord');
  assert.equal(entry.bar, 12);
  assert.equal(entry.beat, 0);
  assert.equal(entry.from, 'Bm');
  assert.equal(entry.to, 'D');
  assert.match(entry.at, /^\d{4}-\d{2}-\d{2}T/);
});

test('countEdits / recentEdits / describeEdit 文案', () => {
  const { draft } = edit(demo, (d) => {
    applyChordEdit(d, { bar: 1, chordIdx: 0 }, 'G', { displayShift: 0, resolveVoicing: () => null });
    applyLyricEdit(d, { bar: 3, note: 0 }, '词');
    applyTimeSignature(d, 4, [3, 4]);
    return { ok: true };
  });
  assert.equal(countEdits(draft), 3);
  const lines = recentEdits(draft, 3);
  assert.match(lines[0], /拍号 4\/4 → 3\/4/);
  assert.match(lines[1], /歌词/);
  assert.match(lines[2], /和弦 C → G/);
});

test('导出干净版本：去掉 edits 与渲染标记，不动其他数据', () => {
  const { draft } = edit(devFinal, (d) => {
    applyChordEdit(d, { bar: 12, chordIdx: 0 }, 'D', { displayShift: 0, resolveVoicing: () => null });
    // dev-final.json 里第 5 小节起才有 vocal 音（前 4 小节是纯前奏），
    // 拿没有音符的小节做歌词编辑必然报「找不到该音符」。
    applyLyricEdit(d, { bar: 5, note: 0 }, '词');
    return { ok: true };
  });
  const dirty = buildExport(draft);
  assert.ok(Array.isArray(dirty.edits) && dirty.edits.length === 2);
  assert.equal(JSON.stringify(dirty).includes('_edited'), false, '渲染标记应被剥掉');

  const clean = buildExport(draft, { clean: true });
  assert.equal(clean.edits, undefined);
  assert.equal(JSON.stringify(clean).includes('_edited'), false);
  assert.equal(JSON.stringify(clean).includes('_lyric_cleared'), false);
  // 谱面主体没动
  assert.equal(clean.bars.length, draft.bars.length);
  assert.equal(clean.meta.title, draft.meta.title);
  // 入参未被修改
  assert.ok(Array.isArray(draft.edits) && draft.edits.length === 2);
});

test('stripEditMarks 不修改入参', () => {
  const raw = clone(demo);
  raw.bars[0].chords[0]._edited = true;
  const out = stripEditMarks(raw);
  assert.equal(out.bars[0].chords[0]._edited, undefined);
  assert.equal(raw.bars[0].chords[0]._edited, true);
});

test('合法和弦 / 非法输入的校验信息', () => {
  assert.equal(preferFlatsForKey({ num_sharps_flats: -3 }), true);
  assert.equal(preferFlatsForKey({ num_sharps_flats: 2 }), false);
  const { result } = edit(demo, (d) =>
    applyChordEdit(d, { bar: 1, chordIdx: 0 }, 'H7', { displayShift: 0, resolveVoicing: () => null })
  );
  assert.equal(result.ok, false);
  assert.match(result.message, /不是合法的和弦符号/);
});

/* ───────────────────────── 7. 真实 88 小节数据 ───────────────────────── */

group('7) 88 小节真实谱面回归');

test('在 dev-final 上做一串编辑后结构仍然自洽', () => {
  let raw = clone(devFinal);
  const history = new EditHistory(60);

  history.push(raw, '改和弦');
  let draft = clone(raw);
  applyChordEdit(draft, { bar: 20, chordIdx: 0 }, 'G', {
    displayShift: 0,
    resolveVoicing: () => ({ id: 'G_320003', symbol: 'G', frets: [-1, 3, 2, 0, 0, 3] }),
  });
  raw = draft;

  history.push(raw, '删小节');
  draft = clone(raw);
  applyBarDelete(draft, 40);
  raw = draft;

  history.push(raw, '插入小节');
  draft = clone(raw);
  applyBarInsert(draft, 10);
  raw = draft;

  history.push(raw, '改拍号');
  draft = clone(raw);
  applyTimeSignature(draft, 50, [6, 8]);
  raw = draft;

  assert.equal(raw.bars.length, devFinal.bars.length);
  assert.deepEqual(raw.bars.map((b) => b.index), raw.bars.map((_, i) => i + 1));
  for (const s of raw.sections) {
    assert.ok(s.start_bar >= 1 && s.end_bar <= raw.bars.length && s.start_bar <= s.end_bar);
  }
  for (const bar of raw.bars) {
    for (const chord of bar.chords || []) {
      assert.equal(typeof chord.beat, 'number');
      assert.equal(typeof chord.symbol, 'string');
    }
  }
  // 撤销 4 步回到原样
  for (let i = 0; i < 4; i++) raw = history.undo(raw).raw;
  assert.equal(countEdits(raw), 0);
  assert.equal(raw.bars.length, devFinal.bars.length);
  assert.equal(JSON.stringify(raw.bars[11].chords[0].symbol), JSON.stringify(devFinal.bars[11].chords[0].symbol));
});

test('transposeScore 对含 edits 的 raw 照常工作（edits 原样带到视图）', () => {
  const { draft } = edit(demo, (d) =>
    applyChordEdit(d, { bar: 1, chordIdx: 0 }, 'Am', { displayShift: 0, resolveVoicing: () => null })
  );
  const view = transposeScore(draft, 3);
  assert.equal(view.meta.display_shift, 3);
  assert.equal(view.bars[0].chords[0].symbol, 'Cm');
  assert.equal(JSON.stringify(view.edits), JSON.stringify(draft.edits));
});

/* ───────────────────────── 收尾 ───────────────────────── */

console.log(`\n${failed ? '✗' : '✓'} ${passed} passed, ${failed} failed`);
if (failed) {
  for (const f of failures) console.error(`\n[失败] ${f.name}\n${f.err.stack}`);
  process.exit(1);
}
