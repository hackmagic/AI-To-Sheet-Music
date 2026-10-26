/**
 * tests/web-musicxml.test.mjs —— MusicXML 前端解析的纯逻辑单测
 *
 *   node tests/web-musicxml.test.mjs
 *
 * `parseMusicXML` 依赖浏览器 `DOMParser`。Node 没有，所以测试里注入一个
 * **最小 XML 解析替身**（够覆盖本模块用到的 tagName / children / textContent /
 * getAttribute / querySelector），这样能真正验证解析逻辑而不是跳过。
 *
 * 目的（评审提示 3）：验证「前端可以直接解析 MusicXML 并渲染」，减少对自定义
 * payload 格式的锁定；同时确认不支持的内容会**如实告警**而不是静默吞掉。
 */

import assert from 'node:assert/strict';

/* ───────────────────── 最小 DOMParser 替身 ───────────────────── */

function parseXMLToNode(xml) {
  // 极简标签扫描：只处理 <tag attr="v">text</tag> 与自闭合
  const tokenRe = /<(\/?)([a-zA-Z0-9:_-]+)((?:\s+[^>]*?)?)(\/?)>|([^<]+)/g;
  const root = { tagName: '#document', children: [], textContent: '', parent: null };
  const stack = [root];
  let match;
  while ((match = tokenRe.exec(xml)) !== null) {
    const [, closing, name, attrs, selfClose, plain] = match;
    if (plain != null) {
      const target = stack[stack.length - 1];
      target.textContent += plain;
      continue;
    }
    if (closing) {
      if (stack.length > 1) stack.pop();
      continue;
    }
    const attrsMap = {};
    const attrRe = /([a-zA-Z0-9:_-]+)="([^"]*)"/g;
    let attrMatch;
    while ((attrMatch = attrRe.exec(attrs || '')) !== null) attrsMap[attrMatch[1]] = attrMatch[2];
    const node = {
      tagName: name,
      children: [],
      attrs: attrsMap,
      textContent: '',
      parent: stack[stack.length - 1],
      getAttribute: (key) => (key in attrsMap ? attrsMap[key] : null),
    };
    node.parent.children.push(node);
    if (!selfClose) stack.push(node);
  }
  return root;
}

class FakeDOMParser {
  parseFromString(xml) {
    const doc = parseXMLToNode(xml);
    doc.documentElement = doc.children[0] || null;
    doc.querySelector = (selector) => {
      if (selector === 'parsererror') {
        return doc.children.some((n) => n.tagName === 'parsererror') ? { textContent: 'bad' } : null;
      }
      return null;
    };
    return doc;
  }
}

globalThis.DOMParser = FakeDOMParser;

const { parseMusicXML, parseHarmony, pitchToMidi, harmonyKindToSuffix, looksLikeMusicXML } =
  await import('../web/js/musicxml.js');

/* ───────────────────── 测试框架 ───────────────────── */

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
    console.error(`  ✗ ${name}\n    ${err.message}`);
  }
}

/* ───────────────────── 基础解析 ───────────────────── */

const SAMPLE = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <movement-title>测试曲</movement-title>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>2</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <direction><direction-type><metronome><beat-unit>quarter</beat-unit></metronome></direction-type><sound tempo="120"/></direction>
      <harmony><root><root-step>C</root-step></root><kind>major</kind></harmony>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
      <harmony><root><root-step>G</root-step></root><kind>dominant</kind></harmony>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice></note>
    </measure>
  </part>
</score-partwise>`;

test('合法 MusicXML 能解析出 score 形状', () => {
  const { score, warnings } = parseMusicXML(SAMPLE);
  assert.ok(score, `应当解析成功，warnings=${warnings.join('|')}`);
  assert.equal(score.schema, 'aitsm.score/v1');
  assert.equal(score.source_format, 'musicxml');
  assert.equal(score.meta.title, '测试曲');
  assert.equal(score.meta.tempo, 120);
  assert.deepEqual(score.meta.time_signature, [4, 4]);
  assert.equal(score.bars.length, 1);
});

test('小节内音符按 divisions 换算成四分音符拍', () => {
  const { score } = parseMusicXML(SAMPLE);
  const vocal = score.bars[0].vocal;
  assert.equal(vocal.length, 3, '应当有 3 个旋律音');
  assert.equal(vocal[0].midi, 60); // C4
  assert.equal(vocal[0].duration ?? vocal[0].dur, 1); // 2 个 division / 2 = 1 拍
  assert.equal(vocal[2].dur, 2); // 4 / 2 = 2 拍
});

test('和弦符号从 <harmony> 还原（含 dominant → 7）', () => {
  const { score } = parseMusicXML(SAMPLE);
  const symbols = score.bars[0].chords.map((c) => c.symbol);
  assert.deepEqual(symbols, ['C', 'G7']);
});

test('小节时值不回填、先出现的和弦铺满到下一和弦', () => {
  const { score } = parseMusicXML(SAMPLE);
  const chords = score.bars[0].chords;
  assert.equal(chords[0].dur, chords[1].beat - chords[0].beat);
});

/* ───────────────────── 组件函数 ───────────────────── */

test('pitchToMidi 处理升降号与八度', () => {
  const node = (step, octave, alter) => ({
    tagName: 'pitch',
    children: [
      { tagName: 'step', textContent: step, children: [] },
      { tagName: 'octave', textContent: String(octave), children: [] },
      ...(alter == null ? [] : [{ tagName: 'alter', textContent: String(alter), children: [] }]),
    ],
  });
  assert.equal(pitchToMidi(node('C', 4)), 60);
  assert.equal(pitchToMidi(node('A', 4)), 69);
  assert.equal(pitchToMidi(node('C', 4, 1)), 61);
  assert.equal(pitchToMidi(node('B', 3, -1)), 58);
  assert.equal(pitchToMidi(null), null);
});

test('harmonyKindToSuffix 覆盖常用和弦性质', () => {
  assert.equal(harmonyKindToSuffix('major'), '');
  assert.equal(harmonyKindToSuffix('minor'), 'm');
  assert.equal(harmonyKindToSuffix('dominant'), '7');
  assert.equal(harmonyKindToSuffix('major-seventh'), 'maj7');
  assert.equal(harmonyKindToSuffix('minor-seventh'), 'm7');
  assert.equal(harmonyKindToSuffix('half-diminished'), 'm7b5');
  assert.equal(harmonyKindToSuffix('diminished'), 'dim');
  assert.equal(harmonyKindToSuffix(''), '');
});

test('parseHarmony 支持斜杠低音', () => {
  const node = {
    tagName: 'harmony',
    children: [
      { tagName: 'root', children: [
        { tagName: 'root-step', textContent: 'G', children: [] },
      ] },
      { tagName: 'kind', textContent: 'major', children: [] },
      { tagName: 'bass', children: [
        { tagName: 'bass-step', textContent: 'B', children: [] },
      ] },
    ],
  };
  assert.equal(parseHarmony(node), 'G/B');
});

test('looksLikeMusicXML 按扩展名与内容判断', () => {
  assert.equal(looksLikeMusicXML('song.musicxml'), true);
  assert.equal(looksLikeMusicXML('song.xml'), true);
  assert.equal(looksLikeMusicXML('song.mp3'), false);
  assert.equal(looksLikeMusicXML('<?xml version="1.0"?><score-partwise/>'), true);
  assert.equal(looksLikeMusicXML(''), false);
});

/* ───────────────────── 拒绝与告警（不隐藏降级） ───────────────────── */

test('空内容与非法 XML 返回 null 并告警', () => {
  for (const value of ['', '   ', null]) {
    const { score, warnings } = parseMusicXML(value);
    assert.equal(score, null);
    assert.ok(warnings.length > 0);
  }
});

test('timewise 布局明确拒绝', () => {
  const { score, warnings } = parseMusicXML('<score-timewise><measure/></score-timewise>');
  assert.equal(score, null);
  assert.ok(warnings[0].includes('timewise'));
});

test('根元素不认识时拒绝', () => {
  const { score, warnings } = parseMusicXML('<foo><bar/></foo>');
  assert.equal(score, null);
  assert.ok(warnings[0].includes('foo'));
});

test('多 part 只取第一个并告警', () => {
  const xml = `<score-partwise>
    <part id="P1"><measure><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note></measure></part>
    <part id="P2"><measure><note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note></measure></part>
  </score-partwise>`;
  const { score, warnings } = parseMusicXML(xml);
  assert.ok(score);
  assert.ok(warnings.some((w) => w.includes('2 个 part')));
  assert.equal(score.bars[0].vocal[0].midi, 60, '取第一个 part 的 C4');
});

test('没有小节时返回 null', () => {
  const { score, warnings } = parseMusicXML('<score-partwise><part id="P1"/></score-partwise>');
  assert.equal(score, null);
  assert.ok(warnings.length > 0);
});

test('非零调号给出关系大小调歧义告警', () => {
  const xml = `<score-partwise>
    <part id="P1"><measure>
      <attributes><divisions>1</divisions><key><fifths>2</fifths></key></attributes>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
    </measure></part>
  </score-partwise>`;
  const { score, warnings } = parseMusicXML(xml);
  assert.ok(score);
  assert.equal(score.meta.key.num_sharps_flats, 2);
  assert.ok(warnings.some((w) => w.includes('调号')));
});

test('装饰音按 0 时值处理并告警（不静默）', () => {
  const xml = `<score-partwise>
    <part id="P1"><measure>
      <attributes><divisions>1</divisions></attributes>
      <note><grace/><pitch><step>C</step><octave>4</octave></pitch></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
    </measure></part>
  </score-partwise>`;
  const { warnings } = parseMusicXML(xml);
  assert.ok(warnings.some((w) => w.includes('装饰音')));
});

test('段落标记从 <rehearsal> 还原', () => {
  const xml = `<score-partwise>
    <part id="P1">
      <measure>
        <attributes><divisions>1</divisions></attributes>
        <direction><direction-type><rehearsal>副歌</rehearsal></direction-type></direction>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
      </measure>
      <measure>
        <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
      </measure>
    </part>
  </score-partwise>`;
  const { score } = parseMusicXML(xml);
  assert.equal(score.sections.length, 1);
  assert.equal(score.sections[0].label, '副歌');
});

/* ───────────────────── 真实产物回归 ───────────────────── */

test('解析 artifacts 里的真实 MusicXML（若存在）', async () => {
  const { readFileSync, existsSync } = await import('node:fs');
  const path = 'artifacts/score.musicxml';
  if (!existsSync(path)) return;
  const xml = readFileSync(path, 'utf8');
  const { score, warnings } = parseMusicXML(xml);
  assert.ok(score, `真实 MusicXML 应能解析，warnings=${warnings.slice(0, 3).join('|')}`);
  assert.ok(score.bars.length > 20, `真实谱应有多个小节，实际 ${score.bars.length}`);
  assert.ok(score.bars.some((b) => (b.chords || []).length > 0), '应有和弦');
});

/* ───────────────────── 汇总 ───────────────────── */

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) {
  for (const { name, err } of failures) console.error(`\n✗ ${name}\n${err.stack}`);
  process.exit(1);
}
