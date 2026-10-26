/**
 * tests/web-payload.test.mjs —— payload 数据层纯 JS 单测（node 直接跑）
 *
 *   node tests/web-payload.test.mjs
 *
 * 只测 web/js/payload.js：归一化、时间轴覆盖、ABC 选择、指法图索引。
 * 这些函数不碰 DOM，所以不需要浏览器环境。
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import {
  PAYLOAD_SCHEMA,
  normalizePayload,
  attachTimeline,
  pickAbc,
  diagramFor,
  diagramKey,
  timelineDuration,
} from '../web/js/payload.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const readJson = (rel) => JSON.parse(readFileSync(resolve(ROOT, rel), 'utf8'));

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

/* ─────────────────────────── 归一化 ─────────────────────────── */

test('非对象输入返回 null', () => {
  for (const value of [null, undefined, 42, 'payload', [], true]) {
    assert.equal(normalizePayload(value), null);
  }
});

test('空对象也能归一化，字段取安全默认值', () => {
  const payload = normalizePayload({});
  assert.equal(payload.schema, '');
  assert.equal(payload.abc, '');
  assert.deepEqual(payload.timeline, []);
  assert.equal(payload.diagrams.size, 0);
  assert.equal(timelineDuration(payload), 0);
});

test('schema/abc/musicxml/audio_url 原样保留', () => {
  const payload = normalizePayload({
    schema: PAYLOAD_SCHEMA,
    title: '晴天',
    abc: 'X:1\nT:晴天\nK:C\n',
    musicxml: '<score-partwise/>',
    audio_url: '/jobs/abc/playback.mp3',
  });
  assert.equal(payload.schema, PAYLOAD_SCHEMA);
  assert.equal(payload.title, '晴天');
  assert.equal(payload.abc, 'X:1\nT:晴天\nK:C\n');
  assert.equal(payload.musicxml, '<score-partwise/>');
  assert.equal(payload.audioUrl, '/jobs/abc/playback.mp3');
});

test('时间轴按小节号排序，并丢弃脏数据', () => {
  const payload = normalizePayload({
    timing: {
      bars: [
        { index: 2, start: 4.0, end: 8.0 },
        { index: 1, start: 0, end: 4.0 },
        { index: 3, start: 9.0, end: 8.0 }, // end < start → 丢弃
        { start: 1, end: 2 }, // 缺 index → 丢弃
        null,
      ],
    },
  });
  assert.deepEqual(payload.timeline, [
    { index: 1, start: 0, end: 4 },
    { index: 2, start: 4, end: 8 },
  ]);
  assert.equal(timelineDuration(payload), 8);
});

/* ─────────────────────────── 时间轴覆盖 ─────────────────────────── */

const score = {
  meta: { tempo: 84 },
  bars: [
    { index: 1, start_sec: 0, end_sec: 2.8 },
    { index: 2, start_sec: 2.8, end_sec: 5.6 },
    { index: 3, start_sec: 5.6, end_sec: 8.4 },
  ],
};

test('没有 payload 时间轴时原样返回同一个对象', () => {
  const payload = normalizePayload({ abc: 'X:1' });
  assert.equal(attachTimeline(score, payload), score);
  assert.equal(attachTimeline(score, null), score);
});

test('按小节号覆盖 start_sec/end_sec，不改原对象', () => {
  const payload = normalizePayload({
    timing: { bars: [{ index: 2, start: 3.0, end: 6.0 }] },
  });
  const next = attachTimeline(score, payload);
  assert.notEqual(next, score);
  assert.equal(score.bars[1].start_sec, 2.8, '原 score 不能被就地修改');
  assert.equal(next.bars[1].start_sec, 3.0);
  assert.equal(next.bars[1].end_sec, 6.0);
  // 没命中的小节保持原值
  assert.equal(next.bars[0].start_sec, 0);
  assert.equal(next.bars[2].end_sec, 8.4);
});

test('时间轴与谱面完全一致时不制造新对象', () => {
  const payload = normalizePayload({
    timing: {
      bars: [
        { index: 1, start: 0, end: 2.8 },
        { index: 2, start: 2.8, end: 5.6 },
        { index: 3, start: 5.6, end: 8.4 },
      ],
    },
  });
  assert.equal(attachTimeline(score, payload), score);
});

test('缺少 bars 的谱面不炸', () => {
  assert.deepEqual(attachTimeline({}, normalizePayload({ timing: { bars: [] } })), {});
  assert.equal(attachTimeline(null, normalizePayload({})), null);
});

/* ─────────────────────────── ABC / 指法图 ─────────────────────────── */

test('payload 的 ABC 优先于 score.json', () => {
  const payload = normalizePayload({ abc: 'X:payload' });
  assert.equal(pickAbc(payload, { abc: 'X:score' }), 'X:payload');
});

test('payload 没有 ABC 时退回 score.json；两边都空返回空串', () => {
  assert.equal(pickAbc(normalizePayload({}), { abc: 'X:score' }), 'X:score');
  assert.equal(pickAbc(normalizePayload({ abc: '   ' }), { abc: '  ' }), '');
  assert.equal(pickAbc(null, null), '');
});

test('指法图按 小节号:符号 索引，缺图返回 null', () => {
  const payload = normalizePayload({
    chord_diagrams: {
      bars: [
        {
          index: 3,
          chords: [
            { symbol: 'G', diagram: { frets: [3, 2, 0, 0, 0, 3] } },
            { symbol: 'Am7', diagram: null },
          ],
        },
      ],
    },
  });
  assert.equal(diagramKey(3, 'G'), '3:G');
  assert.deepEqual(diagramFor(payload, 3, 'G')?.frets, [3, 2, 0, 0, 0, 3]);
  assert.equal(diagramFor(payload, 3, 'Am7'), null);
  assert.equal(diagramFor(payload, 9, 'G'), null);
  assert.equal(diagramFor(null, 1, 'C'), null);
});

/* ─────────────────────────── 真实产物 ─────────────────────────── */

test('真实 score.json 能被 payload 时间轴覆盖（若样例含 timing）', () => {
  let payloadJson = null;
  try {
    payloadJson = readJson('samples/payload.json');
  } catch {
    /* 样例不强制存在 */
  }
  if (!payloadJson) return;
  const payload = normalizePayload(payloadJson);
  assert.ok(payload, '真实 payload 必须能归一化');
  const next = attachTimeline(readJson('samples/demo-score.json'), payload);
  assert.ok(Array.isArray(next.bars));
});

/* ─────────────────────────── 汇总 ─────────────────────────── */

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) {
  for (const { name, err } of failures) console.error(`\n✗ ${name}\n${err.stack}`);
  process.exit(1);
}
