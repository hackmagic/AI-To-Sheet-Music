/**
 * abcjs-fallback.js —— VexFlow 渲染失败时的 ABC 兜底渲染器
 *
 * 设计约束（离线优先）：
 *   1. 导入本模块没有任何副作用：不联网、不创建 DOM、不抛错；
 *   2. 只有真正需要兜底时才懒加载 abcjs：先找本地 `web/vendor/abcjs-basic-min.js`，
 *      再按顺序试 CDN；
 *   3. 全部加载失败时返回 false，并在容器里写一段可读的中文提示，
 *      而不是让页面白屏或抛异常；
 *   4. VexFlow 正常工作时不会加载 abcjs。
 *
 * 导出的三个函数：isAvailable()、renderFallback()、renderWithFallback()。
 */

const LOCAL_VENDOR_URL = new URL('../vendor/abcjs-basic-min.js', import.meta.url).href;

// 本地 vendor 缺失时的兜底 CDN（离线环境下会快速失败并降级）
const CDN_URLS = [
  'https://cdn.jsdelivr.net/npm/abcjs@6/dist/abcjs-basic-min.js',
  'https://unpkg.com/abcjs@6/dist/abcjs-basic-min.js',
];

const LOCAL_TIMEOUT_MS = 3000;
const CDN_TIMEOUT_MS = 3500;

/** 已加载/正在加载的 abcjs 脚本（成功与失败都缓存，避免反复请求）。 */
let loaderPromise = null;

/**
 * abcjs 是否已经可用（同步检查全局对象，不触发加载）。
 * @returns {boolean}
 */
export function isAvailable() {
  const abcjs = globalThis.ABCJS;
  return Boolean(abcjs && typeof abcjs.renderAbc === 'function');
}

/**
 * 动态插入 <script> 并等待加载；超时视为失败。
 * @param {string} url
 * @param {number} timeoutMs
 * @returns {Promise<boolean>}
 */
function loadScript(url, timeoutMs) {
  return new Promise((resolve) => {
    if (!url) {
      resolve(false);
      return;
    }
    let settled = false;
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      script.onload = null;
      script.onerror = null;
      if (!ok) script.remove();
      resolve(ok);
    };
    const timer = setTimeout(() => finish(false), timeoutMs);
    script.onload = () => finish(true);
    script.onerror = () => finish(false);
    document.head.appendChild(script);
  });
}

/**
 * 懒加载 abcjs：本地 vendor 优先，其次 CDN。结果会被缓存。
 * @returns {Promise<boolean>} 是否可用
 */
async function loadAbcjs() {
  if (isAvailable()) return true;
  if (loaderPromise) return loaderPromise;

  loaderPromise = (async () => {
    if (await loadScript(LOCAL_VENDOR_URL, LOCAL_TIMEOUT_MS) && isAvailable()) return true;
    for (const url of CDN_URLS) {
      if (await loadScript(url, CDN_TIMEOUT_MS) && isAvailable()) return true;
    }
    return false;
  })();

  try {
    return await loaderPromise;
  } catch (err) {
    console.warn('[abcjs-fallback] 加载 abcjs 失败：', err);
    return false;
  }
}

/**
 * 把容器参数（元素或 id）解析成元素。
 * @param {Element|string|null} container
 * @returns {Element|null}
 */
function resolveContainer(container) {
  if (!container) return null;
  if (typeof container === 'string') {
    return document.getElementById(container) || document.querySelector(container);
  }
  if (typeof container === 'object' && container.nodeType === 1) return container;
  return null;
}

/**
 * 用 abcjs 渲染 ABC 文本。失败返回 false（不抛错、不清空已有内容）。
 *
 * @param {Element|string} container 渲染容器
 * @param {string} abc ABC 文本
 * @param {object} [opts]
 * @param {boolean} [opts.clear=true] 渲染前是否清空容器
 * @param {number} [opts.scale] 缩放倍数
 * @param {number} [opts.staffwidth] 谱面宽度（像素）
 * @param {string} [opts.responsive='resize'] abcjs responsive 模式
 * @returns {Promise<boolean>}
 */
export async function renderFallback(container, abc, opts = {}) {
  const element = resolveContainer(container);
  if (!element || typeof abc !== 'string' || !abc.trim()) return false;

  const ready = await loadAbcjs();
  if (!ready) return false;

  try {
    if (opts.clear !== false) element.innerHTML = '';
    const options = {
      responsive: opts.responsive || 'resize',
      add_classes: true,
    };
    if (Number.isFinite(opts.scale) && opts.scale > 0) options.scale = opts.scale;
    if (Number.isFinite(opts.staffwidth) && opts.staffwidth > 0) {
      options.staffwidth = opts.staffwidth;
    }
    globalThis.ABCJS.renderAbc(element, abc, options);
    return true;
  } catch (err) {
    console.warn('[abcjs-fallback] ABC 渲染失败：', err);
    return false;
  }
}

/**
 * 兜底入口：优先渲染 ABC；失败时在容器里留下可读提示（含 ABC 源文本）。
 * 任何异常都会被吞掉，调用方不需要 try/catch。
 *
 * @param {Element|string} container
 * @param {string} abc
 * @param {object} [opts] 透传给 renderFallback，另支持：
 *   - `reason`：VexFlow 的失败原因，写进提示
 *   - `notice`：设为 false 时不写提示
 *   - `showSource`：设为 false 时提示里不附带 ABC 源文本
 * @returns {Promise<boolean>} 是否成功用 abcjs 渲染
 */
export async function renderWithFallback(container, abc, opts = {}) {
  const element = resolveContainer(container);
  if (!element || typeof abc !== 'string' || !abc.trim()) return false;

  const ok = await renderFallback(element, abc, opts);
  if (ok) return true;
  if (opts.notice === false) return false;

  try {
    element.innerHTML = '';
    const notice = document.createElement('div');
    notice.className = 'abcjs-fallback-notice';
    notice.setAttribute('role', 'status');
    notice.style.cssText =
      'margin:16px;padding:16px 18px;border:1px dashed #c9a227;border-radius:10px;' +
      'background:#fffbea;color:#5c4708;font-size:14px;line-height:1.7;';

    const title = document.createElement('strong');
    title.textContent = '五线谱渲染失败，备用谱面暂不可用';
    notice.appendChild(title);

    const reasonText = typeof opts.reason === 'string' ? opts.reason.trim() : '';
    if (reasonText) {
      const reason = document.createElement('p');
      reason.style.margin = '8px 0 0';
      reason.textContent = `原因：${reasonText}`;
      notice.appendChild(reason);
    }

    const hint = document.createElement('p');
    hint.style.margin = '8px 0 0';
    hint.textContent =
      '已尝试加载 abcjs 备用渲染器，但本地 vendor 中不存在、CDN 也无法访问（离线环境）。' +
      '可以把 abcjs-basic-min.js 放到 web/vendor/ 后刷新重试；下方保留 ABC 源文本，可手工导入其他打谱软件。';
    notice.appendChild(hint);

    if (opts.showSource !== false && typeof abc === 'string' && abc.trim()) {
      const details = document.createElement('details');
      details.style.margin = '10px 0 0';
      const summary = document.createElement('summary');
      summary.textContent = '查看 ABC 源文本';
      summary.style.cursor = 'pointer';
      const pre = document.createElement('pre');
      pre.style.cssText =
        'max-height:260px;overflow:auto;margin:8px 0 0;padding:10px;' +
        'background:#fff;border-radius:6px;font-size:12px;line-height:1.5;';
      pre.textContent = abc;
      details.appendChild(summary);
      details.appendChild(pre);
      notice.appendChild(details);
    }

    element.appendChild(notice);
  } catch (err) {
    console.warn('[abcjs-fallback] 写入降级提示失败：', err);
  }
  return false;
}
