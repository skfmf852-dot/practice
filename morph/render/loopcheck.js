// Verifies the loop: seek(0) and seek(T) must render the same pixels, and the
// cursor/shape must move at the same speed across the seam.
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 1440 } });
  await p.goto("file://" + path.resolve(__dirname, "../index.html") + "?render=1");
  await p.evaluate(() => window.ready);
  const shot = async t => { await p.evaluate(t => seek(t), t); return p.screenshot(); };
  const a = await shot(0), z = await shot(14);
  console.log("frame(0) == frame(14):", Buffer.compare(a, z) === 0);
  const probe = await p.evaluate(() => {
    const read = t => { seek(t); const c = document.getElementById("cursor").style.transform; const s = document.getElementById("shape").style; return c + "|" + s.width + "|" + s.height + "|" + s.background + "|" + document.getElementById("world").style.transform; };
    return { before: read(14 - 1 / 60), at0: read(0), after: read(1 / 60), wrapBefore: read(-1 / 60) };
  });
  console.log("seek(14-1/60) === seek(-1/60):", probe.before === probe.wrapBefore);
  console.log(probe);
  await b.close();
})();
