// Render frames of index.html with Playwright.
//   node render/render.js times out/beats 0 0.5 1.0 ...   one PNG per time
//   node render/render.js full out/sub [workers] [fps] [sub]  default 60 fps x 4 subframes
//   node render/render.js sounds audio/sounds.json        dump window.SOUNDS
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const URL = "file://" + path.join(ROOT, "index.html") + "?render=1";
const SHUTTER = 0.75; // 270-degree shutter

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1440 }, deviceScaleFactor: 1 });
  await page.goto(URL);
  await page.evaluate(() => window.ready);
  return page;
}

async function shoot(page, t, file) {
  await page.evaluate(t => window.seek(t), t);
  await page.screenshot({ path: file, type: "png" });
}

(async () => {
  const [mode, out, ...rest] = process.argv.slice(2);
  const browser = await chromium.launch();
  if (mode === "sounds") {
    const page = await openPage(browser);
    const s = await page.evaluate(() => ({ T: window.T, events: window.SOUNDS }));
    fs.writeFileSync(out, JSON.stringify(s, null, 1));
  } else if (mode === "times") {
    fs.mkdirSync(out, { recursive: true });
    const page = await openPage(browser);
    for (const ts of rest) await shoot(page, parseFloat(ts), path.join(out, `t${parseFloat(ts).toFixed(3).padStart(7, "0")}.png`));
  } else if (mode === "full") {
    fs.mkdirSync(out, { recursive: true });
    const workers = parseInt(rest[0] || "4", 10), FPS = parseInt(rest[1] || "60", 10), SUB = parseInt(rest[2] || "4", 10);
    const T = 14, frames = T * FPS;
    const jobs = [];
    for (let f = 0; f < frames; f++)
      for (let k = 0; k < SUB; k++) {
        // subframes spread across the shutter interval, centred on the frame time
        const off = SUB === 1 ? 0 : ((k + 0.5) / SUB - 0.5) * SHUTTER;
        jobs.push({ t: (f + off) / FPS, file: path.join(out, `s${String(f * SUB + k).padStart(5, "0")}.png`) });
      }
    let next = 0, done = 0;
    const t0 = Date.now();
    await Promise.all(Array.from({ length: workers }, async () => {
      const page = await openPage(browser);
      while (next < jobs.length) {
        const j = jobs[next++];
        if (!fs.existsSync(j.file)) await shoot(page, j.t, j.file);
        if (++done % 240 === 0) console.log(`${done}/${jobs.length}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      }
    }));
  }
  await browser.close();
})();
