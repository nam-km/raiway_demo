import { spawn } from 'child_process';
import puppeteer from 'puppeteer-core';

const port = 4178;
const server = spawn('npx', ['vite', 'preview', '--port', String(port), '--strictPort'], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 2500));
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1280,800'] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 800 });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`http://localhost:${port}/`);
  await page.waitForFunction(() => !document.getElementById('loading'), { timeout: 60000 });
  await new Promise((r) => setTimeout(r, 1500));
  const drive = async () => {
    await page.keyboard.down('w');
    await new Promise((r) => setTimeout(r, Number(process.argv[2] ?? 6000)));
    await page.keyboard.up('w');
    console.log(JSON.stringify(await page.evaluate(() => ({
      policy: document.getElementById('checkpoint').textContent,
      cmd: document.getElementById('hud-cmd').textContent, act: document.getElementById('hud-act').textContent,
      banner: document.getElementById('banner').hidden ? null : document.getElementById('banner').textContent }))));
  };
  await drive();
  const others = await page.evaluate(() => [...document.getElementById('policy').options].filter((o) => !o.selected).map((o) => o.value));
  for (const name of others) {
    await page.select('#policy', name);
    await new Promise((r) => setTimeout(r, 1500));
    await drive();
  }
  await page.screenshot({ path: process.argv[3] ?? 'test/screenshot.png' });
  console.log('errors:', errors.slice(0, 10));
} finally {
  await browser.close();
  server.kill();
}
