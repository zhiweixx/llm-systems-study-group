/* Week 5 conversion QA: local resources, navigation, layout, print, and PDF fidelity.
 * Run with Playwright on NODE_PATH. Outputs stay under .build/week5-qa/.
 */
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const out = path.join(root, '.build/week5-qa');
const htmlPath = path.join(root, 'week-5-linear-attention.html');
const python = process.env.WEEK5_QA_PYTHON || 'python3';
const report = { entry: htmlPath, slides: [], checks: [], failures: [], pageErrors: [], consoleErrors: [], externalRequests: [], failedRequests: [], responsive: [], screenshots: [] };
function check(name, ok, details) {
  const result = { name, passed: Boolean(ok), ...(details === undefined ? {} : { details }) };
  report.checks.push(result);
  if (!result.passed) report.failures.push(result);
}

(async () => {
  await fs.mkdir(out, { recursive: true });
  const browser = await chromium.launch({
    ...(process.env.WEEK5_QA_CHROME ? { executablePath: process.env.WEEK5_QA_CHROME } : { channel: 'chrome' }),
    headless: true
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 770 }, deviceScaleFactor: 1, offline: true });
    const page = await context.newPage();
    page.on('pageerror', error => report.pageErrors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') report.consoleErrors.push(message.text()); });
    page.on('request', request => { if (/^https?:/i.test(request.url())) report.externalRequests.push(request.url()); });
    page.on('requestfailed', request => report.failedRequests.push({ url: request.url(), error: request.failure()?.errorText }));
    const settle = async () => {
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'load' });
    await page.waitForFunction(() => Boolean(window.deck));
    await settle();

    report.slides = await page.locator('.slide').evaluateAll(slides => slides.map(slide => ({
      id: slide.id, title: slide.dataset.title, svgCount: [...slide.children].filter(child => child.tagName.toLowerCase() === 'svg').length,
      svgViewBox: slide.querySelector(':scope > svg')?.getAttribute('viewBox')
    })));
    check('49 original pages, in order', report.slides.length === 49 && report.slides.every((slide, i) => slide.id === `slide-${i + 1}`), report.slides.map(slide => slide.id));
    check('one direct-child SVG per page', report.slides.every(slide => slide.svgCount === 1));
    const duplicateIds = await page.locator('[id]').evaluateAll(elements => {
      const counts = new Map();
      elements.forEach(element => counts.set(element.id, (counts.get(element.id) || 0) + 1));
      return [...counts].filter(([, count]) => count > 1);
    });
    check('document IDs are unique', duplicateIds.length === 0, duplicateIds);
    const credit = page.locator('#slide-1 > svg #author-credit');
    check('first slide credits Gaotang Li', await credit.count() === 1 && /Original slides by Gaotang Li/.test(await credit.textContent()), await credit.allTextContents());
    check('no notes button', await page.locator('#notes-toggle').count() === 0);
    const controlIds = ['prev', 'next', 'counter', 'overview-toggle', 'fullscreen', 'print', 'help-toggle'];
    for (const id of controlIds) check(`control #${id} exists`, await page.locator(`#${id}`).count() === 1);

    const references = await page.evaluate(() => {
      const refs = [];
      const collectCss = (value, element) => {
        for (const match of value.matchAll(/url\(\s*['"]?([^)'"\s]+)['"]?\s*\)/gi)) refs.push({ tag: element.tagName, attribute: 'url()', value: match[1] });
      };
      for (const element of document.querySelectorAll('.slide > svg, .slide > svg *')) {
        for (const attribute of element.attributes) {
          if ((attribute.localName === 'href' || attribute.localName === 'src') && element.localName !== 'a') {
            refs.push({ tag: element.tagName, attribute: attribute.name, value: attribute.value });
          }
          collectCss(attribute.value, element);
        }
        if (element.localName === 'style') collectCss(element.textContent, element);
      }
      return refs.map(ref => ({ ...ref, ...(ref.value.startsWith('#') ? { targetExists: Boolean(document.getElementById(ref.value.slice(1))) } : {}) }));
    });
    const resourceIssues = [];
    for (const ref of references) {
      if (ref.value.startsWith('#')) {
        if (!ref.targetExists) resourceIssues.push(ref);
      } else if (!/^(data:|blob:)/i.test(ref.value)) {
        const resolved = new URL(ref.value, pathToFileURL(htmlPath));
        if (resolved.protocol !== 'file:') resourceIssues.push(ref);
        else {
          try { await fs.access(decodeURIComponent(resolved.pathname)); }
          catch { resourceIssues.push(ref); }
        }
      }
    }
    report.resourceSummary = { total: references.length, embedded: references.filter(ref => /^data:/i.test(ref.value)).length, fragment: references.filter(ref => ref.value.startsWith('#')).length, issues: resourceIssues };
    check('SVG resources resolve locally or are embedded', resourceIssues.length === 0, report.resourceSummary);

    const assertCurrent = async (index, label) => {
      await settle();
      const state = await page.evaluate(() => ({ index: window.deck.index, active: [...document.querySelectorAll('.slide:not([hidden])')].map(slide => slide.id), counter: document.querySelector('#counter').textContent.trim(), hash: location.hash }));
      check(label, state.index === index && state.active.length === 1 && state.active[0] === `slide-${index + 1}` && state.counter === `${index + 1} / 49` && state.hash === `#slide-${index + 1}`, state);
    };
    await assertCurrent(0, 'offline initial page');
    await page.keyboard.press('ArrowRight'); await assertCurrent(1, 'right arrow advances');
    await page.keyboard.press('PageDown'); await assertCurrent(2, 'PageDown advances');
    await page.keyboard.press('Space'); await assertCurrent(3, 'Space advances');
    await page.keyboard.press('ArrowLeft'); await assertCurrent(2, 'left arrow retreats');
    await page.keyboard.press('PageUp'); await assertCurrent(1, 'PageUp retreats');
    await page.keyboard.press('End'); await assertCurrent(48, 'End reaches final page');
    check('next disabled at final page', await page.locator('#next').isDisabled());
    await page.keyboard.press('Home'); await assertCurrent(0, 'Home reaches first page');
    check('previous disabled at first page', await page.locator('#prev').isDisabled());
    await page.locator('#next').click(); await assertCurrent(1, 'Next button advances');
    await page.keyboard.press('ArrowRight'); await assertCurrent(2, 'keyboard works after chrome button focus');
    await page.locator('#prev').click(); await assertCurrent(1, 'Previous button retreats');
    await page.evaluate(() => { location.hash = '#slide-17'; });
    await page.waitForFunction(() => window.deck.index === 16);
    await assertCurrent(16, 'hash change opens requested page');
    await page.goto(pathToFileURL(htmlPath).href + '#slide-26', { waitUntil: 'load' });
    await page.waitForFunction(() => window.deck?.index === 25);
    await assertCurrent(25, 'direct hash URL opens requested page');
    await page.locator('#overview-toggle').click();
    check('overview opens with all 49 pages', await page.locator('#overview-panel').isVisible() && await page.locator('.overview-item').count() === 49);
    await page.keyboard.press('ArrowRight');
    check('overview keeps current slide stable', await page.evaluate(() => window.deck.index) === 25);
    await page.locator('.overview-item[data-slide-index="31"]').click();
    await assertCurrent(31, 'overview selection opens requested page');
    check('overview closes after selection', await page.locator('#overview-panel').isHidden());
    await page.locator('#overview-toggle').click();
    await page.keyboard.press('Escape');
    check('Escape closes overview', await page.locator('#overview-panel').isHidden());
    await page.locator('#help-toggle').click();
    check('help opens', await page.locator('#help-panel').isVisible());
    await page.keyboard.press('Escape');
    check('Escape closes help', await page.locator('#help-panel').isHidden());
    await page.evaluate(() => document.activeElement?.blur());

    for (let index = 0; index < report.slides.length; index++) {
      await page.evaluate(index => window.deck.goTo(index), index);
      await settle();
      const svg = page.locator('.slide:not([hidden]) > svg');
      const bounds = await svg.boundingBox();
      check(`page ${index + 1} renders within slide viewport`, bounds && bounds.width > 0 && bounds.height > 0 && bounds.x >= -1 && bounds.y >= -1 && bounds.x + bounds.width <= 1281 && bounds.y + bounds.height <= 771, bounds);
      const filename = `slide-${String(index + 1).padStart(2, '0')}.png`;
      await page.screenshot({ path: path.join(out, filename), animations: 'disabled' });
      report.screenshots.push({ page: index + 1, file: filename, bounds });
    }
    for (const [width, height] of [[1600, 950], [1280, 770], [1024, 650], [768, 560], [390, 844]]) {
      await page.setViewportSize({ width, height });
      await settle();
      const layout = await page.evaluate(() => {
        const rect = element => { const b = element.getBoundingClientRect(); return { x: b.x, y: b.y, width: b.width, height: b.height }; };
        return { stage: rect(document.querySelector('#stage')), viewport: rect(document.querySelector('#viewport')), bodyScrollWidth: document.body.scrollWidth, documentScrollWidth: document.documentElement.scrollWidth };
      });
      report.responsive.push({ width, height, ...layout });
      const b = layout.stage;
      check(`responsive stage stays in ${width}x${height}`, b.x >= -1 && b.y >= -1 && b.x + b.width <= width + 1 && b.y + b.height <= height + 1 && layout.documentScrollWidth <= width + 1, layout);
    }
    await page.setViewportSize({ width: 1280, height: 770 });
    await page.locator('#fullscreen').click();
    await page.waitForFunction(() => document.fullscreenElement !== null && document.querySelector('#fullscreen').getAttribute('aria-pressed') === 'true');
    check('fullscreen button opens presentation', await page.locator('#fullscreen').getAttribute('aria-pressed') === 'true');
    await page.locator('#fullscreen').click();
    await page.waitForFunction(() => document.fullscreenElement === null && document.querySelector('#fullscreen').getAttribute('aria-pressed') === 'false');
    check('fullscreen button exits presentation', await page.locator('#fullscreen').getAttribute('aria-pressed') === 'false');
    await page.evaluate(() => { window.__originalPrint = window.print; window.__printCalls = 0; window.print = () => window.__printCalls++; });
    await page.locator('#print').click();
    check('print button invokes browser print', await page.evaluate(() => window.__printCalls) === 1);
    await page.evaluate(() => { window.print = window.__originalPrint; delete window.__originalPrint; });
    await page.evaluate(() => window.deck.goTo(24));
    const printPath = path.join(out, 'print-qa.pdf');
    try {
      await page.pdf({ path: printPath, preferCSSPageSize: true, printBackground: true });
      report.print = JSON.parse(execFileSync(python, ['-c', 'import json,sys; from pypdf import PdfReader; r=PdfReader(sys.argv[1]); print(json.dumps({"pages":len(r.pages),"sizes":[[float(p.mediabox.width),float(p.mediabox.height)] for p in r.pages]}))', printPath], { encoding: 'utf8' }));
      check('print produces exactly 49 pages', report.print.pages === 49, report.print);
      check('all printed pages are 16:9 landscape', report.print.sizes.every(([width, height]) => Math.abs(width / height - 16 / 9) < 0.01), report.print.sizes[0]);
    } finally {
      await fs.rm(printPath, { force: true });
    }
    await assertCurrent(24, 'printing restores active slide');
    check('no browser JavaScript errors', report.pageErrors.length === 0, report.pageErrors);
    check('no browser console errors', report.consoleErrors.length === 0, report.consoleErrors);
    check('no network requests while offline', report.externalRequests.length === 0, report.externalRequests);
    check('no failed resource loads', report.failedRequests.length === 0, report.failedRequests);

    // Render the source at the exact screenshot size. The source PDF's height
    // is 405.014173 pt; width-only rendering rounds to 721 px, and resizing that
    // image to 720 px introduces false glyph-edge differences.
    await fs.mkdir(path.join(out, 'source-exact'), { recursive: true });
    execFileSync(process.env.WEEK5_QA_PDFTOCAIRO || 'pdftocairo', [
      '-png', '-scale-to-x', '1280', '-scale-to-y', '720',
      path.join(root, 'sources/week-5/Linear_Attention_gaotang_li.pdf'),
      path.join(out, 'source-exact/page')
    ], { stdio: 'pipe' });
    report.referenceRender = { width: 1280, height: 720, renderer: 'pdftocairo', scaling: 'explicit width and height' };
    await fs.writeFile(path.join(out, 'review.json'), JSON.stringify(report, null, 2));
    const comparison = String.raw`
import json,sys
from pathlib import Path
from PIL import Image,ImageChops,ImageStat,ImageDraw
root=Path(sys.argv[1]); out=root/'.build/week5-qa'; source=out/'source-exact'
report=json.loads((out/'review.json').read_text()); comparisons=[]; thumbs=[]
for shot in report['screenshots']:
    i=shot['page']; source_path=source/f'page-{i:02d}.png'
    if not source_path.exists(): continue
    image=Image.open(out/shot['file']).convert('RGB'); b=shot['bounds']
    crop=image.crop((round(b['x']),round(b['y']),round(b['x']+b['width']),round(b['y']+b['height'])))
    original=Image.open(source_path).convert('RGB').resize(crop.size,Image.Resampling.LANCZOS)
    diff=ImageChops.difference(original,crop); stat=ImageStat.Stat(diff)
    hist=diff.convert('L').histogram(); changed=sum(hist[48:])/sum(hist)
    comparisons.append({'page':i,'meanAbsoluteRgbDifference':round(sum(stat.mean)/3,4),'fractionPixelsDifferenceAbove48':round(changed,6),'expectedTitleChanges':i==1})
    row=Image.new('RGB',(640,202),'#ededed'); draw=ImageDraw.Draw(row)
    draw.text((8,3),f'{i:02d} source',fill='black'); draw.text((328,3),f'{i:02d} HTML',fill='black')
    row.paste(original.resize((320,180),Image.Resampling.LANCZOS),(0,22)); row.paste(crop.resize((320,180),Image.Resampling.LANCZOS),(320,22)); thumbs.append(row)
for start in range(0,len(thumbs),12):
    batch=thumbs[start:start+12]; sheet=Image.new('RGB',(1280,202*((len(batch)+1)//2)),'white')
    for i,thumb in enumerate(batch): sheet.paste(thumb,((i%2)*640,(i//2)*202))
    sheet.save(out/f'comparison-{start+1:02d}-{start+len(batch):02d}.png')
print(json.dumps(comparisons))
`;
    report.visualComparison = JSON.parse(execFileSync(python, ['-c', comparison, root], { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 }));
    report.visualWarnings = report.visualComparison.filter(item => !item.expectedTitleChanges && (item.meanAbsoluteRgbDifference > 8 || item.fractionPixelsDifferenceAbove48 > 0.05));
    check('source comparisons available for all pages', report.visualComparison.length === 49, { compared: report.visualComparison.length });
    check('no large visual differences outside intentional title edits', report.visualWarnings.length === 0, report.visualWarnings);
  } catch (error) {
    report.fatal = error.stack;
    check('QA run completes', false, error.message);
  } finally {
    await browser.close();
    await fs.writeFile(path.join(out, 'review.json'), JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify({ slides: report.slides.length, checks: report.checks.length, failures: report.failures, visualWarnings: report.visualWarnings, report: path.join(out, 'review.json') }, null, 2));
  if (report.failures.length) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
