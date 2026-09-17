/* Browser QA for the self-contained Week 2 deck. Playwright + Chrome. */
const fs=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'.build/week2-qa');
(async()=>{
  await fs.mkdir(out,{recursive:true});
  const browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({viewport:{width:1600,height:960}});
  const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
  await page.goto('file://'+path.join(root,'week-2-inference.html'));
  const settle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  await settle();
  const slides=await page.locator('.slide').evaluateAll(ss=>ss.map(s=>({id:s.id,title:s.dataset.title,minutes:Number(s.dataset.minutes)})));
  assert.equal(slides.length,32);assert.equal(slides.reduce((n,s)=>n+s.minutes,0),34.75);
  const issues=[];
  async function inspect(label){
    const result=await page.locator('.slide:not([hidden]) svg').evaluate(svg=>{
      const ts=[...svg.querySelectorAll('text')].filter(t=>t.textContent.trim()).map(t=>({text:t.textContent,b:t.getBBox()}));
      const outside=ts.filter(t=>t.b.x<30||t.b.x+t.b.width>1568||t.b.y<0||t.b.y+t.b.height>899).map(t=>t.text);
      const overlaps=[];
      for(let i=0;i<ts.length;i++)for(let j=i+1;j<ts.length;j++){
        const a=ts[i].b,b=ts[j].b;
        const dx=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x);
        const dy=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
        if(dx>5&&dy>Math.min(a.height,b.height)*.3)overlaps.push([ts[i].text,ts[j].text]);
      }
      const foreign=[];
      for(const fo of svg.querySelectorAll('foreignObject')){
        const b=fo.getBoundingClientRect();
        const child=fo.firstElementChild;
        if(child.scrollWidth>fo.width.baseVal.value+2||child.scrollHeight>fo.height.baseVal.value+2)foreign.push('Overflow: '+child.textContent.slice(0,100));
        for(const e of fo.querySelectorAll('math,pre,button,a')){
          const a=e.getBoundingClientRect();
          if(a.left<b.left-2||a.right>b.right+2||a.top<b.top-2||a.bottom>b.bottom+2)foreign.push('Outside: '+e.textContent.slice(0,100));
        }
      }
      return {outside,overlaps,foreign};
    });
    if(result.outside.length||result.overlaps.length||result.foreign.length)issues.push({label,...result});
  }
  for(let i=0;i<slides.length;i++){
    if(slides[i].title.startsWith('Question '))assert.ok(slides[i+1].title.startsWith(slides[i].title.replace('Question','Solution').split(':')[0]));
    await page.evaluate(n=>{location.hash='slide-'+n;},i+1);await settle();
    assert.equal(await page.locator('.slide:not([hidden])').count(),1);
    assert.equal(await page.locator('#counter').innerText(),`${i+1} / ${slides.length}`);
    await inspect(`slide-${i+1}`);
    await page.locator('.slide:not([hidden])').screenshot({path:path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`)});
  }
  const pagingStates=[];
  async function checkPaging(phase){
    const p=await page.evaluate(()=>window.week2Examples.pagingState());
    assert.equal(p.phase,phase);assert.equal(p.blockSize,4);
    assert.equal(p.tokens,[6,8,9,0,3][phase]);
    assert.deepEqual(p.mapping,[[4,1],[4,1],[4,1,5],[],[4]][phase]);
    assert.deepEqual(p.freeBlocks,[[3,5],[3,5],[3],[1,3,4,5],[1,3,5]][phase]);
    const valid=p.mapping.flatMap(id=>p.pool[id].slots).filter(Boolean);
    assert.deepEqual(valid,Array.from({length:p.tokens},(_,i)=>`${p.owner}${i+1}`));
    assert.equal(new Set(p.mapping).size,p.mapping.length);
    for(const id of [0,2])assert.equal(p.pool[id].owner,'C');
    if(phase===1||phase===2)assert.deepEqual(p.pool[4].slots,pagingStates[0].pool[4].slots);
    pagingStates.push(p);
  }
  for(const [kind,number,last] of [['generation',3,2],['paging',15,4]]){
    await page.evaluate(n=>{location.hash='slide-'+n;},number);await settle();
    if(kind==='paging')await checkPaging(0);
    for(let n=1;n<=last;n++){
      await page.locator(`[data-step-next="${kind}"]`).click();
      assert.equal(await page.locator(`#slide-${number}`).getAttribute('data-example-step'),String(n));
      if(kind==='paging')await checkPaging(n);
      await inspect(`${kind}-${n}`);
      await page.locator(`#slide-${number}`).screenshot({path:path.join(out,`${kind}-${n}.png`)});
    }
    await page.locator(`[data-step-prev="${kind}"]`).click();
  }
  assert.equal(await page.locator('#batch-chart').count(),0);
  assert.equal(await page.locator('#slide-25 image').count(),1);
  assert.match(await page.locator('#slide-25').textContent(),/DistServe/);
  for(let n=17;n<=19;n++)assert.ok(await page.locator(`#slide-${n} math`).count()>0);
  assert.match(await page.locator('#slide-22 pre').textContent(),/torch.cat/);
  await page.evaluate(()=>{location.hash='slide-19';});await settle();
  await page.locator('#notes-toggle').click();assert.match(await page.locator('#notes-content').innerText(),/exp\(s−m′\)/);await page.keyboard.press('Escape');
  await page.locator('#overview-toggle').click();assert.equal(await page.locator('.overview-item').count(),32);await page.keyboard.press('Escape');
  for(const [width,height] of [[1280,770],[1024,650],[768,560]]){
    await page.setViewportSize({width,height});await settle();
    const b=await page.locator('#stage').boundingBox();assert.ok(b.x>=-1&&b.y>=-1&&b.x+b.width<=width+1);
  }
  await page.setViewportSize({width:1600,height:960});await settle();
  const pagingBefore=await page.evaluate(()=>window.week2Examples.pagingState());
  await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
  assert.equal(await page.locator('#slide-15').getAttribute('data-example-step'),'4');
  await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));
  assert.deepEqual(await page.evaluate(()=>window.week2Examples.pagingState()),pagingBefore);
  await fs.writeFile(path.join(out,'review.json'),JSON.stringify({slides,issues,errors,requests,pagingStates},null,2));
  await browser.close();
  console.log(JSON.stringify({slides:slides.length,minutes:34.75,issues,errors,requests},null,2));
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);assert.deepEqual(issues,[]);
})().catch(e=>{console.error(e);process.exit(1);});
