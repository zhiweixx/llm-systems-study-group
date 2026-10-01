/* Layout, offline behavior, navigation, and step-frame checks for Week 4. */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const root=path.resolve(__dirname,'..'),out=path.join(root,'.build/week4-qa');
(async()=>{
 await fs.mkdir(out,{recursive:true});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1600,height:960}});
 const errors=[],requests=[],issues=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(/^https?:/.test(r.url()))requests.push(r.url());});
 await page.goto('file://'+path.join(root,'week-4-serving.html'));
 const settle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));await settle();
 const slides=await page.locator('.slide').evaluateAll(ss=>ss.map(s=>({id:s.id,title:s.dataset.title,section:s.dataset.section})));
 assert.equal(slides.length,30);
 assert.equal(slides[1].title,'GPU memory hierarchy and bandwidth');
 assert.ok(slides.slice(2,12).every(s=>s.section.startsWith('Speculative decoding')));
 assert.ok(slides.slice(6,12).every(s=>s.section.startsWith('Speculative decoding theory')));
 assert.ok(slides[6].title.startsWith('Quiz:'));
 assert.ok(slides[7].title.startsWith('Quiz solution:'));
 assert.ok(slides[8].title.startsWith('Why speculative sampling is exact'));
 assert.equal(slides[17].title,'Serve a 405B model on eight H100 nodes');
 assert.ok(slides.slice(17).every(s=>s.section.startsWith('Case study')));
 assert.equal(slides[29].title,'Which layout would we actually choose?');
 const inspect=async label=>{
  const result=await page.locator('.slide:not([hidden])>svg').evaluate(svg=>{
   const visible=e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden';};
   const ts=[...svg.querySelectorAll('text')].filter(t=>t.textContent.trim()&&visible(t)).map(t=>({text:t.textContent,b:t.getBBox()}));
   const outside=ts.filter(t=>t.b.x<45||t.b.x+t.b.width>1557||t.b.y<0||t.b.y+t.b.height>899).map(t=>t.text);
   const overlaps=[];
   for(let i=0;i<ts.length;i++)for(let j=i+1;j<ts.length;j++){
    const a=ts[i].b,b=ts[j].b,dx=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x),dy=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
    if(dx>5&&dy>Math.min(a.height,b.height)*.3)overlaps.push([ts[i].text,ts[j].text]);
   }
   const foreign=[];
   for(const fo of [...svg.querySelectorAll('foreignObject')].filter(visible)){
    const b=fo.getBoundingClientRect(),child=fo.firstElementChild;
    if(child.scrollWidth>fo.width.baseVal.value+2||child.scrollHeight>fo.height.baseVal.value+2)foreign.push('Overflow: '+child.textContent.slice(0,90));
    for(const e of fo.querySelectorAll('math,pre,button,a')){
     const a=e.getBoundingClientRect();if(a.left<b.left-2||a.right>b.right+2||a.top<b.top-2||a.bottom>b.bottom+2)foreign.push('Outside: '+e.textContent.slice(0,90));
    }
   }
   return {outside,overlaps,foreign};
  });
  if(result.outside.length||result.overlaps.length||result.foreign.length)issues.push({label,...result});
 };
 for(let i=0;i<slides.length;i++){
  if(slides[i].title.startsWith('Question '))assert.ok(slides[i+1].title.startsWith(slides[i].title.replace('Question','Solution').split(':')[0]));
  await page.evaluate(n=>window.deck.goTo(n),i);await settle();
  assert.equal(await page.locator('.slide:not([hidden])').count(),1);
  assert.equal(await page.locator('#counter').innerText(),`${i+1} / ${slides.length}`);
  await inspect(slides[i].id);
  await page.locator('.slide:not([hidden])').screenshot({path:path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`)});
  const kinds=await page.locator('.slide:not([hidden]) [data-step-next]').evaluateAll(bs=>bs.map(b=>b.dataset.stepNext));
  for(const kind of kinds){
   const state=await page.evaluate(()=>window.week4Examples.state());
   for(let f=1;f<state[kind].count;f++){
    await page.locator(`[data-step-next="${kind}"]`).click();await settle();
    assert.equal((await page.evaluate(()=>window.week4Examples.state()))[kind].index,f);
    await inspect(`${slides[i].id}-${kind}-${f}`);
    await page.locator('.slide:not([hidden])').screenshot({path:path.join(out,`${kind}-${f}.png`)});
   }
   await page.evaluate(k=>window.week4Examples.show(k,0),kind);
  }
 }
 const costIndex=slides.findIndex(s=>s.title==='When does the extra work pay off?');
 assert.equal(costIndex,5);
 await page.evaluate(n=>window.deck.goTo(n),costIndex);await settle();
 for(const [a,v,expected] of [[0,14,.5],[2,14,1.5],[4,14,2.5],[4,34,1.25],[0,34,.25],[4,10,50/16]]){
  await page.evaluate(([a,v])=>{
   const accepted=document.getElementById('spec-accepted'),verify=document.getElementById('spec-verify');
   accepted.value=a;verify.value=v;verify.dispatchEvent(new Event('input',{bubbles:true}));
  },[a,v]);await settle();
  const state=await page.evaluate(()=>window.specCost.state());
  assert.equal(state.speedup,expected);assert.equal(state.emitted,a+1);
  await inspect(`spec-cost-${a}-${v}`);
  await page.locator('.slide:not([hidden])').screenshot({path:path.join(out,`spec-cost-${a}-${v}.png`)});
 }
 const exampleState=await page.evaluate(()=>window.week4Examples.state());
 assert.equal(exampleState['spec-greedy'],undefined);assert.equal(exampleState['spec-sampling'],undefined);
 await page.locator('#notes-toggle').click();assert.ok((await page.locator('#notes-content').innerText()).length>100);await page.keyboard.press('Escape');
 await page.locator('#overview-toggle').click();assert.equal(await page.locator('.overview-item').count(),slides.length);await page.keyboard.press('Escape');
 for(const [width,height] of [[1280,770],[1024,650],[768,560]]){
  await page.setViewportSize({width,height});await settle();const b=await page.locator('#stage').boundingBox();assert.ok(b.x>=-1&&b.y>=-1&&b.x+b.width<=width+1);
 }
 const before=await page.evaluate(()=>window.week4Examples.state());
 const costBefore=await page.evaluate(()=>window.specCost.state());
 await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));
 const printing=await page.evaluate(()=>window.week4Examples.state());for(const s of Object.values(printing))assert.equal(s.index,s.count-1);
 assert.equal((await page.evaluate(()=>window.specCost.state())).speedup,1.5);
 await page.evaluate(()=>window.dispatchEvent(new Event('afterprint')));assert.deepEqual(await page.evaluate(()=>window.week4Examples.state()),before);
 assert.deepEqual(await page.evaluate(()=>window.specCost.state()),costBefore);
 await fs.writeFile(path.join(out,'review.json'),JSON.stringify({slides,issues,errors,requests},null,2));await browser.close();
 console.log(JSON.stringify({slides:slides.length,issues,errors,requests},null,2));
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);assert.deepEqual(issues,[]);
})().catch(e=>{console.error(e);process.exit(1)});
