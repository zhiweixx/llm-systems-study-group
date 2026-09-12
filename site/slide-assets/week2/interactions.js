/* Offline teaching examples. Default performance values are illustrative only. */
(() => {
  'use strict';
  const BLUE='#245675', INK='#172329', MUTED='#58656d', PALE='#edf3f7', LINE='#aec3d1', TEAL='#397b71';
  const e=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,v,size=28,weight=400,color=INK,anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}">${e(v)}</text>`;
  const rect=(x,y,w,h,fill='white',stroke=LINE)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
  const box=(x,y,w,h,v,fill=PALE,size=27)=>rect(x,y,w,h,fill)+text(x+w/2,y+h/2+size*.34,v,size,600,INK,'middle');
  const line=(x1,y1,x2,y2,color=LINE,width=1.5)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
  const arrow=(x1,y,x2,color=BLUE)=>line(x1,y,x2,y,color,2)+`<path d="M${x2-9},${y-6} L${x2},${y} L${x2-9},${y+6}" fill="none" stroke="${color}" stroke-width="2"/>`;
  const state={generation:0,flash:0,softmax:0};
  const max={generation:2,flash:3,softmax:2};
  const labels={
    generation:['Prefill complete: first token predicted','Decode 1 complete: second token predicted','Decode 2 complete: third token predicted'],
    flash:['1 / 4: load the first K/V tile','2 / 4: compute and retain partial state','3 / 4: replace the tile and update state','4 / 4: normalize and write the output'],
    softmax:['1 / 3: first tile','2 / 3: rescale and include the second tile','3 / 3: normalize the combined result']
  };
  function generation(){
    const n=state.generation, words=['Explain','why','GPUs','help','They','run','fast'];
    let s=text(75,253,'Prompt',25,700,BLUE)+text(849,253,'Generated tokens',25,700,TEAL);
    for(let i=0;i<7;i++){
      const present=i<=4+n, active=i===4+n;
      s+=`<g opacity="${present?1:.22}">`+box(75+i*193,273,174,64,words[i],i<4?PALE:(active?'#dcece6':'#f2f6f4'),29)+'</g>';
    }
    const stages=[['Prefill','Process 4 prompt tokens','Predict “They”'],['Decode 1','Process “They”','Predict “run”'],['Decode 2','Process “run”','Predict “fast”']];
    stages.forEach((a,i)=>{
      s+=`<g opacity="${i<=n?1:.25}">`+rect(75+i*490,376,460,133,i===n?PALE:'white')+text(95+i*490,417,a[0],29,700,BLUE)+text(95+i*490,456,a[1],25)+text(95+i*490,490,a[2],25)+'</g>';
    });
    s+=text(75,555,'KV cache after this pass',27,700)+text(75,601,`${4+n} processed positions`,26,400,MUTED);
    for(let i=0;i<6;i++)s+=`<g opacity="${i<4+n?1:.18}">`+box(435+i*115,549,104,65,words[i],i<4?PALE:'#dcece6',22)+'</g>';
    s+=text(1160,569,'The last predicted token',24)+text(1160,607,'has not entered the cache yet.',24);
    const scene=document.querySelector('#generation-scene');
    scene.innerHTML=s;
    scene.closest('.slide').dataset.exampleStep=String(n);
  }
  // One query with q=1, d=1 gives scores equal to scalar keys ln(1..4).
  function flash(){
    const n=state.flash, tile=n<2?0:1, done=n===3;
    let s=rect(75,237,625,390)+rect(790,237,715,390,PALE)+text(100,282,'HBM',31,700,BLUE)+text(815,282,'On-chip working data',31,700,BLUE);
    s+=text(100,328,'Last causal query: q = 1, head dimension d = 1',25);
    s+=text(100,389,'Keys',26,700)+text(100,461,'Values',26,700);
    for(let i=0;i<4;i++){
      const fill=Math.floor(i/2)===tile?'#dcece6':'white';
      s+=box(210+i*112,351,105,56,`ln ${i+1}`,fill,25)+box(210+i*112,423,105,56,(i+1)*10,fill,25);
    }
    s+=text(100,531,'All four positions remain in HBM.',25,400,MUTED);
    s+=text(100,571,'A score tile is temporary.',25,400,MUTED);
    s+=arrow(714,393,775);
    s+=text(815,330,`Loaded K/V tile ${tile+1}`,27,700);
    for(let i=0;i<2;i++)s+=box(815+i*190,353,172,66,`(ln ${tile*2+i+1}, ${(tile*2+i+1)*10})`,'white',26);
    s+=text(815,456,'Running state stays on chip',27,700);
    let status=n===0?['m = −∞','ℓ = 0','u = 0']:n===1?['m = ln 2','ℓ = 1.5','u = 25']:['m = ln 4','ℓ = 2.5','u = 75'];
    status.forEach((v,i)=>s+=box(815+i*218,481,202,60,v,'white',27));
    s+=text(815,590,done?'Final output: u / ℓ = 75 / 2.5 = 30':n===0?'Next: compute this tile’s contribution.':n===1?'Keep state. Discard the score tile.':'Rescale prior state and add this tile.',27,700,done?TEAL:BLUE);
    if(done)s+=text(100,611,'Output in HBM: 30',28,700,TEAL);
    const scene=document.querySelector('#flash-scene');
    scene.innerHTML=s;
    scene.closest('.slide').dataset.exampleStep=String(n);
  }
  function softmax(){
    const n=state.softmax;
    let s=line(75,288,1525,288)+text(75,338,'Tile 1',30,700,BLUE)+text(815,338,n===0?'State to retain':'Tile 2',30,700,BLUE);
    s+=text(75,391,'Maximum m = ln 2',30)+text(75,444,'Scaled weights: [0.5, 1]',29)+text(75,497,'Normalization sum ℓ = 0.5 + 1 = 1.5',29)+text(75,550,'Weighted sum u = 0.5 × 10 + 1 × 20 = 25',28);
    s+=line(757,313,757,626);
    if(n===0){
      s+=text(815,398,'m sets the exponential scale.',29)+text(815,454,'ℓ stores the total weight.',29)+text(815,510,'u stores the weighted value sum.',29)+text(815,594,'Keep (m, ℓ, u), then load the next tile.',28,700,BLUE);
    } else {
      s+=text(815,387,'New maximum m′ = ln 4',29)+text(815,434,'Rescale old state by exp(ln 2 − ln 4) = 0.5',27)+text(815,481,'New scaled weights: [0.75, 1]',28)+text(815,528,'ℓ′ = 0.5 × 1.5 + 0.75 + 1 = 2.5',28)+text(815,575,'u′ = 0.5 × 25 + 0.75 × 30 + 1 × 40 = 75',27);
      if(n===2)s+=text(815,625,'Final output = u′ / ℓ′ = 75 / 2.5 = 30',30,700,TEAL);
    }
    const scene=document.querySelector('#softmax-scene');
    scene.innerHTML=s;
    scene.closest('.slide').dataset.exampleStep=String(n);
  }
  function update(kind){
    ({generation,flash,softmax})[kind]();
    document.querySelector(`[data-step-prev="${kind}"]`).disabled=state[kind]===0;
    document.querySelector(`[data-step-next="${kind}"]`).disabled=state[kind]===max[kind];
    document.querySelector(`[data-step-label="${kind}"]`).textContent=labels[kind][state[kind]];
  }
  for(const kind of Object.keys(state)){
    document.querySelector(`[data-step-prev="${kind}"]`).addEventListener('click',()=>{state[kind]=Math.max(0,state[kind]-1);update(kind);});
    document.querySelector(`[data-step-next="${kind}"]`).addEventListener('click',()=>{state[kind]=Math.min(max[kind],state[kind]+1);update(kind);});
    update(kind);
  }

  const toy=[1,2,4,8,16,32,64].map(batch_size=>{
    const ms=4+.5*batch_size;
    return {batch_size,decode_step_ms_p50:ms,per_user_tokens_s:1000/ms,aggregate_output_tokens_s:1000*batch_size/ms};
  });
  let data=toy, dataLabel='Illustrative model, not measurements', imported=false;
  const sel=document.querySelector('#batch-select');
  function populate(){
    sel.replaceChildren(...data.map(r=>{const o=document.createElement('option');o.value=String(r.batch_size);o.textContent=String(r.batch_size);return o;}));
    sel.value=String(data[Math.min(5,data.length-1)].batch_size);chart();
  }
  const fmt=x=>x>=10000?Math.round(x).toLocaleString('en-US'):x.toLocaleString('en-US',{maximumFractionDigits:1});
  function nice(x){const p=Math.pow(10,Math.floor(Math.log10(x)));return [1,2,2.5,5,10].find(v=>v>=x/p)*p;}
  function chart(){
    const selected=data.find(r=>String(r.batch_size)===sel.value)||data[0];
    const x0=175,y0=630,w=745,h=340,xmax=nice(Math.max(...data.map(r=>r.per_user_tokens_s))*1.1),ymax=nice(Math.max(...data.map(r=>r.aggregate_output_tokens_s))*1.1);
    const px=r=>x0+w*r.per_user_tokens_s/xmax,py=r=>y0-h*r.aggregate_output_tokens_s/ymax;
    let s=text(75,256,'Total output tokens/s',27,700)+text(550,708,'Tokens/s per user (faster to the right)',27,600,INK,'middle');
    for(let i=0;i<=5;i++){
      const xx=x0+w*i/5,yy=y0-h*i/5;
      s+=line(xx,y0,xx,y0-h,'#e3e7ea')+text(xx,y0+34,fmt(xmax*i/5),23,400,MUTED,'middle');
      s+=line(x0,yy,x0+w,yy,'#e3e7ea')+text(x0-15,yy+8,fmt(ymax*i/5),23,400,MUTED,'end');
    }
    s+=line(x0,y0,x0+w,y0,INK,2)+line(x0,y0,x0,y0-h,INK,2);
    s+=`<path d="${data.map((r,i)=>`${i?'L':'M'}${px(r)},${py(r)}`).join(' ')}" stroke="${BLUE}" stroke-width="3" fill="none"/>`;
    for(const r of data)s+=`<circle cx="${px(r)}" cy="${py(r)}" r="${r===selected?9:5}" fill="${r===selected?TEAL:BLUE}"/>`;
    s+=text(px(selected)+14,py(selected)-15,`B = ${selected.batch_size}`,26,700,TEAL);
    s+=line(995,273,995,708)+text(1040,315,`Batch ${selected.batch_size}`,36,700,BLUE)+text(1040,378,`${fmt(selected.per_user_tokens_s)} tokens/s per user`,28)+text(1040,428,`${fmt(selected.aggregate_output_tokens_s)} tokens/s in total`,28)+text(1040,478,`${fmt(selected.decode_step_ms_p50)} ms per decode step`,27);
    s+=text(1040,543,imported?'Imported fixed-batch measurements':'Toy model: t = 4 + 0.5B ms',25,600,MUTED)+text(1040,580,imported?'See the run metadata for hardware.':'B is batch size, t is step duration.',24,400,MUTED);
    const scene=document.querySelector('#batch-chart');
    scene.innerHTML=s;
    document.querySelector('#chart-data-source').textContent=dataLabel;
    scene.closest('.slide').dataset.dataMode=imported?'measured':'illustrative';
  }
  sel.addEventListener('change',chart);
  populate();
  // RFC-style quoted CSV fields, including escaped quotes and embedded newlines.
  function csvRows(raw){
    raw=raw.replace(/^\uFEFF/,'');
    const rows=[];let row=[],field='',quoted=false;
    for(let i=0;i<raw.length;i++){
      const c=raw[i];
      if(c==='"'){
        if(quoted&&raw[i+1]==='"'){field+='"';i++;}else quoted=!quoted;
      } else if(c===','&&!quoted){row.push(field);field='';}
      else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&raw[i+1]==='\n')i++;row.push(field);if(row.some(v=>v.trim()))rows.push(row);row=[];field='';}
      else field+=c;
    }
    if(quoted)throw Error('CSV contains an unclosed quote.');
    row.push(field);if(row.some(v=>v.trim()))rows.push(row);return rows;
  }
  function parseData(raw){
    const rows=csvRows(raw);if(rows.length<2)throw Error('CSV needs a header and at least one result.');
    const head=rows.shift().map(x=>x.trim());
    const required=['batch_size','decode_step_ms_p50','per_user_tokens_s','aggregate_output_tokens_s'];
    for(const k of required)if(!head.includes(k))throw Error(`Missing CSV column: ${k}`);
    const out=[], seen=new Set(), lengths=new Set();
    for(const values of rows){
      const r=Object.fromEntries(head.map((k,i)=>[k,(values[i]||'').trim()]));
      if(r.status&&r.status!=='ok')continue;
      const z=Object.fromEntries(required.map(k=>[k,Number(r[k])]));
      if(required.some(k=>!r[k]||!Number.isFinite(z[k])||z[k]<=0)||!Number.isInteger(z.batch_size))throw Error('Successful rows need positive, finite timings, rates, and integer batch sizes.');
      if(seen.has(z.batch_size))throw Error('Use one result per batch size in a single CSV.');
      if(Math.abs(z.aggregate_output_tokens_s/z.per_user_tokens_s-z.batch_size)>.015*z.batch_size)throw Error('Total rate must equal batch size × per-user rate.');
      if(Math.abs(z.per_user_tokens_s*z.decode_step_ms_p50/1000-1)>.015)throw Error('Per-user rate must match the decode-step duration.');
      seen.add(z.batch_size);lengths.add(`${r.prompt_tokens||''}/${r.output_tokens||''}`);out.push(z);
    }
    if(!out.length)throw Error('No successful result rows found.');
    if(lengths.size>1)throw Error('Use a fixed prompt and output length for this batch-size comparison.');
    return out.sort((a,b)=>a.batch_size-b.batch_size);
  }
  function importData(raw,name='benchmark.csv'){
    const next=parseData(raw);data=next;imported=true;
    dataLabel=`Measurements: ${name.length>30?name.slice(0,27)+'…':name}`;
    populate();
    document.querySelector('#import-status').textContent=`Loaded ${data.length} successful batch sizes. The batch-size chart now shows measurements from your CSV. Nothing was uploaded.`;
  }
  const input=document.querySelector('#csv-file');
  document.querySelector('#import-button').addEventListener('click',()=>input.click());
  input.addEventListener('change',async()=>{
    const file=input.files[0];if(!file)return;
    try {if(file.size>2_000_000)throw Error('Please choose a CSV smaller than 2 MB.');importData(await file.text(),file.name);}
    catch(err){document.querySelector('#import-status').textContent=`Could not import CSV: ${err.message} The current chart is unchanged.`;}
    input.value='';
  });
  document.querySelector('#reset-data').addEventListener('click',()=>{
    data=toy;imported=false;dataLabel='Illustrative model, not measurements';populate();
    document.querySelector('#import-status').textContent='No GPU measurements loaded. The batch-size chart currently shows an illustrative model.';
  });
  let saved;
  window.addEventListener('beforeprint',()=>{saved={...state};for(const k of Object.keys(state)){state[k]=max[k];update(k);}});
  window.addEventListener('afterprint',()=>{if(saved){Object.assign(state,saved);for(const k of Object.keys(state))update(k);saved=null;}});
  // Numerical contract exposed for local validation, not needed for presentation.
  window.week2Examples={parseData,importData,softmaxResult:()=>{
    let m=-Infinity,l=0,u=0;
    for(const tile of [[1,2],[3,4]]){
      const next=Math.max(m,...tile.map(Math.log)),a=Math.exp(m-next);
      l*=a;u*=a;
      for(const i of tile){const p=Math.exp(Math.log(i)-next);l+=p;u+=p*(10*i);}
      m=next;
    }
    return {m,l,u,output:u/l};
  }};
})();
