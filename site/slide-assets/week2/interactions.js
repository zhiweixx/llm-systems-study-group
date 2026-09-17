/* Offline generation and paged-allocation teaching examples. */
(() => {
  'use strict';
  const BLUE='#245675', INK='#172329', MUTED='#58656d', PALE='#edf3f7', LINE='#aec3d1', TEAL='#397b71';
  const e=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const text=(x,y,v,size=28,weight=400,color=INK,anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}">${e(v)}</text>`;
  const rect=(x,y,w,h,fill='white',stroke=LINE)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
  const box=(x,y,w,h,v,fill=PALE,size=27)=>rect(x,y,w,h,fill)+text(x+w/2,y+h/2+size*.34,v,size,600,INK,'middle');
  const line=(x1,y1,x2,y2,color=LINE,width=1.5)=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}"/>`;
  const arrow=(x1,y,x2,color=BLUE)=>line(x1,y,x2,y,color,2)+`<path d="M${x2-9},${y-6} L${x2},${y} L${x2-9},${y+6}" fill="none" stroke="${color}" stroke-width="2"/>`;
  const state={generation:0,paging:0};
  const max={generation:2,paging:4};
  const labels={
    generation:['Prefill complete: first token predicted','Decode 1 complete: second token predicted','Decode 2 complete: third token predicted'],
    paging:['1 / 5: A has 6 cached tokens','2 / 5: append to 8 tokens; fill P1','3 / 5: append to 9 tokens; allocate P5','4 / 5: A finishes; reclaim its blocks','5 / 5: B arrives; reuse P4']
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
    s+=text(435,534,'Stored K/V for these processed positions',23,400,MUTED);
    for(let i=0;i<6;i++)s+=`<g opacity="${i<4+n?1:.18}">`+box(435+i*115,549,104,65,words[i],i<4?PALE:'#dcece6',22)+'</g>';
    s+=text(1160,569,'The last predicted token',24)+text(1160,607,'has not entered the cache yet.',24);
    const scene=document.querySelector('#generation-scene');
    scene.innerHTML=s;
    scene.closest('.slide').dataset.exampleStep=String(n);
  }
  // Four slots per block is a teaching choice, not an implementation default.
  // Null slots are capacity, never valid KV entries for an attention read.
  function pagingState(){
    const phase=state.paging, owner=phase===3?null:phase===4?'B':'A';
    const tokens=[6,8,9,0,3][phase], mapping=phase===3?[]:phase===4?[4]:phase===2?[4,1,5]:[4,1];
    const pool=Array.from({length:6},(_,id)=>({id,owner:null,logicalBlock:null,slots:Array(4).fill(null)}));
    const assign=(request,tokenCount,table)=>table.forEach((id,logicalBlock)=>{
      pool[id]={id,owner:request,logicalBlock,slots:Array.from({length:4},(_,slot)=>{
        const token=logicalBlock*4+slot+1;
        return token<=tokenCount?`${request}${token}`:null;
      })};
    });
    assign('C',8,[0,2]);
    if(owner)assign(owner,tokens,mapping);
    return {phase,owner,tokens,mapping,pool,freeBlocks:pool.filter(b=>b.owner===null).map(b=>b.id),blockSize:4};
  }
  function paging(){
    const p=pagingState(), ownerFill=p.owner==='B'?'#dcece6':PALE;
    let s=text(75,260,p.owner?`${p.owner}: ${p.tokens} valid cached tokens`:'A has finished',29,700,BLUE);
    s+=text(815,260,'Physical KV pool',29,700,BLUE)+text(815,289,'Toy: 4 token slots per block · no prefix sharing',24,400,MUTED);
    if(p.owner){
      s+=text(75,297,'Logical blocks',26,700)+text(460,297,'Block table',26,700);
      p.mapping.forEach((id,logicalBlock)=>{
        const y=315+logicalBlock*78, b=p.pool[id];
        s+=rect(75,y,335,66,'white')+text(90,y+42,`L${logicalBlock}`,26,700,BLUE);
        b.slots.forEach((token,slot)=>s+=box(147+slot*61,y+12,56,42,token||'—',token?ownerFill:'white',25));
        s+=arrow(416,y+33,451)+box(460,y,285,66,`L${logicalBlock} → P${id}`,ownerFill,29);
      });
    } else {
      s+=rect(75,315,670,222,'white')+text(100,360,'A’s block table is released.',29,700);
      s+=text(100,410,'P4, P1 and P5 are free again.',28)+text(100,460,'No attention reads remain for A.',26,400,MUTED);
      s+=text(100,510,'C’s table and KV stay in place.',26,400,MUTED);
    }
    p.pool.forEach(b=>{
      const x=815+(b.id%2)*355,y=299+Math.floor(b.id/2)*82;
      const active=b.owner!==null&&b.owner===p.owner;
      const fill=active?ownerFill:b.owner?'#f4f5f6':'white';
      s+=`<g data-physical-block="${b.id}" data-owner="${b.owner||'free'}">`;
      s+=rect(x,y,340,76,fill,active?BLUE:LINE)+text(x+14,y+29,`P${b.id} · ${b.owner||'free'}`,26,700,active?BLUE:MUTED);
      if(b.owner)s+=text(x+326,y+29,`${b.slots.filter(Boolean).length}/4 valid`,24,400,MUTED,'end');
      b.slots.forEach((token,slot)=>{
        s+=box(x+14+slot*80,y+39,72,35,token||(b.owner?'—':''),'white',24);
      });
      s+='</g>';
    });
    const action=[
      'A occupies P4 and P1; the last block has two reserved slots.',
      'A7–A8 fill P1. Existing KV stays in place; no new block is allocated.',
      'A9 allocates P5 for L2. Existing KV stays in P4 and P1; no copying.',
      'Request completion returns A’s blocks to the free pool; C keeps P0 and P2.',
      'B reuses freed P4 for its own KV. C’s blocks remain unchanged.'
    ][p.phase];
    if(p.owner){
      const readBlocks=p.mapping.map(id=>{
        const valid=p.pool[id].slots.filter(Boolean);
        return `P${id} (${valid.length===1?valid[0]:`${valid[0]}–${valid[valid.length-1]}`})`;
      }).join(' → ');
      s+=text(75,566,`Read table [${p.mapping.join(', ')}] → ${readBlocks}.`,25,700,BLUE);
      s+=text(75,630,`Read only ${p.tokens} valid positions in logical order; “—” slots never participate in attention.`,25,400,MUTED);
    } else {
      s+=text(75,566,'No table for A: its former blocks can now hold another request’s KV.',26,700,BLUE);
      s+=text(75,630,'C still reads its own K/V through table [0, 2].',25,400,MUTED);
    }
    s+=text(75,600,action,25);
    const scene=document.querySelector('#paging-scene');
    scene.innerHTML=s;
    scene.closest('.slide').dataset.exampleStep=String(p.phase);
  }
  function update(kind){
    ({generation,paging})[kind]();
    document.querySelector(`[data-step-prev="${kind}"]`).disabled=state[kind]===0;
    document.querySelector(`[data-step-next="${kind}"]`).disabled=state[kind]===max[kind];
    document.querySelector(`[data-step-label="${kind}"]`).textContent=labels[kind][state[kind]];
  }
  for(const kind of Object.keys(state)){
    document.querySelector(`[data-step-prev="${kind}"]`).addEventListener('click',()=>{state[kind]=Math.max(0,state[kind]-1);update(kind);});
    document.querySelector(`[data-step-next="${kind}"]`).addEventListener('click',()=>{state[kind]=Math.min(max[kind],state[kind]+1);update(kind);});
    update(kind);
  }

  let saved;
  window.addEventListener('beforeprint',()=>{saved={...state};for(const k of Object.keys(state)){state[k]=max[k];update(k);}});
  window.addEventListener('afterprint',()=>{if(saved){Object.assign(state,saved);for(const k of Object.keys(state))update(k);saved=null;}});
  window.week2Examples={pagingState};
})();
