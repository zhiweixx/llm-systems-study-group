/* Stepwise teaching diagrams. Each SVG frame is also a complete static figure. */
(() => {
  function init(){
    const registry=new Map();
    for(const el of document.querySelectorAll('[data-sequence]')){
      const name=el.dataset.sequence;
      if(!registry.has(name))registry.set(name,{frames:[],index:0});
      registry.get(name).frames.push(el);
    }
    function show(name,index){
      const item=registry.get(name);if(!item)return;
      item.index=Math.max(0,Math.min(item.frames.length-1,index));
      item.frames.forEach((frame,i)=>{frame.style.display=i===item.index?'inline':'none';});
      document.querySelectorAll('[data-step-prev]').forEach(b=>{if(b.dataset.stepPrev===name)b.disabled=item.index===0;});
      document.querySelectorAll('[data-step-next]').forEach(b=>{if(b.dataset.stepNext===name)b.disabled=item.index===item.frames.length-1;});
      document.querySelectorAll('[data-step-label]').forEach(b=>{if(b.dataset.stepLabel===name)b.textContent=`Step ${item.index+1} / ${item.frames.length}`;});
      item.frames[0].closest('.slide').dataset.exampleStep=String(item.index);
    }
    for(const [name,item] of registry){
      item.frames.sort((a,b)=>Number(a.dataset.frame)-Number(b.dataset.frame));show(name,0);
    }
    document.querySelectorAll('[data-step-prev]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.stepPrev,registry.get(b.dataset.stepPrev).index-1)));
    document.querySelectorAll('[data-step-next]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.stepNext,registry.get(b.dataset.stepNext).index+1)));
    let saved;
    window.addEventListener('beforeprint',()=>{saved=new Map([...registry].map(([n,s])=>[n,s.index]));for(const [n,s] of registry)show(n,s.frames.length-1);});
    window.addEventListener('afterprint',()=>{if(saved)for(const [n,i] of saved)show(n,i);saved=null;});
    window.week4Examples=Object.freeze({show,state:()=>Object.fromEntries([...registry].map(([n,s])=>[n,{index:s.index,count:s.frames.length}]))});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();

/* Explicitly hypothetical timings; compare the same number of emitted tokens. */
(() => {
  function init(){
    const accepted=document.getElementById('spec-accepted');
    const verify=document.getElementById('spec-verify');
    if(!accepted||!verify)return;
    let current,saved;
    const node=key=>document.querySelector(`[data-spec-cost="${key}"]`);
    const label=(key,value)=>{node(key).textContent=value;};
    function show(a,v){
      a=Math.max(0,Math.min(4,Math.round(Number(a))));
      v=Math.max(10,Math.min(34,Math.round(Number(v)/2)*2));
      accepted.value=String(a);verify.value=String(v);
      const emitted=a+1,baseline=emitted*10,total=4+v+2,speedup=baseline/total;
      current={accepted:a,verify:v,emitted,baseline,total,speedup};
      const extra=a===4?'bonus':'correction';
      document.getElementById('spec-accepted-value').textContent=a;
      document.getElementById('spec-verify-value').textContent=v;
      label('description',`${a} accepted + 1 ${extra}: compare time for ${emitted} output token${emitted===1?'':'s'}.`);
      node('baseline-bar').setAttribute('width',baseline*20);
      node('baseline-label').setAttribute('x',390+baseline*20+25);
      label('baseline-label',`${baseline} ms`);
      node('verify-bar').setAttribute('width',v*20);
      node('verify-label').setAttribute('x',470+v*10);
      label('verify-label',`${v} ms`);
      node('overhead-bar').setAttribute('x',470+v*20);
      node('total-label').setAttribute('x',535+v*20);
      label('total-label',`${total} ms total`);
      const outcome=speedup<1?`${speedup.toFixed(2)}× → ${(1/speedup).toFixed(2)}× the time`:speedup===1?'1.00× — break-even':`${speedup.toFixed(2)}× speedup`;
      label('ratio',`${emitted} × 10 / (4 + ${v} + 2) = ${outcome}`);
      node('ratio').setAttribute('fill',speedup<1?'#a36330':'#245675');
      document.getElementById('spec-cost-explanation').textContent=speedup<1
        ?`Below 1×: speculation is slower (${total} ms versus ${baseline} ms). Draft and verification work outweigh the saved calls.`
        :speedup===1?'Break-even: the extra work exactly offsets the target calls saved.'
        :`Above 1×: speculation saves ${baseline-total} ms for the same ${emitted} output tokens. These are illustrative timings.`;
    }
    accepted.addEventListener('input',()=>show(accepted.value,verify.value));
    verify.addEventListener('input',()=>show(accepted.value,verify.value));
    window.addEventListener('beforeprint',()=>{saved={...current};show(2,14);});
    window.addEventListener('afterprint',()=>{if(saved)show(saved.accepted,saved.verify);saved=null;});
    window.specCost=Object.freeze({set:show,state:()=>({...current})});
    show(2,14);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
