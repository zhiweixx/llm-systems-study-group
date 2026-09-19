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
    window.week3Examples=Object.freeze({show,state:()=>Object.fromEntries([...registry].map(([n,s])=>[n,{index:s.index,count:s.frames.length}]))});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
