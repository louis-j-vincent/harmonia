// Replicate the app_shell bar-time-index + hitAtTime logic on a synthetic model.
// bars 0..3: chorded(0-2s), HELD(2-4s), chorded(4-6s), N.C.(6-8s)
const bars=[
  {idxs:[0], t0:0, t1:2},
  {idxs:[],  t0:2, t1:4},   // held (%): must still highlight
  {idxs:[1], t0:4, t1:6},
  {idxs:[2], t0:6, t1:8},   // N.C. bar (a chord with nc, but still has time)
];
const chords=[{spans:[[0,2]],t0:0,t1:2},{spans:[[4,6]],t0:4,t1:6},{spans:[[6,8]],t0:6,t1:8}];
// build tspans + barTimeIndex (copy of loadModel logic)
for(let bi=0;bi<bars.length;bi++){const b=bars[bi];
  if(b.idxs.length){const nrep=Math.max(1,...b.idxs.map(i=>(chords[i].spans||[]).length||1));b.tspans=[];
    for(let r=0;r<nrep;r++){let t0=Infinity,t1=-Infinity;
      b.idxs.forEach(i=>{const sp=(chords[i].spans&&(chords[i].spans[r]||chords[i].spans[0]))||[chords[i].t0,chords[i].t1];t0=Math.min(t0,sp[0]);t1=Math.max(t1,sp[1]);});
      b.tspans.push([t0,t1]);}}
  else b.tspans=[[b.t0,b.t1]];}
const barTimeIndex=[];
bars.forEach((b,bi)=>b.tspans.forEach((sp,rep)=>barTimeIndex.push({bi,t0:sp[0],t1:sp[1],rep})));
barTimeIndex.sort((a,b)=>a.t0-b.t0);
function hitAtTime(t){let lo=0,hi=barTimeIndex.length-1,best=null;
  while(lo<=hi){const mid=(lo+hi)>>1,e=barTimeIndex[mid];if(t<e.t0)hi=mid-1;else{best=e;lo=mid+1;}}
  if(best&&t<best.t1+0.001)return best;return best&&t>=best.t0?best:null;}
let ok=true;
[[1,0],[3,1],[5,2],[7,3],[2.0,1],[3.99,1]].forEach(([t,exp])=>{
  const h=hitAtTime(t);const got=h?h.bi:-1;const pass=got===exp;ok=ok&&pass;
  console.log(`t=${t}s -> bar ${got} (expect ${exp}) ${pass?'OK':'FAIL'}`);
});
console.log(ok?'ALL PASS — held bar (bar 1) and N.C. bar (bar 3) highlight by time':'FAIL');
