import{g as J,s as K,a as Q,b as Y,t as tt,q as et,_ as s,l as w,c as at,H as it,L as rt,M as st,N as L,O as nt,e as ot,A as lt,P as ct,I as dt}from"./theme.Bk1eP5nF.js";import{p as pt}from"./chunk-4BX2VUAB.BneMh6xf.js";import{p as gt}from"./wardley-RL74JXVD.D5SjrHWM.js";import"./framework.BSym9njW.js";import"./min.Dy3BFpP6.js";import"./baseUniq.C3S_EolG.js";var ht=dt.pie,C={sections:new Map,showData:!1},u=C.sections,D=C.showData,ut=structuredClone(ht),ft=s(()=>structuredClone(ut),"getConfig"),mt=s(()=>{u=new Map,D=C.showData,lt()},"clear"),vt=s(({label:t,value:a})=>{if(a<0)throw new Error(`"${t}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);u.has(t)||(u.set(t,a),w.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),xt=s(()=>u,"getSections"),St=s(t=>{D=t},"setShowData"),wt=s(()=>D,"getShowData"),G={getConfig:ft,clear:mt,setDiagramTitle:et,getDiagramTitle:tt,setAccTitle:Y,getAccTitle:Q,setAccDescription:K,getAccDescription:J,addSection:vt,getSections:xt,setShowData:St,getShowData:wt},Ct=s((t,a)=>{pt(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),Dt={parse:s(async t=>{const a=await gt("pie",t);w.debug(a),Ct(a,G)},"parse")},$t=s(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),yt=$t,Tt=s(t=>{const a=[...t.values()].reduce((r,o)=>r+o,0),$=[...t.entries()].map(([r,o])=>({label:r,value:o})).filter(r=>r.value/a*100>=1);return ct().value(r=>r.value).sort(null)($)},"createPieArcs"),At=s((t,a,$,y)=>{var z;w.debug(`rendering pie chart
`+t);const r=y.db,o=at(),T=it(r.getConfig(),o.pie),A=40,n=18,p=4,c=450,d=c,f=rt(a),l=f.append("g");l.attr("transform","translate("+d/2+","+c/2+")");const{themeVariables:i}=o;let[b]=st(i.pieOuterStrokeWidth);b??(b=2);const _=T.textPosition,g=Math.min(d,c)/2-A,O=L().innerRadius(0).outerRadius(g),P=L().innerRadius(g*_).outerRadius(g*_);l.append("circle").attr("cx",0).attr("cy",0).attr("r",g+b/2).attr("class","pieOuterCircle");const h=r.getSections(),B=Tt(h),I=[i.pie1,i.pie2,i.pie3,i.pie4,i.pie5,i.pie6,i.pie7,i.pie8,i.pie9,i.pie10,i.pie11,i.pie12];let m=0;h.forEach(e=>{m+=e});const E=B.filter(e=>(e.data.value/m*100).toFixed(0)!=="0"),v=nt(I).domain([...h.keys()]);l.selectAll("mySlices").data(E).enter().append("path").attr("d",O).attr("fill",e=>v(e.data.label)).attr("class","pieCircle"),l.selectAll("mySlices").data(E).enter().append("text").text(e=>(e.data.value/m*100).toFixed(0)+"%").attr("transform",e=>"translate("+P.centroid(e)+")").style("text-anchor","middle").attr("class","slice");const N=l.append("text").text(r.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText"),k=[...h.entries()].map(([e,S])=>({label:e,value:S})),x=l.selectAll(".legend").data(k).enter().append("g").attr("class","legend").attr("transform",(e,S)=>{const F=n+p,X=F*k.length/2,Z=12*n,j=S*F-X;return"translate("+Z+","+j+")"});x.append("rect").attr("width",n).attr("height",n).style("fill",e=>v(e.label)).style("stroke",e=>v(e.label)),x.append("text").attr("x",n+p).attr("y",n-p).text(e=>r.getShowData()?`${e.label} [${e.value}]`:e.label);const U=Math.max(...x.selectAll("text").nodes().map(e=>(e==null?void 0:e.getBoundingClientRect().width)??0)),q=d+A+n+p+U,R=((z=N.node())==null?void 0:z.getBoundingClientRect().width)??0,H=d/2-R/2,V=d/2+R/2,M=Math.min(0,H),W=Math.max(q,V)-M;f.attr("viewBox",`${M} 0 ${W} ${c}`),ot(f,c,W,T.useMaxWidth)},"draw"),bt={draw:At},Ft={parser:Dt,db:G,renderer:bt,styles:yt};export{Ft as diagram};
