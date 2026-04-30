import { Presentation, text, fill, hug, column } from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
s.compose(column({width:fill,height:fill},[text('x',{width:fill,height:hug})]),{frame:{left:0,top:0,width:1920,height:1080},baseUnit:8});
console.log(Object.keys(s));
console.log(s.constructor.name); console.log(typeof s.getPresentation, typeof p.getPresentation);
console.log('composeRuns', s.composeRuns, s._composeRuns, s.compositionRuns);
for (const k of Object.keys(s)) if (k.toLowerCase().includes('compose')) console.log(k, s[k]);
