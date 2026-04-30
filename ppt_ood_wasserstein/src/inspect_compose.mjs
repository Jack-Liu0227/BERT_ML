import { Presentation, text, fill, hug, column } from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
const r=s.compose(column({width:fill,height:fill},[text('x',{width:fill,height:hug})]),{frame:{left:0,top:0,width:1920,height:1080},baseUnit:8});
console.log('return', r); console.log('symbols', Object.getOwnPropertySymbols(s));
console.log('proto keys', Object.keys(s.toProto()));
