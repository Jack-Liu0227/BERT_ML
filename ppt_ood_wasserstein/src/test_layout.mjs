import { Presentation, createPresentationLayoutExportBlob, text, fill, hug, column } from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
const root=column({width:fill,height:fill},[text('x',{name:'txt',width:fill,height:hug,style:{fontSize:64}})]); const opts={frame:{left:0,top:0,width:1920,height:1080},baseUnit:8};
s.compose(root, opts);
const blob=createPresentationLayoutExportBlob(s, [{root, ...opts}]);
console.log(blob.type, blob.size);
await blob.save('scratch/test.layout.json');
