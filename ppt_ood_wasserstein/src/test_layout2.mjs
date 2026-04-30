import { Presentation, createPresentationLayoutExportBlob, composeSlideDetailed, text, fill, hug, column } from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
const root=column({width:fill,height:fill},[text('x',{name:'txt',width:fill,height:hug,style:{fontSize:64}})]); const opts={frame:{left:0,top:0,width:1920,height:1080},baseUnit:8};
const detailed=composeSlideDetailed(s, root, opts);
console.log(Object.keys(detailed), detailed.run && Object.keys(detailed.run));
const blob=createPresentationLayoutExportBlob(s, [detailed.run]);
console.log(blob.type, blob.size);
await blob.save('scratch/test.layout.json');
