import { Presentation, PresentationFile, column, text, fill, hug } from "@oai/artifact-tool";
const p=Presentation.create({slideSize:{width:1920,height:1080}});
const s=p.slides.add();
s.compose(column({width:fill,height:fill,padding:72,gap:20},[text('Hello',{width:fill,height:hug,style:{fontSize:64,bold:true,color:'#111827'}})]),{frame:{left:0,top:0,width:1920,height:1080},baseUnit:8});
const blob=await PresentationFile.exportPptx(p); await blob.save('output/test.pptx');
