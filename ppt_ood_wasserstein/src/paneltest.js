import { Presentation, PresentationFile, column, panel, text, fill, hug, fixed } from "@oai/artifact-tool";
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
s.compose(column({width:fill,height:fill,padding:72,gap:20},[
 panel({name:'p',width:fixed(500),height:hug,padding:{x:30,y:20},background:'#F1F5F9',borderRadius:24}, text('Panel text',{width:fill,height:hug,style:{fontSize:32,color:'#111827'}}))
]),{frame:{left:0,top:0,width:1920,height:1080},baseUnit:8});
const blob=await PresentationFile.exportPptx(p); await blob.save('output/panel.pptx');
