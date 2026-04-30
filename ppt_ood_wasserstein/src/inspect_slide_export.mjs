import {Presentation} from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add(); console.log(s.export.toString().slice(0,1500));
