import {Presentation} from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); console.log(p.export.toString().slice(0,1500)); console.log(p.inspect.toString().slice(0,500));
