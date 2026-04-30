import {PresentationFile, Presentation} from '@oai/artifact-tool';
console.log(Object.getOwnPropertyNames(PresentationFile));
console.log(Object.getOwnPropertyNames(Presentation));
const p=Presentation.create({slideSize:{width:1920,height:1080}}); console.log('p methods', Object.getOwnPropertyNames(Object.getPrototypeOf(p)).filter(x=>x.includes('export')||x.includes('inspect')||x.includes('load')).join('\n'));
