import { Presentation, text, fill, hug, column } from '@oai/artifact-tool';
import { writeFile } from 'node:fs/promises';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
s.compose(column({width:fill,height:fill},[text('x',{name:'txt',width:fill,height:hug,style:{fontSize:64}})]),{frame:{left:0,top:0,width:1920,height:1080},baseUnit:8});
const lb=await s.export({format:'layout'}); console.log('layout', lb.constructor.name, typeof lb.save, lb.type, lb.size); await writeFile('scratch/export.layout.json', Buffer.from(await lb.arrayBuffer()));
try { const pb=await s.export({format:'png'}); console.log('png', pb.constructor.name, typeof pb.save, pb.type, pb.size); if (pb.save) await pb.save('scratch/export.png'); else await writeFile('scratch/export.png', Buffer.from(await pb.arrayBuffer())); } catch(e) { console.error('PNGERR', e.stack); process.exitCode=2; }
