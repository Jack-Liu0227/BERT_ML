import { PresentationFile } from '@oai/artifact-tool';
import { readFile, writeFile } from 'node:fs/promises';
const bytes = await readFile('output/ood_wasserstein_formula_slide.pptx');
const p = await PresentationFile.importPptx(bytes);
console.log('slides', p.slides.count);
const blob = await p.export({ slide: p.slides.getItem(0), format: 'png' });
await writeFile('scratch/ood_wasserstein_formula_slide.pptx_rerender.png', Buffer.from(await blob.arrayBuffer()));
const insp = await p.inspect({kind:'textbox,shape,slide', maxChars: 5000});
console.log(insp.ndjson.slice(0,1000));
