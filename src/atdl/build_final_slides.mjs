import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BUILD = path.join(ROOT, '.presentation-build');
const FINAL = path.join(ROOT, 'reports/ATDL_Final_Project_Results.pptx');
const STAGED_FINAL = path.join(ROOT, 'reports/.pptx-finalization', `ATDL_Final_Project_Results_${Date.now()}.pptx`);
const FONT = 'DejaVu Sans';
const W = 1280, H = 720;
const C = { navy:'#12263A', ink:'#1C2938', muted:'#526477', teal:'#087E8B', aqua:'#DDF2F1', orange:'#D45B36', pale:'#F3F6F8', white:'#FFFFFF', line:'#C9D3DC', green:'#2F855A' };

const json = async p => JSON.parse(await fs.readFile(path.join(ROOT,p),'utf8'));
const csv = async p => {
  const txt = await fs.readFile(path.join(ROOT,p),'utf8');
  const lines = txt.trim().split(/\r?\n/);
  const parse = line => {
    const out=[]; let cur='', quote=false;
    for(let i=0;i<line.length;i++){ const ch=line[i]; if(ch==='"'){ if(quote&&line[i+1]==='"'){cur+='"';i++;} else quote=!quote; } else if(ch===','&&!quote){out.push(cur);cur='';} else cur+=ch; }
    out.push(cur); return out;
  };
  const headers=parse(lines[0]);
  return lines.slice(1).map(line=>Object.fromEntries(parse(line).map((v,i)=>[headers[i],v])));
};
const fmt = (x,n=3) => Number.isFinite(Number(x)) ? Number(x).toFixed(n) : '—';
// Native charts only need six decimal places; preserve full precision in reports/JSON.
const chartValue = x => Number(Number(x).toFixed(6));
const metric = (r,k) => Number(r?.global_metrics?.[k] ?? 0);

const architecture = await json('results/comprehensive/model_transfer_runs.json');
const interventions = await json('results/comprehensive/full_data_intervention_runs.json');
const comparison = await csv('results/comprehensive/candidate_comparison.csv');
const archSummary = await csv('results/comprehensive/model_transfer_summary.csv');
const interventionSummary = await csv('results/comprehensive/full_data_intervention_summary.csv');
const resolution = await json('results/comprehensive/resolution_ablation_summary.json');
const selection = await json('results/comprehensive/final_model_selection.json');
const test = await json(`results/comprehensive/test_evaluations/${selection.selected_candidate_group}/seed_${selection.selected_seed}_test_metrics.json`);
const phase2 = await json('results/phase2/phase2_dataset_summary.json');
const split = await csv('results/phase2/split_class_distribution.csv');
const duplicateRows = await csv('results/tables/exact_duplicate_yolo_label_rows.csv');
const baseTest = await json('results/final_test_metrics.json');

const pres = Presentation.create({ slideSize:{width:W,height:H} });

function shape(slide,{x,y,w,h,fill='none',stroke='none',sw=0,geometry='rect',name}) {
  return slide.shapes.add({ geometry, name, position:{left:x,top:y,width:w,height:h}, fill,
    line:{style:'solid',fill:stroke,width:sw} });
}
function text(slide,content,{x,y,w,h,size=24,color=C.ink,bold=false,align='left',valign='top',name,fill='none'}={}) {
  const s=shape(slide,{x,y,w,h,fill,name}); s.text=content;
  s.text.style={fontSize:size,typeface:FONT,color,bold,alignment:align,verticalAlignment:valign,wrap:'square',autoFit:'shrinkText',insets:0};
  return s;
}
function base(section,title,sub='') {
  const s=pres.slides.add(); s.background.fill=C.white;
  text(s,section.toUpperCase(),{x:72,y:28,w:520,h:22,size:12,color:C.teal,bold:true});
  text(s,title,{x:72,y:62,w:1136,h:56,size:33,color:C.navy,bold:true});
  if(sub) text(s,sub,{x:72,y:123,w:1120,h:44,size:17,color:C.muted});
  shape(s,{x:72,y:174,w:1136,h:2,fill:C.line});
  text(s,'ATDL Final Year Project  •  NEU-DET steel defect detection',{x:72,y:682,w:850,h:18,size:10,color:C.muted});
  text(s,String(pres.slides.items.length).padStart(2,'0'),{x:1150,y:680,w:58,h:20,size:10,color:C.muted,align:'right'});
  return s;
}
function notes(s,txt) { s.speakerNotes.text=txt; s.speakerNotes.setVisible(true); }
function image(s,relative,x,y,w,h,alt) {
  const full=path.join(ROOT,relative);
  const bytes=readFileSync(full);
  return s.images.add({blob:bytes,contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function chart(s,type,config){
  return s.charts.add(type,{...config,
    titleTextStyle:{typeface:FONT,fontSize:17,bold:true,fill:C.navy},
    chartFill:C.white,chartLine:{style:'solid',fill:C.white,width:0},
    plotAreaFill:C.white,plotAreaLine:{style:'solid',fill:C.white,width:0},
  });
}
function meanGroup(group,key){
  const rows=[...architecture,...interventions].filter(r=>(r.model_name||r.treatment)===group);
  return rows.length?rows.reduce((sum,r)=>sum+Number(r.per_class?.[key]?.ap50??0),0)/rows.length:0;
}
function groupRecords(group){return [...architecture,...interventions].filter(r=>(r.model_name||r.treatment)===group);}
function callout(s,x,y,value,label,color=C.teal){
  text(s,value,{x,y,w:220,h:54,size:36,color,bold:true});
  text(s,label,{x,y:y+56,w:220,h:44,size:15,color:C.muted});
}
function bulletList(s,items,{x=90,y=210,w=1080,size=22,step=76}={}){
  items.forEach((item,i)=>{
    text(s,'•',{x,y:y+i*step,w:28,h:34,size:size,color:C.teal,bold:true});
    text(s,item,{x:x+34,y:y+i*step,w:w-34,h:step-8,size,color:C.ink});
  });
}

const trainImages=Number(split.find(r=>r.split==='train')?.total_images??1257);
const valImages=Number(split.find(r=>r.split==='val')?.total_images??266);
const testImages=Number(split.find(r=>r.split==='test')?.total_images??276);
const totalBoxes=Number(phase2.object_count_converted??0)-duplicateRows.reduce((sum,row)=>sum+Number(row.duplicate_rows),0);
const best=comparison[0];
const baseArch=archSummary.find(r=>r.model_name==='yolov8n');
const baseTail=Number(baseArch?.minority_macro_AP50_mean??0);
const bestDelta=Number(best?.minority_macro_AP50_mean??0)-baseTail;
const m=test.global_metrics;

// 1 — Cover
{
 const s=pres.slides.add(); s.background.fill=C.navy;
 shape(s,{x:0,y:0,w:18,h:H,fill:C.teal});
 text(s,'ATDL FINAL YEAR PROJECT',{x:76,y:76,w:900,h:24,size:15,color:'#78D5D0',bold:true});
 text(s,'Steel surface defect\ndetection under class imbalance',{x:76,y:146,w:1050,h:160,size:43,color:C.white,bold:true});
 text(s,'A controlled evaluation of pretrained YOLO models and full-data tail-class interventions',{x:80,y:345,w:990,h:68,size:22,color:'#D9E4EC'});
 shape(s,{x:80,y:468,w:1120,h:1,fill:'#506579'});
 callout(s,82,508,`${trainImages.toLocaleString()} / ${valImages} / ${testImages}`,'train / validation / test images','#70D6CE');
 callout(s,430,508,totalBoxes.toLocaleString(),'unique defect boxes','#70D6CE');
 callout(s,780,508,'3 seeds','per candidate group','#70D6CE');
 text(s,'Results generated from the versioned Phase 8 experiment records',{x:82,y:653,w:900,h:22,size:12,color:'#D9E4EC'});
 notes(s,'Project dataset and experiment results: repository Phase 2 summary and Phase 8 records. The four project-defined tail classes are crazing, pitted_surface, rolled-in_scale and scratches.');
}

// 2 — Objective
{
 const s=base('Research question','Can tail-aware training improve defect detection where the data are weakest?','The primary endpoint is fixed before the final comparison.');
 bulletList(s,[
  'Primary: mean AP50 across crazing, pitted_surface, rolled-in_scale and scratches.',
  'Guardrail: overall mAP50; also report mAP50–95, per-class precision, recall and F1.',
  'Compare four COCO-pretrained YOLO variants, then three full-data imbalance interventions.',
  'Use three seeds per group; select by validation only and evaluate the selected checkpoint once on test.'
 ],{y:215,size:21,step:92});
 text(s,`Validation leader: ${best?.candidate ?? '—'}  |  tail AP50 ${fmt(best?.minority_macro_AP50_mean)} ± ${fmt(best?.minority_macro_AP50_sd)}  |  overall mAP50 ${fmt(best?.mAP50_mean)}`,{x:92,y:592,w:1080,h:52,size:18,color:C.navy,bold:true});
 notes(s,'The tail-class set follows the existing project definition used in Phases 3–5. AP50 is average precision at IoU 0.50. Literature context: Lv et al. (2020), NEU-DET benchmark, https://doi.org/10.3390/s20061562');
}

// 3 — dataset
{
 const s=base('Dataset','Fixed grouped split after duplicate cleanup','All methods share the same untouched validation and test images.');
 callout(s,86,212,trainImages.toLocaleString(),'training images');
 callout(s,390,212,String(valImages),'validation images');
 callout(s,694,212,String(testImages),'test images');
 callout(s,998,212,totalBoxes.toLocaleString(),'labeled boxes',C.orange);
 image(s,'results/comprehensive/figures/dataset_split_class_distribution.png',98,326,650,298,'Defect object counts by class and split');
 image(s,'results/comprehensive/figures/training_box_area_by_class.png',774,326,410,298,'Training box area by defect class');
 text(s,'The class/split chart shows source label rows. Three exact repeated box rows are removed by the loader, leaving the unique-box total above.',{x:98,y:626,w:1085,h:32,size:15,color:C.muted});
 notes(s,'Source: results/phase2/phase2_dataset_summary.json and split_class_distribution.csv. The 1,799 unique-image total and split counts are project-generated audit results.');
}

// 4 — protocol
{
 const s=base('Study protocol','Each stage has a defined role in model selection and evaluation.','The held-out split is only used after the checkpoint is fixed.');
 const steps=[
  ['01','Audit + split','Deduplicate, convert annotations, fixed grouped 70/15/15 split'],
  ['02','Transfer sweep','4 pretrained models × 3 seeds; train on full train split'],
  ['03','Imbalance methods','3 interventions × 3 seeds on the selected architecture'],
  ['04','Validation choice','Rank complete groups by tail macro AP50; overall mAP50 tie-break'],
  ['05','Final evaluation','One selected checkpoint, one test pass; val-calibrated thresholds']
 ];
 steps.forEach((a,i)=>{
  const y=205+i*88;
  shape(s,{x:104,y:y+3,w:54,h:54,fill:i===4?C.orange:C.teal,geometry:'ellipse'});
  text(s,a[0],{x:104,y:y+17,w:54,h:22,size:16,color:C.white,bold:true,align:'center'});
  text(s,a[1],{x:190,y:y,w:245,h:34,size:20,color:C.navy,bold:true});
  text(s,a[2],{x:450,y:y,w:720,h:52,size:18,color:C.ink});
 });
 notes(s,'Reproducible details and scripts are in notebook/atdl_phase8_comprehensive_evaluation.ipynb and src/atdl/. Candidate selection code uses validation records only.');
}

// 5 — architecture benchmark
{
 const s=base('Pretrained model comparison','Model capacity and family both affect the validation result.','All four candidates use the same fixed split, input size, batch and three seeds.');
 const names=archSummary.map(r=>r.model_name);
 chart(s,'bar',{position:{left:100,top:205,width:765,height:390},title:'Three-seed mean validation metrics',categories:names,
  series:[
   {name:'Tail macro AP50',values:archSummary.map(r=>chartValue(r.minority_macro_AP50_mean)),fill:C.teal},
   {name:'Overall mAP50',values:archSummary.map(r=>chartValue(r.mAP50_mean)),fill:C.orange}
  ],hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'column',grouping:'clustered',gapWidth:85},
  yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.00',title:'Validation score'},
  xAxis:{title:'COCO-pretrained detector'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:12,typeface:FONT}}
 });
 const rows=archSummary.map(r=>`${r.model_name.toUpperCase()}: ${fmt(r.minority_macro_AP50_mean)} ± ${fmt(r.minority_macro_AP50_sd)} tail AP50\n${fmt(r.mAP50_mean)} mAP50 • ${(Number(r.parameters_mean)/1e6).toFixed(1)}M parameters`).join('\n\n');
 text(s,rows,{x:900,y:220,w:300,h:290,size:16,color:C.ink});
 text(s,'YOLOv8n seed 42 reuses the Phase 3 run; other seeds/models were trained in Phase 8.',{x:900,y:522,w:300,h:76,size:14,color:C.muted});
 notes(s,'Sources: results/comprehensive/model_transfer_summary.csv; model training used official Ultralytics COCO-pretrained checkpoints. Model documentation: https://docs.ultralytics.com/models/ and https://docs.ultralytics.com/tasks/detect/.');
}

// 6 — input-size ablation
{
 const s=base('Compute and resolution','A 320 px input trades detail for lower inference time.','Paired seeds use the selected model family; the ablation stays separate from candidate ranking.');
 chart(s,'bar',{position:{left:110,top:220,width:710,height:360},title:`${resolution.model_name}: validation comparison`,categories:['Tail macro AP50','Overall mAP50'],
  series:[
   {name:'640 px',values:[chartValue(resolution.imgsz_640_reference.mean_tail_macro_AP50),chartValue(resolution.imgsz_640_reference.mean_mAP50)],fill:C.muted},
   {name:'320 px',values:[chartValue(resolution.imgsz_320.mean_tail_macro_AP50),chartValue(resolution.imgsz_320.mean_mAP50)],fill:C.teal}
  ],hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
  yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.00'},xAxis:{title:'Validation metric'}
 });
 const speed=resolution.paired_deltas.validation_inference_ms_per_image.mean_delta_320_minus_640;
 const tail=resolution.paired_deltas.minority_macro_ap50.mean_delta_320_minus_640;
 text(s,`320 px change vs 640 px\nTail macro AP50: ${tail>=0?'+':''}${fmt(tail)}\nInference time: ${speed>=0?'+':''}${fmt(speed,2)} ms/image\nSeeds: 42, 43, 44`,{x:875,y:257,w:300,h:205,size:21,color:C.navy,bold:true});
 text(s,'Only input resolution changes. Training data, pretrained model family, split, seed, optimizer, batch and epoch cap are matched.',{x:875,y:496,w:300,h:98,size:15,color:C.muted});
 notes(s,'Source: results/comprehensive/resolution_ablation_summary.json. Each 320 px run is paired by seed with the corresponding 640 px architecture-transfer run. This is a separate ablation and does not affect final candidate selection.');
}

// 7 — interventions
{
 const s=base('Imbalance interventions','Treatments retain the full dataset and match image exposure.','The original moderate-subset Phase 5 methods remain exploratory context.');
 const info=[
  ['Repeat sampling','All 1,257 originals + square-root inverse-frequency repeats','1,543'],
  ['Targeted augmentation','Two label-correct transforms for tail-containing images','2,937'],
  ['Combined','Repeat sampling plus both targeted transforms','3,223']
 ];
 info.forEach((r,i)=>{
   const y=210+i*115;
   shape(s,{x:95,y,w:1080,h:94,fill:i%2?C.pale:'#EFF7F7',stroke:C.line,sw:1});
   text(s,r[0],{x:120,y:y+14,w:250,h:30,size:20,color:C.navy,bold:true});
   text(s,r[1],{x:395,y:y+14,w:565,h:54,size:17,color:C.ink});
   text(s,r[2],{x:1000,y:y+18,w:140,h:42,size:27,color:C.teal,bold:true,align:'right'});
 });
 text(s,'Each run targets 62,850 train-image presentations (1,257 × 50). All treatments are evaluated on the same validation split.',{x:105,y:570,w:1060,h:46,size:18,color:C.muted});
 notes(s,'Treatment definitions and generated dataset manifest hashes are in results/comprehensive/full_data_intervention_runs.json and data/phase8_generated/*/dataset_summary.json. Repeated sampling uses an image-level maximum over project-defined tail classes; augmentation transforms labels analytically.');
}

// 7 — all candidates
{
 const s=base('Validation ranking','The winning validation group is the best observed under the pre-set objective.','Mean ± standard deviation across three seeds; no test metrics enter this ranking.');
 chart(s,'bar',{position:{left:95,top:210,width:1085,height:390},title:'Validation tail AP50 and overall mAP50',categories:comparison.map(r=>r.candidate),
  series:[
   {name:'Tail macro AP50',values:comparison.map(r=>chartValue(r.minority_macro_AP50_mean)),fill:C.teal},
   {name:'Overall mAP50',values:comparison.map(r=>chartValue(r.mAP50_mean)),fill:C.orange}
  ],hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'bar',grouping:'clustered',gapWidth:75},
  xAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.00'},
  yAxis:{title:'Candidate group'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:11,typeface:FONT}}
 });
 text(s,`Selected: ${selection.selected_candidate_group} (seed ${selection.selected_seed})  •  tail AP50 change vs YOLOv8n mean: ${bestDelta>=0?'+':''}${fmt(bestDelta)}`,{x:100,y:618,w:1080,h:34,size:17,color:C.navy,bold:true});
 notes(s,'Source: results/comprehensive/candidate_comparison.csv and final_model_selection.json. Selection rule: highest three-seed mean tail macro AP50; tie-break by mean overall mAP50. This is a validation result; three seeds are descriptive and do not establish statistical significance.');
}

// 8 — classwise
{
 const s=base('Class-wise validation','Per-class validation AP50','This view makes the tail-class trade-offs visible.');
 const classes=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches'];
 const selectedGroup=selection.selected_candidate_group;
 const baseRecords=groupRecords('yolov8n'); const selectedRecords=groupRecords(selectedGroup);
 const meanAp=(records,c)=>records.reduce((sum,r)=>sum+Number(r.per_class[c]?.ap50??0),0)/Math.max(1,records.length);
 chart(s,'bar',{position:{left:95,top:215,width:1080,height:390},title:'Mean validation AP50 by class',categories:classes,
  series:[
   {name:'YOLOv8n baseline',values:classes.map(c=>chartValue(meanAp(baseRecords,c))),fill:C.muted},
   {name:`Selected group: ${selectedGroup}`,values:classes.map(c=>chartValue(meanAp(selectedRecords,c))),fill:C.teal}
  ],hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'bar',grouping:'clustered',gapWidth:75},
  xAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.00'},yAxis:{title:'Defect category'}
 });
 text(s,'Compare each class before interpreting an aggregate gain. A class-specific decline remains important even when the macro score rises.',{x:100,y:617,w:1080,h:38,size:16,color:C.muted});
 notes(s,'Source: Phase 8 validation run JSONs. The baseline is YOLOv8n across three seeds; the selected group is the validation-ranked group across three seeds.');
}

// 9 — test summary
{
 const s=base('Final test evaluation','The selected checkpoint was evaluated once after validation selection.','Test metrics below are descriptive because the earlier Phase 6 baseline already used this split.');
 callout(s,90,222,fmt(metric(test,'metrics/mAP50(B)')),'test mAP50');
 callout(s,382,222,fmt(metric(test,'metrics/mAP50-95(B)')),'test mAP50–95');
 callout(s,674,222,fmt(metric(test,'metrics/F1(B)')),'test global F1');
 callout(s,966,222,fmt(test.minority_macro_test_ap50),'test tail macro AP50',C.orange);
 chart(s,'bar',{position:{left:110,top:390,width:1040,height:250},title:'Selected checkpoint: test AP50 by class',categories:Object.keys(test.per_class_ap_metrics),
  series:[{name:'Test AP50',values:Object.values(test.per_class_ap_metrics).map(r=>chartValue(r.ap50)),fill:C.teal}],
  hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:85},
  xAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.00'},yAxis:{title:'Defect category'},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:12,typeface:FONT}}
 });
 text(s,`${selection.selected_candidate_group} • seed ${selection.selected_seed} • ${test.test_images_evaluated} test images • thresholds were calibrated on validation only`,{x:105,y:645,w:1080,h:24,size:13,color:C.muted});
 notes(s,`Source: ${`results/comprehensive/test_evaluations/${selection.selected_candidate_group}/seed_${selection.selected_seed}_test_metrics.json`}. Test AP follows the Ultralytics validator confidence sweep. Per-class operating-point thresholds were selected on validation at IoU 0.50 and then applied unchanged to test. The same split had a historical YOLOv8n Phase 6 evaluation.`);
}

// 10 — confusion matrix
{
 const s=base('Error analysis','The confusion matrix shows where predictions are exchanged or missed.','Counts are from the single final test run.');
 image(s,'results/comprehensive/figures/final_test_confusion_matrix.png',220,204,840,430,'Final test confusion matrix');
 text(s,'Read the diagonal as correct class detections; off-diagonal cells show confusions. Background rows/columns represent missed or extra detections.',{x:126,y:644,w:1020,h:30,size:15,color:C.muted});
 notes(s,'Source: the final test evaluation plot saved by Ultralytics and copied to results/comprehensive/figures/final_test_confusion_matrix.png.');
}

// 11 — qualitative validation only
{
 const s=base('Validation error review','Validation examples identify concrete missed and extra boxes.','Green = ground truth; dashed red = predictions after validation-calibrated confidence thresholds.');
 image(s,'results/comprehensive/figures/validation_error_examples.png',160,205,960,430,'Four validation images showing false positives and missed detections');
 text(s,'These are validation images. Test images are not used for qualitative error selection.',{x:160,y:638,w:960,h:30,size:15,color:C.muted});
 notes(s,'Source: results/comprehensive/figures/validation_error_examples.png and validation_error_cases.csv. Examples are chosen from validation predictions only and are ranked by false positives plus missed boxes.');
}

// 12 — training behavior
{
 const s=base('Training behavior','Selected checkpoint training curves','Interpret the trend together with repeated-seed variation and the fixed epoch budget.');
 image(s,'results/comprehensive/figures/selected_training_curves.png',110,206,1060,430,'Training and validation loss and metric curves for selected checkpoint');
 text(s,'The transfer runs requested 50 epochs. Intervention epoch count was adjusted to match planned train-image exposure.',{x:113,y:641,w:1050,h:28,size:15,color:C.muted});
 notes(s,'Source: selected checkpoint training results image copied from its Ultralytics run directory. Training settings and run signatures are stored alongside checkpoints under outputs/checkpoints/phase8/.');
}

// 13 — interpretation and limits
{
 const s=base('Findings and limits','The project now has controlled evidence; claims should match its scope.','Three-seed comparisons improve reliability but remain small-sample evidence.');
 bulletList(s,[
  `Validation leader: ${best?.candidate ?? '—'} at tail AP50 ${fmt(best?.minority_macro_AP50_mean)} ± ${fmt(best?.minority_macro_AP50_sd)}; change vs YOLOv8n ${bestDelta>=0?'+':''}${fmt(bestDelta)}.`,
  `Historic YOLOv8n test: mAP50 ${fmt(metric(baseTest,'metrics/mAP50(B)'))}; Phase 8 candidate test: mAP50 ${fmt(metric(test,'metrics/mAP50(B)'))}. This comparison is descriptive.`,
  'One dataset, one split, three seeds, and a small benchmark do not establish statistical significance or production readiness.',
  'The Phase 6 baseline already used the same test split; a new untouched holdout is needed for a confirmatory comparison.'
 ],{y:215,size:19,step:94});
 notes(s,'Historical baseline result: results/final_test_metrics.json. Phase 8 candidate result: final test metrics JSON. The test split had been viewed in an earlier phase; Phase 8 selection scripts only used validation, but the project-wide comparison is not fully blind.');
}

// 14 — reproducibility and references
{
 const s=base('Reproducibility','Project files and commands','Run from the repository root using the .venv kernel or .venv/bin/python.');
 text(s,'Experiment commands',{x:90,y:208,w:420,h:34,size:21,color:C.navy,bold:true});
 text(s,'.venv/bin/python -m src.atdl.run_comprehensive_study --stage architectures\n.venv/bin/python -m src.atdl.run_full_data_interventions\n.venv/bin/python -m src.atdl.finalize_evaluation --select-only\n.venv/bin/python -m src.atdl.finalize_evaluation\n.venv/bin/python -m src.atdl.build_final_report',{x:92,y:251,w:550,h:182,size:15,color:C.ink});
 text(s,'Key artifacts',{x:694,y:208,w:430,h:34,size:21,color:C.navy,bold:true});
 text(s,'Phase 8 VS Code notebook\nRun scripts and signatures\nBest checkpoints by seed\nCSV/JSON metrics and figures\nFinal report and checkpoint manifest',{x:696,y:251,w:440,h:190,size:17,color:C.ink});
 shape(s,{x:90,y:468,w:1100,h:1,fill:C.line});
 text(s,'References',{x:90,y:490,w:350,h:30,size:18,color:C.navy,bold:true});
 text(s,'Lv et al. (2020), NEU-DET benchmark, doi:10.3390/s20061562\nComparison of pretrained YOLO models on steel defects (2022), Sensors 22(24):9926\nUltralytics model/detection docs: docs.ultralytics.com/models and /tasks/detect\nYaman et al. (2023), Instance and Repeat Factor Sampling, arXiv:2305.08069',{x:90,y:530,w:1090,h:92,size:14,color:C.muted});
 notes(s,'Full bibliography and links are included in reports/FINAL_EXPERIMENT_REPORT.md. Method context: https://doi.org/10.3390/s20061562 ; https://www.mdpi.com/1424-8220/22/24/9926 ; https://docs.ultralytics.com/models/ ; https://docs.ultralytics.com/tasks/detect/ ; https://arxiv.org/abs/2305.08069');
}

await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.dirname(FINAL),{recursive:true});
await fs.mkdir(path.dirname(STAGED_FINAL),{recursive:true});
const previewDir=path.join(BUILD,'slides'); await fs.mkdir(previewDir,{recursive:true});
for(let i=0;i<pres.slides.items.length;i++){
  const blob=await pres.export({slide:pres.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(previewDir,`slide-${String(i+1).padStart(2,'0')}.png`),Buffer.from(await blob.arrayBuffer()));
}
const montage=await pres.export({format:'webp',montage:true,scale:0.45});
await fs.writeFile(path.join(BUILD,'montage.webp'),Buffer.from(await montage.arrayBuffer()));
const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(pres)).save(candidate);
const skillDir=process.env.ATDL_PRESENTATION_SKILL_DIR;
if (!skillDir) throw new Error('Set ATDL_PRESENTATION_SKILL_DIR to the presentation skill directory containing container_tools/artifact_tool_utils.mjs.');
const presentationPython=process.env.ATDL_PRESENTATION_PYTHON || 'python3';
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const finalizer=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:STAGED_FINAL,
  pythonExecutable:presentationPython,
  integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:pres.slides.items.length,
  requiredNativeChartOwnerSlides:[5,6,8,9,10],
  materializeLiteralChartWorkbooks:true,
  // Artifact Tool writes its native chart axis labels in Arial by default.
  fontPolicy:{basis:'design',families:[FONT,'Arial']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(BUILD,`ATDL_Final_Project_Results_${Date.now()}.validation.json`),
});
await fs.copyFile(STAGED_FINAL,FINAL);
await fs.unlink(STAGED_FINAL);
await fs.rmdir(path.dirname(STAGED_FINAL)).catch(()=>{});
console.log(JSON.stringify({final:FINAL,slides:pres.slides.items.length,finalizer},null,2));
