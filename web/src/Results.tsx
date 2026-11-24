import { useState } from 'react';
import { ArrowDown, ArrowUpRight } from 'lucide-react';
import { fmt, label } from './metrics';
import type { Project } from './types';

const figures = {
  'selected_validation_pr_curve.png': 'Validation precision–recall curve',
  'final_test_confusion_matrix.png': 'Final test confusion matrix',
  'per_class_ap50_baseline_vs_selected.png': 'Per-class validation AP50',
  'validation_error_examples.png': 'Saved validation error montage',
};

export function Results({ project, onOpenCase }: { project: Project; onOpenCase: (name: string) => void }) {
  const [group, setGroup] = useState('full_combined');
  const [className, setClassName] = useState('crazing');
  const [figure, setFigure] = useState<keyof typeof figures>('selected_validation_pr_curve.png');
  const [caseName, setCaseName] = useState(project.validation_cases[0]?.name ?? '');
  const viewCase = project.validation_cases.find(item => item.name === caseName);
  const { test } = project;
  const operating = test.per_class_operating[className];
  const run = project.candidate_runs[group];
  const testMetrics = [
    ['mAP50', test.mAP50], ['mAP50–95', test.mAP50_95], ['Precision', test.precision],
    ['Recall', test.recall], ['F1', test.F1],
  ] as const;
  return <main className="workspace results-workspace">
    <div className="page-heading"><div><h1>Results</h1><p>Validation comparisons and the recorded test evaluation.</p></div>
      <button className="button" onClick={() => document.getElementById("validation-ranking")?.scrollIntoView()}>Compare runs <ArrowDown size={15} /></button></div>

    <div className="results-lead">
      <section className="panel results-summary">
        <div className="research-header"><h2>Final test</h2><p>Combined YOLO26n · seed 42 · {test.images} images</p></div>
        <div className="summary-metrics">{testMetrics.map(([title, value]) =>
          <div key={title}><span>{title}</span><strong>{fmt(value, 4)}</strong></div>)}</div>
        <p className="result-note">The test split had already been used for the historical baseline. Test differences are descriptive.</p>
      </section>
      <section className="panel case-review">
        <div className="research-header"><h2>Review a validation frame</h2>
          <p>Compare live predictions with the recorded annotations.</p></div>
        <div className="case-layout">
          {viewCase?.available && <img src={`/api/cases/${encodeURIComponent(viewCase.name)}/image`}
            alt="Steel surface from the selected validation case" />}
          <div className="case-controls">
            <label className="select-field">Case<select aria-label="Validation case" value={caseName} onChange={e => setCaseName(e.target.value)}>
              {project.validation_cases.map(item => <option key={item.name}>{item.name}</option>)}
            </select></label>
            {viewCase && <p>{viewCase.false_positives} false positives · {viewCase.missed_boxes} missed boxes</p>}
            {viewCase?.available
              ? <button className="button primary" onClick={() => onOpenCase(viewCase.name)}>Inspect case <ArrowUpRight size={15} /></button>
              : <><button className="button" onClick={() => {
                  setFigure('validation_error_examples.png');
                  document.getElementById('saved-figures')?.scrollIntoView();
                }}>Show validation montage</button>
                <p>Restore local validation images to inspect individual cases.</p></>}
          </div>
        </div>
      </section>
    </div>

    <div className="research-grid result-details" id="validation-ranking">
      <section className="panel research-major">
        <div className="research-header"><h2>Validation ranking</h2><p>Mean across three seeds. Tail AP50 was the selection objective.</p></div>
        <div className="table-scroll"><table>
          <thead><tr><th>Candidate</th><th>Tail AP50</th><th>mAP50</th><th>mAP50–95</th><th>GPU ms</th></tr></thead>
          <tbody>{project.candidates.map(candidate => <tr key={candidate.candidate} className={candidate.candidate === group ? 'current' : ''}>
            <th scope="row"><button className="table-link" aria-pressed={group === candidate.candidate} onClick={() => setGroup(candidate.candidate)}>
              {label(candidate.candidate)}</button></th>
            <td>{fmt(candidate.minority_macro_AP50_mean)} <span>± {fmt(candidate.minority_macro_AP50_sd)}</span></td>
            <td>{fmt(candidate.mAP50_mean)}</td><td>{fmt(candidate.mAP50_95_mean)}</td>
            <td>{fmt(candidate.validation_inference_ms_mean, 2)}</td>
          </tr>)}</tbody>
        </table></div>
      </section>
      <section className="panel research-minor">
        <div className="research-header"><h2>Variation by seed</h2><p>{label(group)} · validation tail AP50</p></div>
        <div className="data-list">{run.map(entry => <div key={entry.seed}><span>Seed {entry.seed}</span><strong>{fmt(entry.tail_ap50, 4)}</strong></div>)}</div>
      </section>

      <section className="panel research-major">
        <div className="research-header"><h2>Defect performance</h2>
          <p>Test AP50 and counts at the per-class validation-selected cutoffs.</p></div>
        <div className="table-scroll"><table>
          <thead><tr><th>Defect class</th><th>AP50</th><th>Cutoff</th><th>Matched</th><th>False +</th><th>Missed</th></tr></thead>
          <tbody>{project.classes.map(name => {
            const stats = test.per_class_operating[name];
            return <tr key={name} className={name === className ? 'current' : ''}>
              <th scope="row"><button className="table-link" aria-pressed={name === className} onClick={() => setClassName(name)}>
                {label(name)}</button></th>
              <td>{fmt(test.per_class_ap[name].ap50)}</td><td>{fmt(stats.confidence_threshold, 2)}</td>
              <td>{stats.tp}</td><td>{stats.fp}</td><td>{stats.fn}</td>
            </tr>;
          })}</tbody>
        </table></div>
      </section>
      <section className="panel research-minor">
        <div className="research-header"><h2>{label(className)}</h2><p>Test operating point · IoU 0.50</p></div>
        <div className="data-list">
          <div><span>Precision</span><strong>{fmt(operating.test_precision_at_iou50)}</strong></div>
          <div><span>Recall</span><strong>{fmt(operating.test_recall_at_iou50)}</strong></div>
          <div><span>F1</span><strong>{fmt(operating.test_f1_at_iou50)}</strong></div>
        </div>
        {className === 'crazing' && <div className="research-note">{operating.fn} of {operating.tp + operating.fn} annotated crazing boxes were missed at this cutoff.</div>}
      </section>

      <section className="panel research-wide" id="saved-figures">
        <div className="research-header row-header"><div><h2>Saved figures</h2><p>Generated by the original evaluation.</p></div>
          <label className="select-field">Figure <select aria-label="Saved figure" value={figure}
            onChange={e => setFigure(e.target.value as keyof typeof figures)}>
            {Object.entries(figures).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select></label></div>
        <div className="figure-content"><a href={`/api/figures/${figure}`} target="_blank" rel="noreferrer" aria-label="Open full-size figure">
          <img src={`/api/figures/${figure}`} alt={figures[figure]} /></a></div>
      </section>
    </div>
  </main>;
}
