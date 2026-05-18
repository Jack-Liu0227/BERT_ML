import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(r'D:\XJTU\ImportantFile\auto-design-alloy\BERT_ML')
OOD_RESULTS = ROOT / 'output' / 'ood_results'
REPORTS_ROOT = ROOT / 'output' / 'ood_summary_reports'
BACKUP_ROOT = REPORTS_ROOT / '_backups'
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime('%Y%m%d_%H%M%S')

OOD_METHOD_ORDER = ['RandomCV','Extrapolation','LOCO','SparseXcluster','SparseXsingle','SparseYcluster','SparseYsingle']
TRAD_MODEL_MAP = {
    'catboost_results':'CatBoost',
    'lightgbm_results':'LightGBM',
    'mlp_results':'MLP',
    'sklearn_rf_results':'RF',
    'xgboost_results':'XGB',
}
TRAD_METHODS = {
    'RandomCV':('experiment1_all_ml_models_random_cv_baseline','random_cv_baseline'),
    'Extrapolation':('experiment1_all_ml_models_extrapolation','target_extrapolation'),
    'LOCO':('experiment1_all_ml_models_loco','loco_k5'),
    'SparseXcluster':('experiment1_all_ml_models_sparse_x_cluster','sparse_x_cluster_k5'),
    'SparseXsingle':('experiment1_all_ml_models_sparse_x_single','sparse_x_single_k5'),
    'SparseYcluster':('experiment1_all_ml_models_sparse_y_cluster','sparse_y_cluster_k5'),
    'SparseYsingle':('experiment1_all_ml_models_sparse_y_single','sparse_y_single_k5'),
}
BERT_ROOTS = {
    'SciBERT':'experiment2a_all_nn_scibert',
    'SteelBERT':'experiment2b_all_nn_steelbert',
    'MatSciBERT':'experiment2c_all_nn_matscibert',
}
BERT_METHOD_SUFFIX = {
    'RandomCV':'random_cv_baseline',
    'Extrapolation':'extrapolation',
    'LOCO':'loco',
    'SparseXcluster':'sparse_x_cluster',
    'SparseXsingle':'sparse_x_single',
    'SparseYcluster':'sparse_y_cluster',
    'SparseYsingle':'sparse_y_single',
}
BERT_RAW_DIR = {
    'RandomCV':'random_cv_baseline',
    'Extrapolation':'target_extrapolation',
    'LOCO':'loco_k5',
    'SparseXcluster':'sparse_x_cluster_k5',
    'SparseXsingle':'sparse_x_single_k5',
    'SparseYcluster':'sparse_y_cluster_k5',
    'SparseYsingle':'sparse_y_single_k5',
}
TARGET_COL = 'yield strength'
ALLOY = 'Matbench Steel'
DATASET = 'matbench_steels_ood'

CANONICAL_COLUMNS = None
combined_file = REPORTS_ROOT / 'Combined' / 'data' / 'all_model_families_ood_summary.csv'
if combined_file.exists():
    CANONICAL_COLUMNS = pd.read_csv(combined_file, nrows=0, encoding='utf-8-sig').columns.tolist()


def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ('utf-8-sig','utf-8','gb18030','gbk','latin1'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)


def resolve_cols(df: pd.DataFrame):
    dataset_col = 'Dataset' if 'Dataset' in df.columns else ('set' if 'set' in df.columns else None)
    pairs = [
        ('yield strength_Actual','yield strength_Predicted'),
        ('True_yield_strength','Pred_yield_strength'),
        ('Actual_yield_strength','Predicted_yield_strength'),
        ('True_yieldstrength','Pred_yieldstrength'),
    ]
    for a,b in pairs:
        if a in df.columns and b in df.columns:
            return dataset_col,a,b
    actual = [c for c in df.columns if c.endswith('_Actual') or c.startswith('True_') or c.startswith('Actual_')]
    for a in actual:
        if a.endswith('_Actual'):
            b=a[:-7]+'_Predicted'
        elif a.startswith('True_'):
            b='Pred_'+a[5:]
        else:
            b='Predicted_'+a[7:]
        if b in df.columns:
            return dataset_col,a,b
    raise ValueError(f'Cannot resolve prediction columns: {df.columns.tolist()}')


def metrics_from_prediction_file(path: Path) -> dict:
    df=read_csv_any(path)
    dataset_col,actual,pred=resolve_cols(df)
    if dataset_col:
        labels=df[dataset_col].astype(str).str.strip().str.lower().str.replace('_','',regex=False).str.replace(' ','',regex=False)
        mask=labels.isin({'test','testing','oodtest','ood','oodtesting','extrapolationtest'})
        sub=df.loc[mask].copy()
        if sub.empty:
            sub=df.copy()
    else:
        sub=df.copy()
    sub=sub[[actual,pred]].dropna()
    y=sub[actual].to_numpy(dtype=float); yp=sub[pred].to_numpy(dtype=float)
    return {
        'r2': float(r2_score(y,yp)),
        'mae': float(mean_absolute_error(y,yp)),
        'rmse': float(np.sqrt(mean_squared_error(y,yp))),
        'row_count': int(len(sub)),
        'predictions_file': str(path),
    }


def aggregate_prediction_files(files: list[Path]) -> dict | None:
    rec=[]
    for p in files:
        if p.exists():
            try:
                m=metrics_from_prediction_file(p)
                m['path']=str(p)
                rec.append(m)
            except Exception as exc:
                print('[WARN]', p, exc)
    if not rec:
        return None
    df=pd.DataFrame(rec)
    mae=float(df['mae'].mean()); r2=float(df['r2'].mean()); rmse=float(df['rmse'].mean())
    # representative closest to mean MAE, then best MAE
    rep=df.assign(_d=(df['mae']-mae).abs()).sort_values(['_d','mae','path']).iloc[0]
    return {
        'summary_test_r2': r2,
        'summary_test_r2_std': float(df['r2'].std()) if len(df)>1 else 0.0,
        'summary_test_mae': mae,
        'summary_test_mae_std': float(df['mae'].std()) if len(df)>1 else 0.0,
        'summary_test_rmse': rmse,
        'summary_test_rmse_std': float(df['rmse'].std()) if len(df)>1 else 0.0,
        'fold_count': int(len(df)),
        'representative_test_r2': float(rep['r2']),
        'representative_test_mae': float(rep['mae']),
        'representative_test_rmse': float(rep['rmse']),
        'representative_predictions_file': str(rep['predictions_file']),
        'artifact_test_r2': r2,
        'artifact_test_mae': mae,
        'artifact_test_rmse': rmse,
        'artifact_test_row_count': float(df['row_count'].sum()),
        'artifact_predictions_file': str(rep['predictions_file']),
        'plot_test_r2': r2,
        'plot_test_mae': mae,
        'plot_test_rmse': rmse,
    }


def base_row(model_family, model, display_label, ood_method, model_dir, source_dir, metrics, trial_count=1):
    row={
        'alloy_family':ALLOY,
        'dataset_name':DATASET,
        'property':TARGET_COL,
        'ood_method':ood_method,
        'model_family':model_family,
        'model':model,
        'display_label':display_label,
        'model_dir':str(model_dir),
        'source_dir':str(source_dir),
        'trial_count':trial_count,
        'fold_count':metrics.get('fold_count',1),
        'summary_test_r2':metrics['summary_test_r2'],
        'summary_test_r2_std':metrics.get('summary_test_r2_std',0.0),
        'summary_test_mae':metrics['summary_test_mae'],
        'summary_test_mae_std':metrics.get('summary_test_mae_std',0.0),
        'summary_test_rmse':metrics['summary_test_rmse'],
        'summary_test_rmse_std':metrics.get('summary_test_rmse_std',0.0),
        'representative_selection_mode':'matbench_direct_prediction_summary',
        'representative_trial_id':pd.NA,
        'representative_fold':pd.NA,
        'representative_test_r2':metrics['representative_test_r2'],
        'representative_test_mae':metrics['representative_test_mae'],
        'representative_test_rmse':metrics['representative_test_rmse'],
        'representative_predictions_file':metrics['representative_predictions_file'],
        'representative_plot_file':pd.NA,
        'loco_outer_fold_best_count':pd.NA,
        'loco_outer_fold_best_details_json':pd.NA,
        'tabpfn_loco_fold_count':pd.NA,
        'tabpfn_loco_fold_details_json':pd.NA,
        'artifact_selection_mode':'matbench_direct_prediction_summary',
        'artifact_predictions_file':metrics['artifact_predictions_file'],
        'artifact_expected_split_file':pd.NA,
        'artifact_test_r2':metrics['artifact_test_r2'],
        'artifact_test_mae':metrics['artifact_test_mae'],
        'artifact_test_rmse':metrics['artifact_test_rmse'],
        'artifact_test_row_count':metrics['artifact_test_row_count'],
        'plot_test_r2':metrics['plot_test_r2'],
        'plot_test_mae':metrics['plot_test_mae'],
        'plot_test_rmse':metrics['plot_test_rmse'],
        'family_best_metric':'summary_test_r2',
        'family_rank_score':metrics['summary_test_r2'],
        'rank_within_family':pd.NA,
        'is_family_best':False,
        'source_family_dir':pd.NA,
    }
    if CANONICAL_COLUMNS:
        for c in CANONICAL_COLUMNS:
            row.setdefault(c,pd.NA)
        return {c:row.get(c,pd.NA) for c in CANONICAL_COLUMNS}
    return row

trad_rows=[]
for method,(exp,raw) in TRAD_METHODS.items():
    base=OOD_RESULTS/exp/'MatbenchSteels'/DATASET/TARGET_COL/raw/'tradition'
    for model_dir_name, label in TRAD_MODEL_MAP.items():
        if method in {'RandomCV','LOCO'}:
            files=sorted((base/'folds').glob(f'fold_*/model_comparison/{model_dir_name}/predictions/test_predictions.csv'))
            model_dir=base/'folds'
        else:
            model_dir=base/'model_comparison'/model_dir_name
            files=[model_dir/'predictions'/'test_predictions.csv']
        metrics=aggregate_prediction_files(files)
        if metrics is None:
            print('[MISS TRAD]',method,label,base)
            continue
        trad_rows.append(base_row('Traditional',label,label,method,model_dir,OOD_RESULTS/exp,metrics,trial_count=1))

bert_rows=[]
for model_label, root_prefix in BERT_ROOTS.items():
    raw_model=model_label.lower()
    for method,suffix in BERT_METHOD_SUFFIX.items():
        exp=OOD_RESULTS/f'{root_prefix}_{suffix}'
        raw=BERT_RAW_DIR[method]
        model_root=exp/'MatbenchSteels'/DATASET/TARGET_COL/raw/raw_model
        if method in {'RandomCV','LOCO'}:
            files=sorted(model_root.glob('folds/fold_*/predictions/best_model_all_predictions.csv'))
        else:
            files=[model_root/'predictions'/'best_model_all_predictions.csv']
        metrics=aggregate_prediction_files(files)
        if metrics is None:
            print('[MISS BERT]',method,model_label,model_root)
            continue
        bert_rows.append(base_row('BERT',model_label,model_label,method,model_root,exp,metrics,trial_count=1))

print('trad_rows',len(trad_rows),'bert_rows',len(bert_rows))

# Upsert into family summary files preserving old rows.
def upsert_summary(path: Path, rows: list[dict], family_name: str):
    if path.exists():
        backup=BACKUP_ROOT/f'{path.stem}.bak_before_matbench_{TS}.csv'
        shutil.copy2(path, backup)
        df=pd.read_csv(path, encoding='utf-8-sig')
    else:
        df=pd.DataFrame(columns=CANONICAL_COLUMNS or [])
    add=pd.DataFrame(rows)
    if df.empty:
        out=add
    else:
        mask=(df.get('alloy_family',pd.Series(dtype=str)).astype(str).eq(ALLOY) & df.get('dataset_name',pd.Series(dtype=str)).astype(str).eq(DATASET) & df.get('property',pd.Series(dtype=str)).astype(str).eq(TARGET_COL))
        out=pd.concat([df.loc[~mask].copy(), add], ignore_index=True)
    # rank within family per case/method/family by R2 desc
    if 'rank_within_family' in out.columns:
        out['family_rank_score']=pd.to_numeric(out.get('summary_test_r2'), errors='coerce')
        keys=['alloy_family','dataset_name','property','ood_method','model_family']
        out['rank_within_family']=out.groupby(keys, dropna=False)['family_rank_score'].rank(ascending=False, method='min')
        out['is_family_best']=out['rank_within_family'].eq(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path,index=False,encoding='utf-8-sig')
    print('wrote', path, 'rows', len(out), 'matbench', len(add))

upsert_summary(REPORTS_ROOT/'Traditional'/'00_summary_tables'/'all_traditional_ood_model_summary.csv', trad_rows, 'Traditional')
upsert_summary(REPORTS_ROOT/'BERT'/'00_summary_tables'/'all_bert_ood_model_summary.csv', bert_rows, 'BERT')

# Minimal case dirs for traceability.
for family_name, rows in [('Traditional',trad_rows),('BERT',bert_rows)]:
    case_dir=REPORTS_ROOT/family_name/'01_alloy_cases'/ALLOY/DATASET/TARGET_COL.replace(' ','_')
    case_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(case_dir/'case_model_summary.csv', index=False, encoding='utf-8-sig')
