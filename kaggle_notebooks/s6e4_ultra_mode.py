import os
import gc
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import lightgbm as lgb
except Exception:
    raise ImportError('lightgbm is required')

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
except Exception:
    raise ImportError('catboost is required')
from sklearn.ensemble import HistGradientBoostingClassifier


TARGET = 'Irrigation_Need'
ID_COL = 'id'
LABELS = ['Low', 'Medium', 'High']
LABEL_TO_INT = {k: i for i, k in enumerate(LABELS)}
INT_TO_LABEL = {i: k for k, i in LABEL_TO_INT.items()}
NUM_COLS = ['Soil_Moisture', 'Rainfall_mm', 'Temperature_C', 'Wind_Speed_kmh', 'Humidity']
CAT_COLS = [
    'Crop_Type', 'Soil_Type', 'Region', 'Weather_Condition',
    'Crop_Growth_Stage', 'Mulching_Used', 'Irrigation_System_Type',
    'Season', 'Irrigation_Type', 'Water_Source'
]


USE_ADV_WEIGHTS = True
ADV_WEIGHT_ALPHA = 0.35
USE_PSEUDO = False
PSEUDO_CONF = 0.995
PSEUDO_MAX_FRACTION = 0.08


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_file(filename: str) -> str:
    candidates = [
        '/kaggle/input/playground-series-s6e4',
        '/kaggle/input',
        '/content',
        '.'
    ]
    for d in candidates:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{filename} not found')


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(find_file('train.csv'))
    test = pd.read_csv(find_file('test.csv'))
    sample = pd.read_csv(find_file('sample_submission.csv'))
    return train, test, sample


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['soil_lt_25'] = (df['Soil_Moisture'] < 25).astype(int)
    df['rain_lt_300'] = (df['Rainfall_mm'] < 300).astype(int)
    df['temp_gt_30'] = (df['Temperature_C'] > 30).astype(int)
    df['wind_gt_10'] = (df['Wind_Speed_kmh'] > 10).astype(int)
    df['high_score'] = 2 * df['soil_lt_25'] + 2 * df['rain_lt_300'] + df['temp_gt_30'] + df['wind_gt_10']
    df['is_harvest'] = (df['Crop_Growth_Stage'] == 'Harvest').astype(int)
    df['is_sowing'] = (df['Crop_Growth_Stage'] == 'Sowing').astype(int)
    df['mulch_yes'] = (df['Mulching_Used'] == 'Yes').astype(int)
    df['low_score'] = 2 * df['is_harvest'] + 2 * df['is_sowing'] + df['mulch_yes']
    df['magic_score'] = df['high_score'] - df['low_score']
    df['formula_pred_int'] = np.where(df['magic_score'] <= 0, 0, np.where(df['magic_score'] <= 3, 1, 2))
    df['heat_index'] = df['Temperature_C'] * (df['Humidity'] / 100.0)
    df['moisture_stress'] = df['Soil_Moisture'] / (df['Temperature_C'] + 1e-6)
    df['rain_per_temp'] = df['Rainfall_mm'] / (df['Temperature_C'] + 1e-6)
    df['dry_and_hot'] = ((df['Soil_Moisture'] < 20) & (df['Temperature_C'] > 35)).astype(int)
    df['wet_and_cool'] = ((df['Soil_Moisture'] > 60) & (df['Temperature_C'] < 20)).astype(int)
    for col in ['Soil_Type', 'Region']:
        df[f'{col}_mean_moisture'] = df.groupby(col)['Soil_Moisture'].transform('mean')
        df[f'{col}_moisture_diff'] = df['Soil_Moisture'] - df[f'{col}_mean_moisture']
    return df


def encode_categories(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cat_cols:
        all_vals = pd.Index(pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0).unique())
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=all_vals).codes.astype(np.int32)
        test_df[col] = pd.Categorical(test_df[col].astype(str), categories=all_vals).codes.astype(np.int32)
    return train_df, test_df


def adversarial_weights(train_x: pd.DataFrame, test_x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    X_adv = pd.concat([train_x, test_x], axis=0).reset_index(drop=True)
    y_adv = np.r_[np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)]
    adv_idx = np.arange(len(X_adv))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_adv), dtype=float)
    for tr, va in skf.split(adv_idx, y_adv):
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_adv.iloc[tr], y_adv[tr])
        oof[va] = model.predict_proba(X_adv.iloc[va])[:, 1]
    train_adv = oof[:len(train_x)]
    w = 1.0 + ADV_WEIGHT_ALPHA * (0.5 - train_adv)
    return np.clip(w, 0.7, 1.3) * compute_sample_weight(class_weight='balanced', y=y)


def tune_thresholds(y_true: np.ndarray, y_probs: np.ndarray) -> np.ndarray:
    def objective(thresholds):
        preds = np.argmax(y_probs * thresholds, axis=1)
        return -balanced_accuracy_score(y_true, preds)
    res = minimize(objective, x0=np.ones(y_probs.shape[1]), method='Nelder-Mead', tol=1e-6)
    return res.x


def clean_predict_proba(model, X_part):
    if hasattr(model, 'best_iteration_') and getattr(model, 'best_iteration_', None) is not None:
        try:
            return model.predict_proba(X_part, num_iteration=model.best_iteration_)
        except Exception:
            pass
    return model.predict_proba(X_part)


def cv_lgbm(X, y, X_test, sample_weight=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3), dtype=np.float32)
    tst = np.zeros((len(X_test), 3), dtype=np.float32)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=16000,
            learning_rate=0.02,
            num_leaves=128,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        sw = sample_weight[tr] if sample_weight is not None else compute_sample_weight(class_weight='balanced', y=y[tr])
        model.fit(
            X.iloc[tr], y[tr],
            sample_weight=sw,
            eval_set=[(X.iloc[va], y[va])],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        pv = clean_predict_proba(model, X.iloc[va])
        pt = clean_predict_proba(model, X_test)
        oof[va] = pv
        tst += pt / n_splits
        sc = balanced_accuracy_score(y[va], pv.argmax(axis=1))
        scores.append(sc)
        print(f'LGBM fold {fold}: {sc:.6f}')
        del model, sw, pv, pt
        gc.collect()
    return oof, tst, scores


def cv_xgb(X, y, X_test, sample_weight=None, n_splits=5):
    if not HAS_XGB:
        print('xgboost is unavailable; using HistGradientBoosting fallback')
        return cv_hgb(X, y, X_test, sample_weight=sample_weight, n_splits=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3), dtype=np.float32)
    tst = np.zeros((len(X_test), 3), dtype=np.float32)
    scores = []
    base_params = dict(
        objective='multi:softprob',
        num_class=3,
        n_estimators=18000,
        learning_rate=0.015,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=SEED,
        n_jobs=-1,
    )
    try:
        import torch
        if torch.cuda.is_available():
            base_params['device'] = 'cuda'
    except Exception:
        pass
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        params = dict(base_params)
        params['random_state'] = SEED + fold
        model = xgb.XGBClassifier(**params)
        sw = sample_weight[tr] if sample_weight is not None else compute_sample_weight(class_weight='balanced', y=y[tr])
        try:
            model.fit(X.iloc[tr], y[tr], sample_weight=sw, eval_set=[(X.iloc[va], y[va])], verbose=False)
        except Exception:
            params.pop('device', None)
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[tr], y[tr], sample_weight=sw, eval_set=[(X.iloc[va], y[va])], verbose=False)
        pv = model.predict_proba(X.iloc[va])
        pt = model.predict_proba(X_test)
        oof[va] = pv
        tst += pt / n_splits
        sc = balanced_accuracy_score(y[va], pv.argmax(axis=1))
        scores.append(sc)
        print(f'XGB  fold {fold}: {sc:.6f}')
        del model, sw, pv, pt
        gc.collect()
    return oof, tst, scores


def cv_hgb(X, y, X_test, sample_weight=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3), dtype=np.float32)
    tst = np.zeros((len(X_test), 3), dtype=np.float32)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            random_state=SEED + fold
        )
        sw = sample_weight[tr] if sample_weight is not None else compute_sample_weight(class_weight='balanced', y=y[tr])
        model.fit(X.iloc[tr], y[tr], sample_weight=sw)
        pv = model.predict_proba(X.iloc[va])
        pt = model.predict_proba(X_test)
        oof[va] = pv
        tst += pt / n_splits
        sc = balanced_accuracy_score(y[va], pv.argmax(axis=1))
        scores.append(sc)
        print(f'HGB  fold {fold}: {sc:.6f}')
        del model, sw, pv, pt
        gc.collect()
    return oof, tst, scores


def cv_cat(X, y, X_test, cat_idx, sample_weight=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 3), dtype=np.float32)
    tst = np.zeros((len(X_test), 3), dtype=np.float32)
    scores = []
    task_type = 'GPU'
    try:
        import torch
        if not torch.cuda.is_available():
            task_type = 'CPU'
    except Exception:
        task_type = 'CPU'
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='MultiClass',
            iterations=18000,
            learning_rate=0.02,
            depth=8,
            random_seed=SEED + fold,
            task_type=task_type,
            auto_class_weights='Balanced',
            od_type='Iter',
            od_wait=300,
            verbose=0,
        )
        fit_kwargs = dict(
            cat_features=cat_idx if len(cat_idx) else None,
            eval_set=(X.iloc[va], y[va]),
            use_best_model=True,
            verbose=0,
        )
        try:
            model.fit(X.iloc[tr], y[tr], **fit_kwargs)
        except Exception:
            if task_type == 'GPU':
                model = CatBoostClassifier(
                    loss_function='MultiClass',
                    eval_metric='MultiClass',
                    iterations=18000,
                    learning_rate=0.02,
                    depth=8,
                    random_seed=SEED + fold,
                    task_type='CPU',
                    auto_class_weights='Balanced',
                    od_type='Iter',
                    od_wait=300,
                    verbose=0,
                )
                model.fit(X.iloc[tr], y[tr], **fit_kwargs)
            else:
                raise
        pv = model.predict_proba(X.iloc[va])
        pt = model.predict_proba(X_test)
        oof[va] = pv
        tst += pt / n_splits
        sc = balanced_accuracy_score(y[va], pv.argmax(axis=1))
        scores.append(sc)
        print(f'CAT  fold {fold}: {sc:.6f}')
        del model, pv, pt
        gc.collect()
    return oof, tst, scores


def maybe_pseudo_label(train_x, train_y, test_x, test_probs):
    conf = test_probs.max(axis=1)
    order = np.argsort(-conf)
    keep = order[conf[order] >= PSEUDO_CONF]
    max_keep = int(len(test_x) * PSEUDO_MAX_FRACTION)
    keep = keep[:max_keep]
    if len(keep) == 0:
        return train_x, train_y
    pseudo_x = test_x.iloc[keep].copy()
    pseudo_y = test_probs[keep].argmax(axis=1)
    out_x = pd.concat([train_x, pseudo_x], axis=0).reset_index(drop=True)
    out_y = np.r_[train_y, pseudo_y]
    print(f'Pseudo-labeling added {len(keep)} rows')
    return out_x, out_y


def main():
    set_seed(SEED)
    train, test, sample = load_data()
    train_fe = add_features(train)
    test_fe = add_features(test)

    train_fe, test_fe = encode_categories(train_fe, test_fe, CAT_COLS)
    feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
    X = train_fe[feature_cols].copy()
    X_test = test_fe[feature_cols].copy()
    y = train_fe[TARGET].map(LABEL_TO_INT).astype(int).values
    cat_idx = [X.columns.get_loc(c) for c in CAT_COLS if c in X.columns]

    adv_w = adversarial_weights(X, X_test, y) if USE_ADV_WEIGHTS else compute_sample_weight(class_weight='balanced', y=y)
    print('Running base models...')
    oof_lgb, tst_lgb, _ = cv_lgbm(X, y, X_test, sample_weight=adv_w)
    oof_xgb, tst_xgb, _ = cv_xgb(X, y, X_test, sample_weight=adv_w)
    oof_cat, tst_cat, _ = cv_cat(X, y, X_test, cat_idx=cat_idx, sample_weight=adv_w)

    base_scores = np.array([
        balanced_accuracy_score(y, oof_lgb.argmax(axis=1)),
        balanced_accuracy_score(y, oof_xgb.argmax(axis=1)),
        balanced_accuracy_score(y, oof_cat.argmax(axis=1))
    ])
    print('Base OOF scores:', base_scores)
    blend_w = 1.0 / np.clip(base_scores, 1e-6, None)
    blend_w = blend_w / blend_w.sum()
    print('Blend weights:', blend_w)

    oof_blend = blend_w[0] * oof_lgb + blend_w[1] * oof_xgb + blend_w[2] * oof_cat
    tst_blend = blend_w[0] * tst_lgb + blend_w[1] * tst_xgb + blend_w[2] * tst_cat
    print('Blend OOF before thresholds:', balanced_accuracy_score(y, oof_blend.argmax(axis=1)))

    thresholds = tune_thresholds(y, oof_blend)
    print('Optimized thresholds:', thresholds)
    oof_preds = (oof_blend * thresholds).argmax(axis=1)
    tst_preds = (tst_blend * thresholds).argmax(axis=1)
    print('Blend OOF after thresholds:', balanced_accuracy_score(y, oof_preds))

    if USE_PSEUDO:
        X_aug, y_aug = maybe_pseudo_label(X, y, X_test, tst_blend)
        if len(X_aug) > len(X):
            print('Retraining LightGBM on pseudo-labeled data...')
            pseudo_oof, pseudo_tst, _ = cv_lgbm(X_aug, y_aug, X_test)
            pseudo_thresholds = tune_thresholds(y, oof_lgb)
            tst_preds = (pseudo_tst * pseudo_thresholds).argmax(axis=1)

    extreme_mask = (test_fe['magic_score'].values <= -2) | (test_fe['magic_score'].values >= 5)
    tst_preds = np.where(extreme_mask, test_fe['formula_pred_int'].values, tst_preds)

    submission = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET: [INT_TO_LABEL[int(x)] for x in tst_preds]
    })
    out_path = 'submission_ultra_mode.csv'
    submission.to_csv(out_path, index=False)
    print(f'Saved {out_path}')
    print(submission[TARGET].value_counts())
    return submission


if __name__ == '__main__':
    main()
