"""Meaningful numerical and leakage checks before the full experiment."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import run


class StudyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=run.previous()
        cls.ids,_,cls.x,cls.y,cls.strata,cls.units=cls.old.load_frozen()

    def test_all_nested_partitions(self):
        for index,u in enumerate(self.units):
            dev,te=np.array(u['development']),np.array(u['test'])
            seen=np.zeros(len(dev),int)
            for tr,va in run.inner_partitions(dev,self.strata,run.SEED+index*101+1):
                self.assertFalse(set(dev[tr])&set(te))
                self.assertFalse(set(dev[va])&set(te))
                self.assertFalse(set(tr)&set(va))
                self.assertEqual(set(tr)|set(va),set(range(len(dev))))
                self.assertEqual(set(self.y[dev[tr]]),{0,1})
                seen[va]+=1
            np.testing.assert_array_equal(seen,np.ones(len(dev)))

    def test_original_fixed_model_reproduced(self):
        unit=self.units[0]
        tr,te=np.array(unit['development']),np.array(unit['test'])
        model,notes=run.fit(run.BASELINE,self.x[tr],self.y[tr],unit['fit_seed'])
        self.assertEqual(notes,[])
        p=model.predict_proba(self.x[te])[:,1]
        old=pd.read_csv(run.OLD/'classical/units'/f"{unit['name']}.csv")
        np.testing.assert_allclose(p,old.loc[old.method=='XGBoost','score'],rtol=1e-7,atol=1e-8)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.ubj'
            model.save_model(path)
            restored=XGBClassifier(n_jobs=1)
            restored.load_model(path)
            np.testing.assert_array_equal(p,restored.predict_proba(self.x[te])[:,1])

    def test_threshold_objective_and_ties(self):
        rng=np.random.default_rng(13)
        for _ in range(10):
            y=np.array([0]*30+[1]*15)
            s=rng.random(len(y))
            expected=min(run.THRESHOLDS,key=lambda t:(-f1_score(y,s>=t),round(abs(t-.5),8),t))
            chosen,values=run.select_threshold(y,s)
            self.assertEqual(chosen,expected)
            np.testing.assert_allclose(values,[f1_score(y,s>=t) for t in run.THRESHOLDS])
        chosen,_=run.select_threshold(np.array([0,1]),np.array([0.,1.]))
        self.assertEqual(chosen,.5)

    def test_study_resumes_sampler_deterministically(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=12,n_startup_trials=2))
        study.enqueue_trial(run.BASELINE)
        for i in range(3):
            t=study.ask();config=run.suggest(t)
            if i==0:self.assertEqual(config,run.BASELINE)
            study.tell(t,float(i))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.joblib'
            run.save_pickle(path,dict(study=study))
            restored=joblib.load(path)['study']
            self.assertEqual(run.suggest(study.ask()),run.suggest(restored.ask()))

    def test_inner_objective_on_synthetic_data(self):
        rng=np.random.default_rng(21)
        x=rng.normal(size=(30,8)).astype(np.float32)
        y=(x[:,0]>0).astype(int)
        from sklearn.model_selection import StratifiedKFold
        inner=list(StratifiedKFold(3,shuffle=True,random_state=1).split(x,y))
        cfg=dict(run.BASELINE,n_estimators=10)
        value,scores,notes=run.evaluate_config(cfg,x,y,inner,42)
        self.assertEqual(notes,[])
        self.assertEqual(value,float(f1_score(y,scores>=.5)))
        expected=np.full(len(y),np.nan)
        for k,(tr,va) in enumerate(inner):
            m,_=run.fit(cfg,x[tr],y[tr],42+k)
            expected[va]=m.predict_proba(x[va])[:,1]
        np.testing.assert_array_equal(scores,expected)


if __name__=='__main__':
    unittest.main(verbosity=2)
