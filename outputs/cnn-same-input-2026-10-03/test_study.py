"""Technical tests before the protocol is frozen; no outer-test model scoring."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import study


class IntegrityTests(unittest.TestCase):
    def test_entire_author_dataset_and_labels(self):
        ids,x,flat,y,strata=study.inputs()
        self.assertEqual(len(ids),173)
        self.assertIn('sub-M2019',ids)
        self.assertEqual(y[np.flatnonzero(ids=='sub-M2087')[0]],0)
        self.assertEqual(np.bincount(y).tolist(),[119,54])
        np.testing.assert_array_equal(x,flat.reshape(x.shape))

    def test_normalization_matches_author_column_scaling(self):
        ids,x,flat,y,strata=study.inputs()
        m=loadmat(study.DATA,squeeze_me=True)
        names=np.array(['sub-'+str(v) if not str(v).startswith('sub-') else str(v) for v in m['subsOnly']])
        raw=m['data'][...,np.argsort(names)]
        author=MinMaxScaler(feature_range=(-1,1)).fit_transform(raw.reshape(-1,173)).T
        np.testing.assert_allclose(flat,author,atol=1e-7)

    def test_network_snapshot_matches_audited_author_source(self):
        self.assertEqual(study.sha(study.NETWORK),'6088e711ae2486f327d5fcbe7d6e703c5dd46792c0932847eefada29d4a59623')
        self.assertEqual(study.sha(study.PILOT),'8d0b4a03ae7c4e8dcd1d7e39a1f8c60608e4627d4962e7ade2bfebe00946b0a7')

    def test_linear_and_rbf_precomputed_predictions_equal_direct(self):
        rng=np.random.default_rng(409)
        a=rng.normal(size=(40,7));b=rng.normal(size=(9,7));y=np.tile([0,1],20)
        w=study.inverse_weights(y);cw={0:w[0],1:w[1]}
        for kind in ['linear','rbf']:
            for scale in [.5,3,100]:
                direct=SVC(C=1.7,kernel=kind,gamma=1,class_weight=cw,tol=1e-6).fit(a/scale,y)
                pre=SVC(C=1.7,kernel='precomputed',class_weight=cw,tol=1e-6).fit(study.kernel_matrix(a,a,kind,scale),y)
                np.testing.assert_allclose(pre.decision_function(study.kernel_matrix(b,a,kind,scale)),
                                           direct.decision_function(b/scale),atol=1e-5)

    def test_heldout_images_do_not_change_training_kernel(self):
        rng=np.random.default_rng(808)
        x=rng.normal(size=(40,8));tr=np.arange(30)
        changed=x.copy();changed[30:]+=10000
        for kind in ['linear','rbf']:
            np.testing.assert_array_equal(study.kernel_matrix(x[tr],x[tr],kind,2),
                                          study.kernel_matrix(changed[tr],changed[tr],kind,2))

    def test_class_weights_derived_from_fit_members_only(self):
        y=np.array([0]*8+[1]*2)
        np.testing.assert_allclose(study.inverse_weights(y),[1.25,5])

    def test_search_uses_declared_mean_loss(self):
        rng=np.random.default_rng(81);x=rng.normal(size=(36,9));y=np.tile([0,1],18)
        inner=list(StratifiedKFold(3,shuffle=True,random_state=9).split(x,y))
        old=study.OUT
        with tempfile.TemporaryDirectory() as d:
            study.OUT=Path(d)
            try:
                chosen,trials=study.tune_svm(x,y,inner,[dict(C=.1,scale=1),dict(C=2,scale=2)],'rbf','synthetic')
                self.assertEqual(chosen['candidate'],int(np.argmin([r['mean_loss'] for r in trials])))
                for r in trials: self.assertAlmostEqual(r['mean_loss'],np.mean(r['inner_losses']))
            finally: study.OUT=old

    def test_numeric_metadata_serializes_without_nonstandard_json(self):
        json.dumps(study.json_safe(dict(missing=float('nan'),x=np.int64(2))),allow_nan=False)

    def test_baseline_implementations_fit_and_persist_on_synthetic_data(self):
        import joblib
        import warnings
        from threadpoolctl import threadpool_limits
        rng=np.random.default_rng(6001);x=rng.uniform(-1,1,(36,12));y=np.tile([0,0,1],12)
        with tempfile.TemporaryDirectory() as d, threadpool_limits(limits=2):
            for name in study.FIXED_MODELS:
                model,w=study.estimator(name,91,y)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if w is None: model.fit(x,y)
                    else: model.fit(x,y,sample_weight=w)
                    pred=model.predict_proba(x[:3])
                    path=Path(d)/f'{name}.joblib';joblib.dump(model,path)
                    np.testing.assert_allclose(joblib.load(path).predict_proba(x[:3]),pred)
                json.dumps(study.json_safe(model.get_params()),allow_nan=False)


if __name__=='__main__': unittest.main(verbosity=2)
