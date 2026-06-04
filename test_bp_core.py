import unittest
import numpy as np
from bp_core import (
    compute_sqi, _filter_signals, calculate_pat_both, CalibModel,
    SQIResult, PATResult, FS, DEFAULT_FOOT, DEFAULT_PEAK
)


class TestSignalQuality(unittest.TestCase):

    def test_sqi_valid_signals(self):
        t = np.arange(0, 5, 1/FS)
        ecg_raw = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
        ppg_raw = 500 + 50 * np.sin(2 * np.pi * 1.2 * t) + 10 * np.random.randn(len(t))
        
        result = compute_sqi(ecg_raw, ppg_raw)
        print(f"SQI valid signals: ok={result.ok}, reason={result.reason}, n_peaks={result.n_peaks}")
        self.assertTrue(result.ok, f"SQI should be valid. Got: {result.reason}")
        self.assertGreater(result.n_peaks, 2, f"n_peaks={result.n_peaks} should be > 2")

    def test_sqi_low_ecg_amplitude(self):
        ecg_raw = np.random.randn(len(np.arange(0, 5, 1/FS))) * 0.001
        ppg_raw = np.ones(len(ecg_raw)) * 500
        
        result = compute_sqi(ecg_raw, ppg_raw)
        print(f"SQI low amplitude: ok={result.ok}, reason={result.reason}")
        self.assertFalse(result.ok, f"Should reject low amplitude. Got ok={result.ok}")
        self.assertIn("amplitude", result.reason.lower(), f"Expected 'amplitude' in reason: {result.reason}")

    def test_sqi_no_ppg_signal(self):
        t = np.arange(0, 5, 1/FS)
        ecg_raw = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        ppg_raw = np.ones(len(ecg_raw)) * 500
        
        result = compute_sqi(ecg_raw, ppg_raw)
        print(f"SQI no PPG: ok={result.ok}, reason={result.reason}")
        self.assertFalse(result.ok, f"Should reject flat PPG. Got ok={result.ok}")
        self.assertIn("PPG", result.reason, f"Expected 'PPG' in reason: {result.reason}")


class TestSignalFiltering(unittest.TestCase):

    def test_filter_signals_valid_length(self):
        ecg_raw = np.random.randn(2500)
        ppg_raw = np.random.randn(2500) * 100 + 500
        
        ecg_f, ppg_lp = _filter_signals(ecg_raw, ppg_raw)
        
        print(f"Filter valid: ecg_len={len(ecg_f)}, ppg_len={len(ppg_lp)}")
        self.assertEqual(len(ecg_f), len(ecg_raw), f"ECG length {len(ecg_f)} != {len(ecg_raw)}")
        self.assertEqual(len(ppg_lp), len(ppg_raw), f"PPG length {len(ppg_lp)} != {len(ppg_raw)}")
        self.assertIsInstance(ecg_f, np.ndarray, f"ECG type is {type(ecg_f)}, expected ndarray")
        self.assertIsInstance(ppg_lp, np.ndarray, f"PPG type is {type(ppg_lp)}, expected ndarray")

    def test_filter_signals_too_short(self):
        ecg_raw = np.random.randn(10)
        ppg_raw = np.random.randn(10)
        
        with self.assertRaises(ValueError) as context:
            _filter_signals(ecg_raw, ppg_raw)
        
        print(f"Filter too short raised: {str(context.exception)}")
        self.assertIn("too short", str(context.exception).lower(), f"Expected 'too short' in error: {context.exception}")

    def test_filter_removes_dc_component(self):
        t = np.arange(0, 5, 1/FS)
        ecg_raw = 5.0 + 0.5 * np.sin(2 * np.pi * 1.2 * t)
        ppg_raw = np.ones(len(t)) * 500
        
        ecg_f, _ = _filter_signals(ecg_raw, ppg_raw)
        
        dc_mean = np.abs(np.mean(ecg_f))
        print(f"Filter DC removal: mean={dc_mean:.6f}")
        self.assertLess(dc_mean, 0.5, f"DC mean {dc_mean} should be < 0.5")


class TestPATCalculation(unittest.TestCase):

    def test_pat_valid_signals(self):
        t = np.arange(0, 10, 1/FS)
        ecg_raw = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        ppg_raw = 500 + 50 * np.sin(2 * np.pi * 1.2 * (t - 0.2))
        
        result = calculate_pat_both(ecg_raw, ppg_raw)
        
        print(f"PAT valid: sqi_ok={result.sqi.ok}, foot={result.foot}, peak={result.peak}")
        self.assertTrue(result.sqi.ok, f"SQI failed: {result.sqi.reason}")
        self.assertIsNotNone(result.foot, f"Foot PAT is None")
        self.assertIsNotNone(result.peak, f"Peak PAT is None")
        self.assertGreater(result.peak, 40, f"Peak PAT {result.peak} should be > 40")
        self.assertLess(result.peak, 300, f"Peak PAT {result.peak} should be < 300")

    def test_pat_insufficient_peaks(self):
        ecg_raw = np.random.randn(500) * 0.01
        ppg_raw = np.ones(500) * 500 + np.random.randn(500) * 5
        
        result = calculate_pat_both(ecg_raw, ppg_raw)
        
        print(f"PAT insufficient peaks: sqi_ok={result.sqi.ok}, reason={result.sqi.reason}")
        self.assertFalse(result.sqi.ok, f"SQI should fail with low peaks. Got ok={result.sqi.ok}")


class TestCalibModel(unittest.TestCase):

    def setUp(self):
        self.model = CalibModel()

    def test_predict_default_foot(self):
        pat = 150
        sbp, dbp = self.model.predict(pat, method="foot")
        
        print(f"Predict foot: pat={pat}, sbp={sbp}, dbp={dbp}, using_defaults={self.model.foot_using_defaults}")
        self.assertIsInstance(sbp, float, f"SBP type {type(sbp)} != float")
        self.assertIsInstance(dbp, float, f"DBP type {type(dbp)} != float")
        self.assertGreater(sbp, 0, f"SBP {sbp} should be > 0")
        self.assertGreater(dbp, 0, f"DBP {dbp} should be > 0")
        self.assertAlmostEqual(sbp, 140, delta=1, msg=f"SBP {sbp} should be ~140")

    def test_predict_default_peak(self):
        pat = 200
        sbp, dbp = self.model.predict(pat, method="peak")
        
        print(f"Predict peak: pat={pat}, sbp={sbp}, dbp={dbp}")
        self.assertIsInstance(sbp, float, f"SBP type {type(sbp)} != float")
        self.assertIsInstance(dbp, float, f"DBP type {type(dbp)} != float")

    def test_add_calibration(self):
        result = self.model.add_calibration(
            pat_foot=150.0, pat_peak=180.0, sbp=120, dbp=75, label="morning"
        )
        
        print(f"Add calib: records={len(self.model.calibrations)}, fit_foot_updated={result['foot']['updated']}")
        self.assertEqual(len(self.model.calibrations), 1, f"Should have 1 record, got {len(self.model.calibrations)}")
        self.assertEqual(self.model.calibrations[0]["sbp"], 120, f"SBP {self.model.calibrations[0]['sbp']} != 120")
        self.assertEqual(self.model.calibrations[0]["label"], "morning", f"Label mismatch")
        self.assertFalse(result["foot"]["updated"], f"Fit should not update with 1 record")

    def test_model_fitting_sufficient_data(self):
        pats = [140, 170, 200]
        sbps = [110, 130, 150]
        dbps = [70, 85, 95]
        
        for pat, sbp, dbp in zip(pats, sbps, dbps):
            self.model.add_calibration(pat, pat, sbp, dbp)
        
        print(f"Model fit: records={len(self.model.calibrations)}, foot_using_defaults={self.model.foot_using_defaults}")
        self.assertFalse(self.model.foot_using_defaults, f"Should use custom foot model after fit")

    def test_remove_calibration(self):
        self.model.add_calibration(150, 180, 120, 75)
        self.assertEqual(len(self.model.calibrations), 1, f"Initial count should be 1")
        
        removed = self.model.remove_calibration(0)
        
        print(f"Remove calib: removed={removed}, remaining={len(self.model.calibrations)}")
        self.assertTrue(removed, f"Remove should return True")
        self.assertEqual(len(self.model.calibrations), 0, f"Should have 0 records after remove")

    def test_update_calibration(self):
        self.model.add_calibration(150, 180, 120, 75, label="test")
        self.assertEqual(self.model.calibrations[0]["sbp"], 120, f"Initial SBP should be 120")
        
        updated = self.model.update_calibration(0, sbp=125, dbp=80)
        
        print(f"Update calib: updated={updated}, sbp={self.model.calibrations[0]['sbp']}, dbp={self.model.calibrations[0]['dbp']}")
        self.assertTrue(updated, f"Update should return True")
        self.assertEqual(self.model.calibrations[0]["sbp"], 125, f"SBP should be 125, got {self.model.calibrations[0]['sbp']}")
        self.assertEqual(self.model.calibrations[0]["dbp"], 80, f"DBP should be 80, got {self.model.calibrations[0]['dbp']}")

    def test_insufficient_sbp_diversity(self):
        for i in range(3):
            self.model.add_calibration(140 + i*5, 180 + i*5, 120, 75)
        
        print(f"Insufficient diversity: foot_using_defaults={self.model.foot_using_defaults}")
        self.assertTrue(self.model.foot_using_defaults, f"Should keep defaults due to low SBP diversity")

    def test_model_serialization(self):
        self.model.add_calibration(150, 180, 120, 75, label="test")
        
        d = self.model.to_dict()
        new_model = CalibModel.from_dict(d)
        
        print(f"Serialize: records={len(new_model.calibrations)}, label={new_model.calibrations[0]['label']}, foot_s_a={new_model.foot_s_a}")
        self.assertEqual(len(new_model.calibrations), 1, f"Should have 1 record after deserialize")
        self.assertEqual(new_model.calibrations[0]["label"], "test", f"Label mismatch after deserialize")
        self.assertEqual(new_model.foot_s_a, self.model.foot_s_a, f"foot_s_a mismatch")


class TestIntegration(unittest.TestCase):

    def test_end_to_end_signal_processing(self):
        t = np.arange(0, 10, 1/FS)
        heart_rate = 1.25
        pat_ms = 180
        
        ecg_raw = 0.5 * np.sin(2 * np.pi * heart_rate * t) + 0.1 * np.random.randn(len(t))
        ppg_raw = (500 + 50 * np.sin(2 * np.pi * heart_rate * (t - pat_ms/1000))
                   + 10 * np.random.randn(len(t)))
        
        ecg_f, ppg_lp = _filter_signals(ecg_raw, ppg_raw)
        print(f"Step 1 Filter: ecg_mean={np.mean(ecg_f):.4f}, ppg_mean={np.mean(ppg_lp):.2f}")
        
        sqi = compute_sqi(ecg_f, ppg_lp)
        print(f"Step 2 SQI: ok={sqi.ok}, n_peaks={sqi.n_peaks}")
        self.assertTrue(sqi.ok, f"SQI failed: {sqi.reason}")
        
        result = calculate_pat_both(ecg_raw, ppg_raw)
        print(f"Step 3 PAT: foot={result.foot}, peak={result.peak}")
        self.assertTrue(result.sqi.ok, f"PAT SQI failed: {result.sqi.reason}")
        self.assertIsNotNone(result.peak, f"Peak PAT is None")
        
        model = CalibModel()
        sbp, dbp = model.predict(result.peak, method="peak")
        print(f"Step 4 Predict: sbp={sbp}, dbp={dbp}")
        self.assertGreater(sbp, 0, f"SBP {sbp} should be > 0")
        self.assertGreater(dbp, 0, f"DBP {dbp} should be > 0")


if __name__ == '__main__':
    unittest.main(verbosity=2)
