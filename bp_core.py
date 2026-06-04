"""
bp_core.py — Blood Pressure Estimation Algorithm
=================================================
PAT Methods:
  foot : PPG local minimum  (pulse onset)
  peak : PPG local maximum  (systolic peak)

Data Sources:
  CSVStreamer  — Plays back a CSV file as if live (for testing)
  BLEStreamer  — Receives live data from ESP32 via Bluetooth
                 Format: "timestamp,ecg,ppg,leadOff"
"""

import json
import asyncio
import threading
import queue
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════
FS            = 200      # Sampling frequency (Hz)
WIN_SEC       = 10       # Analysis window (seconds)
STEP_SEC      = 5        # Sliding step (seconds)
MIN_CALIB     = 3        # Minimum calibrations for regression
MIN_SBP_RANGE = 10       # Minimum SBP diversity (mmHg)
MIN_R2        = 0.5      # Regression acceptance R² threshold

# BLE
BLE_SERVICE_UUID        = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
BLE_CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
BLE_DEVICE_NAME         = "330_Project_Wireless"

# Default coefficients — for PAT range 150-350ms
# (Used until calibration is performed, rough estimate)
DEFAULT_FOOT = {"s_a": -0.30, "s_b": 185.00, "d_a": -0.20, "d_b": 120.00}
DEFAULT_PEAK = {"s_a": -1.50, "s_b": 252.50, "d_a": -1.00, "d_b": 165.00}


# ════════════════════════════════════════════
# SIGNAL QUALITY INDEX (SQI)
# ════════════════════════════════════════════
@dataclass
class SQIResult:
    ok:       bool
    reason:   str   = ""
    snr_db:   float = 0.0
    n_peaks:  int   = 0


def compute_sqi(ecg_f: np.ndarray, ppg_lp: np.ndarray, fs: int = FS) -> SQIResult:
    """
    Three basic checks:
      A) Is ECG amplitude reasonable?
      B) Is there a PPG signal?
      C) Were at least 3 R-peaks detected?
    + SNR estimate (informational)
    """
    if np.ptp(ecg_f) < 0.01:
        return SQIResult(False, "ECG amplitude too low")
    if np.std(ppg_lp - np.mean(ppg_lp)) < 1e-6:
        return SQIResult(False, "No PPG signal")

    r_locs, _ = find_peaks(ecg_f,
                            distance=int(fs * 0.4),
                            height=np.mean(ecg_f) + 0.5 * np.std(ecg_f))
    if len(r_locs) < 3:
        return SQIResult(False, f"Insufficient R-peaks ({len(r_locs)})", n_peaks=len(r_locs))

    # SNR: zero out R-peak regions and estimate noise power
    residual = ecg_f.copy()
    hw = int(fs * 0.05)
    for loc in r_locs:
        residual[max(0, loc-hw):min(len(residual), loc+hw)] = 0
    snr = 10 * np.log10(np.var(ecg_f) / (np.var(residual) + 1e-12))
    return SQIResult(True, "OK", snr_db=round(snr, 1), n_peaks=len(r_locs))


# ════════════════════════════════════════════
# FILTERS
# ════════════════════════════════════════════
def _make_ecg_filter(fs=FS):
    nyq = 0.5 * fs
    return butter(2, [0.5/nyq, 35/nyq], btype='band')

def _make_ppg_filter(fs=FS):
    nyq = 0.5 * fs
    return butter(2, 8/nyq, btype='low')

_ECG_B, _ECG_A = _make_ecg_filter()
_PPG_B, _PPG_A = _make_ppg_filter()


def _filter_signals(ecg_raw: np.ndarray, ppg_raw: np.ndarray):
    """
    ECG: 0.5–35 Hz bandpass
    PPG: 8 Hz lowpass (NO inversion — onset/peak detection on original signal)
    """
    if len(ecg_raw) < 15:
        raise ValueError(f"Signal too short ({len(ecg_raw)} samples, min 15 required)")
    ecg_f  = filtfilt(_ECG_B, _ECG_A, ecg_raw.astype(float))
    ppg_lp = filtfilt(_PPG_B, _PPG_A, ppg_raw.astype(float))
    return ecg_f, ppg_lp


# ════════════════════════════════════════════
# PAT CALCULATION — BOTH METHODS SIMULTANEOUSLY
# ════════════════════════════════════════════
@dataclass
class PATResult:
    foot:      Optional[float]        # Onset (local minimum) PAT — ms
    peak:      Optional[float]        # Systolic peak (local maximum) PAT — ms
    sqi:       SQIResult
    foot_pats: list = field(default_factory=list)   # Per-beat foot PAT
    peak_pats: list = field(default_factory=list)   # Per-beat peak PAT


def calculate_pat_both(ecg_raw: np.ndarray, ppg_raw: np.ndarray,
                        fs: int = FS) -> PATResult:
    """
    Calculates both foot and peak PAT in a single pass.

    FOOT method (pulse onset):
      Local minimum of PPG lowpass signal = end of diastole, the "foot"
      of the pulse wave. First PPG trough 80–350 ms after R-peak.
      More precise but slightly more sensitive to noise.

    PEAK method (pulse peak):
      Local maximum of PPG lowpass signal = systolic peak.
      First PPG peak 150–450 ms after R-peak.
      More prominent and more robust against noise.
      (This is the method used in last year's project.)

    Both methods use:
      - IQR-based outlier removal
      - Median for stable PAT estimation
    """
    try:
        ecg_f, ppg_lp = _filter_signals(ecg_raw, ppg_raw)
    except ValueError as e:
        return PATResult(None, None, SQIResult(False, str(e)))

    sqi = compute_sqi(ecg_f, ppg_lp, fs)
    if not sqi.ok:
        return PATResult(None, None, sqi)

    # R-peak detection
    r_locs, _ = find_peaks(ecg_f,
                            distance=int(fs * 0.4),
                            height=np.mean(ecg_f) + 0.5 * np.std(ecg_f))

    ppg_prom = max(50, 0.05 * np.ptp(ppg_lp))

    # PPG local minima (foot)
    foot_locs, _ = find_peaks(-ppg_lp, distance=int(fs * 0.3), prominence=ppg_prom)
    # PPG local maxima (peak)
    peak_locs, _ = find_peaks( ppg_lp, distance=int(fs * 0.3), prominence=ppg_prom)

    foot_pats, peak_pats = [], []

    for rp in r_locs:
        # FOOT: R+80ms → R+350ms
        cands_f = foot_locs[(foot_locs >= rp + int(0.08*fs)) &
                             (foot_locs <  rp + int(0.35*fs))]
        if len(cands_f):
            pat = (cands_f[0] - rp) * (1000.0 / fs)
            if 80 < pat < 350:
                foot_pats.append(pat)

        # PEAK: R+40ms → R+300ms
        cands_p = peak_locs[(peak_locs >= rp + int(0.04*fs)) &
                             (peak_locs <  rp + int(0.30*fs))]
        if len(cands_p):
            pat = (cands_p[0] - rp) * (1000.0 / fs)
            if 40 < pat < 300:
                peak_pats.append(pat)

    def iqr_median(pats):
        if len(pats) < 3:
            return None
        a = np.array(pats)
        q25, q75 = np.percentile(a, [25, 75])
        iqr = q75 - q25
        clean = a[(a >= q25 - 1.5*iqr) & (a <= q75 + 1.5*iqr)]
        return float(np.median(clean)) if len(clean) >= 2 else None

    foot_med = iqr_median(foot_pats)
    peak_med = iqr_median(peak_pats)

    if foot_med is None and peak_med is None:
        sqi = SQIResult(False,
                        f"PAT could not be calculated (foot:{len(foot_pats)}, peak:{len(peak_pats)})",
                        n_peaks=sqi.n_peaks)

    return PATResult(foot_med, peak_med, sqi, foot_pats, peak_pats)


# ════════════════════════════════════════════
# CALIBRATION MODEL
# ════════════════════════════════════════════
@dataclass
class CalibModel:
    # Foot coefficients
    foot_s_a: float = DEFAULT_FOOT["s_a"]
    foot_s_b: float = DEFAULT_FOOT["s_b"]
    foot_d_a: float = DEFAULT_FOOT["d_a"]
    foot_d_b: float = DEFAULT_FOOT["d_b"]
    foot_using_defaults: bool = True

    # Peak coefficients
    peak_s_a: float = DEFAULT_PEAK["s_a"]
    peak_s_b: float = DEFAULT_PEAK["s_b"]
    peak_d_a: float = DEFAULT_PEAK["d_a"]
    peak_d_b: float = DEFAULT_PEAK["d_b"]
    peak_using_defaults: bool = True

    # Calibration records:
    # [{"pat_foot", "pat_peak", "sbp", "dbp", "label", "file"}, ...]
    calibrations: list = field(default_factory=list)

    def predict(self, pat: float, method: str = "foot") -> tuple:
        """method: 'foot' or 'peak'"""
        if method == "foot":
            return (round(self.foot_s_a * pat + self.foot_s_b, 1),
                    round(self.foot_d_a * pat + self.foot_d_b, 1))
        else:
            return (round(self.peak_s_a * pat + self.peak_s_b, 1),
                    round(self.peak_d_a * pat + self.peak_d_b, 1))

    def add_calibration(self, pat_foot: Optional[float], pat_peak: Optional[float],
                         sbp: float, dbp: float,
                         label: str = "", file: str = "") -> dict:
        self.calibrations.append({
            "pat_foot": pat_foot, "pat_peak": pat_peak,
            "sbp": sbp, "dbp": dbp, "label": label, "file": file
        })
        return {"foot": self._fit("foot"), "peak": self._fit("peak")}

    def remove_calibration(self, index: int) -> bool:
        if 0 <= index < len(self.calibrations):
            self.calibrations.pop(index)
            self._fit("foot"); self._fit("peak")
            return True
        return False

    def update_calibration(self, index: int, sbp: float, dbp: float) -> bool:
        if 0 <= index < len(self.calibrations):
            self.calibrations[index]["sbp"] = sbp
            self.calibrations[index]["dbp"] = dbp
            self._fit("foot"); self._fit("peak")
            return True
        return False

    def _fit(self, method: str) -> dict:
        """Regression for the specified method. Checks update criteria."""
        pat_key = f"pat_{method}"
        valid = [c for c in self.calibrations if c.get(pat_key) is not None]

        if len(valid) < MIN_CALIB:
            return {"updated": False,
                    "reason": f"Min {MIN_CALIB} records required ({len(valid)} valid)"}

        pats = np.array([c[pat_key] for c in valid])
        sbps = np.array([c["sbp"]   for c in valid])
        dbps = np.array([c["dbp"]   for c in valid])

        if np.ptp(sbps) < MIN_SBP_RANGE:
            return {"updated": False,
                    "reason": f"Insufficient SBP diversity ({np.ptp(sbps):.1f} < {MIN_SBP_RANGE} mmHg)"}

        s_c = np.polyfit(pats, sbps, 1)
        d_c = np.polyfit(pats, dbps, 1)

        # R² check
        ss_res = np.sum((sbps - np.polyval(s_c, pats))**2)
        ss_tot = np.sum((sbps - np.mean(sbps))**2) + 1e-12
        r2 = 1 - ss_res / ss_tot

        if r2 < MIN_R2:
            return {"updated": False,
                    "reason": f"R²={r2:.2f} < {MIN_R2} — more diverse calibration required",
                    "r2": round(r2, 3)}

        if method == "foot":
            self.foot_s_a, self.foot_s_b = s_c[0], s_c[1]
            self.foot_d_a, self.foot_d_b = d_c[0], d_c[1]
            self.foot_using_defaults = False
        else:
            self.peak_s_a, self.peak_s_b = s_c[0], s_c[1]
            self.peak_d_a, self.peak_d_b = d_c[0], d_c[1]
            self.peak_using_defaults = False

        return {"updated": True, "reason": "Model updated", "r2": round(r2, 3)}

    def to_dict(self) -> dict:
        return {
            "foot_s_a": self.foot_s_a, "foot_s_b": self.foot_s_b,
            "foot_d_a": self.foot_d_a, "foot_d_b": self.foot_d_b,
            "foot_using_defaults": self.foot_using_defaults,
            "peak_s_a": self.peak_s_a, "peak_s_b": self.peak_s_b,
            "peak_d_a": self.peak_d_a, "peak_d_b": self.peak_d_b,
            "peak_using_defaults": self.peak_using_defaults,
            "calibrations": self.calibrations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibModel":
        m = cls(
            foot_s_a=d.get("foot_s_a", DEFAULT_FOOT["s_a"]),
            foot_s_b=d.get("foot_s_b", DEFAULT_FOOT["s_b"]),
            foot_d_a=d.get("foot_d_a", DEFAULT_FOOT["d_a"]),
            foot_d_b=d.get("foot_d_b", DEFAULT_FOOT["d_b"]),
            foot_using_defaults=d.get("foot_using_defaults", True),
            peak_s_a=d.get("peak_s_a", DEFAULT_PEAK["s_a"]),
            peak_s_b=d.get("peak_s_b", DEFAULT_PEAK["s_b"]),
            peak_d_a=d.get("peak_d_a", DEFAULT_PEAK["d_a"]),
            peak_d_b=d.get("peak_d_b", DEFAULT_PEAK["d_b"]),
            peak_using_defaults=d.get("peak_using_defaults", True),
        )
        m.calibrations = d.get("calibrations", [])
        return m


# ════════════════════════════════════════════
# USER DATABASE
# ════════════════════════════════════════════
class UserDB:
    def __init__(self, path: str = "users_db.json"):
        self.path  = path
        self.users: dict = {}
        self.load()

    def load(self):
        try:
            with open(self.path) as f:
                raw = json.load(f)
            for name, data in raw.items():
                self.users[name] = {
                    "info":  data["info"],
                    "model": CalibModel.from_dict(data["model"])
                }
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            self.users = {}

    def save(self):
        raw = {n: {"info": d["info"], "model": d["model"].to_dict()}
               for n, d in self.users.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=4, ensure_ascii=False)

    def add_user(self, name: str, age: int, gender: str) -> bool:
        if name in self.users:
            return False
        self.users[name] = {"info": {"age": age, "gender": gender},
                             "model": CalibModel()}
        self.save(); return True

    def delete_user(self, name: str):
        self.users.pop(name, None); self.save()

    def get_model(self, name: str) -> Optional[CalibModel]:
        return self.users.get(name, {}).get("model")

    def list_users(self) -> list:
        return list(self.users.keys())


# ════════════════════════════════════════════
# CSV STREAMER (Test mode)
# ════════════════════════════════════════════
class CSVStreamer:
    """
    Delivers CSV window by window as if it were a live Bluetooth stream.
    In Bluetooth integration, this class is replaced by BLEStreamer.
    """
    def __init__(self, df: pd.DataFrame,
                 win_sec: int = WIN_SEC, step_sec: int = STEP_SEC, fs: int = FS):
        self.df   = df
        self.win  = win_sec  * fs
        self.step = step_sec * fs
        self.idx  = 0
        self.done = False

    def next_window(self) -> Optional[pd.DataFrame]:
        end = self.idx + self.win
        if end > len(self.df):
            self.done = True
            return None
        seg = self.df.iloc[self.idx:end].copy()
        self.idx += self.step
        return seg

    def reset(self):
        self.idx = 0; self.done = False


# ════════════════════════════════════════════
# BLE STREAMER (Live Bluetooth mode)
# ════════════════════════════════════════════
class BLEStreamer:
    """
    Receives data from ESP32 via BLE.
    Format: "timestamp,ecg,ppg,leadOff"  (UTF-8, comma-separated)

    Usage:
        streamer = BLEStreamer()
        streamer.start()           # connects and collects in background

        # Get a window every 10 seconds:
        seg = streamer.get_window()   # DataFrame or None
        if seg is not None:
            pat, sqi = calculate_pat_both(seg['ecg'], seg['ppg'])

        streamer.stop()
    """

    def __init__(self,
                 device_name: str   = BLE_DEVICE_NAME,
                 service_uuid: str  = BLE_SERVICE_UUID,
                 char_uuid: str     = BLE_CHARACTERISTIC_UUID,
                 win_sec: int       = WIN_SEC,
                 fs: int            = FS):
        self.device_name  = device_name
        self.service_uuid = service_uuid
        self.char_uuid    = char_uuid
        self.win_samples  = win_sec * fs
        self.fs           = fs

        self._buffer: list  = []       # [(timestamp_us, ecg, ppg), ...]
        self._lock          = threading.Lock()
        self._running       = False
        self._thread        = None
        self._status        = "idle"   # idle | connecting | connected | error
        self._status_msg    = ""
        self._lead_off      = False    # Is electrode connection lost?
        self._q             = queue.Queue()  # Status messages for GUI

    # ── Status queries ───────────────────────
    @property
    def status(self) -> str:
        return self._status

    @property
    def status_msg(self) -> str:
        return self._status_msg

    @property
    def lead_off(self) -> bool:
        return self._lead_off

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_status_message(self) -> Optional[str]:
        """Returns the next status message in the queue (for GUI)."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    # ── Start / Stop ─────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._status  = "idle"

    # ── Get window ───────────────────────────
    def get_window(self) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame when enough data has accumulated (win_samples),
        and discards old data (step = win/2, 50% overlap).
        """
        with self._lock:
            if len(self._buffer) < self.win_samples:
                return None
            chunk   = self._buffer[:self.win_samples]
            step    = self.win_samples // 2
            self._buffer = self._buffer[step:]

        ts, ecg, ppg = zip(*chunk)
        return pd.DataFrame({"timestamp_us": ts, "ecg": ecg, "ppg": ppg})

    def peek_buffer(self, max_samples: int = None) -> Optional[pd.DataFrame]:
        """Return a snapshot of current buffered BLE samples without consuming them."""
        with self._lock:
            if not self._buffer:
                return None
            data = list(self._buffer)
        if max_samples is not None and len(data) > max_samples:
            data = data[-max_samples:]
        ts, ecg, ppg = zip(*data)
        return pd.DataFrame({"timestamp_us": ts, "ecg": ecg, "ppg": ppg})

    # ── Background BLE loop ──────────────────
    def _run_loop(self):
        """Runs asyncio event loop in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ble_task())
        except Exception as e:
            self._status     = "error"
            self._status_msg = str(e)
            self._q.put(f"BLE Error: {e}")
        finally:
            loop.close()

    async def _ble_task(self):
        from bleak import BleakClient, BleakScanner

        while self._running:
            self._status     = "connecting"
            self._status_msg = f"Searching for '{self.device_name}'..."
            self._q.put(self._status_msg)

            try:
                device = await BleakScanner.find_device_by_name(
                    self.device_name, timeout=10.0)
            except Exception as e:
                self._status_msg = f"Scan error: {e}"
                self._q.put(self._status_msg)
                await asyncio.sleep(3)
                continue

            if device is None:
                self._status_msg = "Device not found by name, discovering nearby devices..."
                self._q.put(self._status_msg)
                try:
                    candidates = await BleakScanner.discover(timeout=8.0)
                except Exception as e:
                    self._status_msg = f"Discovery error: {e}"
                    self._q.put(self._status_msg)
                    await asyncio.sleep(3)
                    continue

                # Log discovered devices for debugging
                if candidates:
                    self._q.put(f"Discovered {len(candidates)} devices:")
                    for c in candidates:
                        name = getattr(c, 'name', 'Unknown')
                        addr = getattr(c, 'address', 'N/A')
                        self._q.put(f"  - {name} ({addr})")
                else:
                    self._q.put("No devices discovered.")

                for candidate in candidates:
                    name = (candidate.name or "").lower()
                    if self.device_name.lower() in name:
                        device = candidate
                        break
                    uuids = []
                    if hasattr(candidate, 'metadata'):
                        uuids = [u.lower() for u in candidate.metadata.get('uuids', []) or []]
                    if self.service_uuid.lower() in uuids:
                        device = candidate
                        break

                if device is None:
                    self._status_msg = "Device not found, retrying..."
                    self._q.put(self._status_msg)
                    await asyncio.sleep(3)
                    continue

                self._q.put(f"Connecting: {getattr(device, 'name', 'Unknown')} ({getattr(device, 'address', 'N/A')})")

            def on_notify(sender, data: bytearray):
                try:
                    decoded = data.decode("utf-8").strip()
                    parts   = decoded.split(",")
                    if len(parts) == 4:
                        ts, ecg_v, ppg_v, lo = parts
                        self._lead_off = (lo.strip() == "1")
                        if not self._lead_off:
                            with self._lock:
                                self._buffer.append(
                                    (int(ts), int(ecg_v), int(ppg_v)))
                except Exception:
                    pass   # Corrupted packet — skip

            try:
                async with BleakClient(device) as client:
                    self._status     = "connected"
                    self._status_msg = f"Connected: {device.name}"
                    self._q.put("✔ Connection successful — data stream started")

                    await client.start_notify(self.char_uuid, on_notify)

                    while self._running and client.is_connected:
                        await asyncio.sleep(0.5)

                    await client.stop_notify(self.char_uuid)

            except Exception as e:
                self._status     = "error"
                self._status_msg = f"Connection lost: {e}"
                self._q.put(self._status_msg)
                await asyncio.sleep(2)   # Short wait, retry

        self._status = "idle"
