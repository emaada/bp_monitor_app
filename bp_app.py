"""
bp_app.py — Blood Pressure Monitor Application
"""
 
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
import threading
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
 
from bp_core import (
    UserDB, CalibModel, CSVStreamer, BLEStreamer,
    calculate_pat_both, PATResult, SQIResult,
    FS, WIN_SEC, STEP_SEC
)
from scipy.signal import butter, filtfilt, find_peaks


# ════════════════════════════════════════════
# FINGER PAT CALCULATOR
# Wider timing windows for finger placement.
# Neck/ear PAT: 80–350 ms foot, 40–300 ms peak.
# Finger PAT:  200–600 ms foot, 150–550 ms peak.
# ════════════════════════════════════════════
def _finger_filter(ecg_raw: np.ndarray, ppg_raw: np.ndarray, fs: int = FS):
    nyq = 0.5 * fs
    b_ecg, a_ecg = butter(2, [0.5/nyq, 35/nyq], btype='band')
    b_ppg, a_ppg = butter(2, 8/nyq,              btype='low')
    ecg_f  = filtfilt(b_ecg, a_ecg, ecg_raw.astype(float))
    ppg_lp = filtfilt(b_ppg, a_ppg, ppg_raw.astype(float))
    return ecg_f, ppg_lp


def calculate_pat_finger(ecg_raw: np.ndarray, ppg_raw: np.ndarray,
                          fs: int = FS) -> PATResult:
    """
    PAT calculation tuned for finger PPG placement.
    Finger has ~150-300 ms longer transit time than neck/ear,
    so search windows are widened accordingly.

    FOOT: R+200 ms → R+600 ms  (trough / pulse onset)
    PEAK: R+150 ms → R+550 ms  (systolic peak)
    """
    if len(ecg_raw) < 15:
        return PATResult(None, None, SQIResult(False, "Signal too short"))

    try:
        ecg_f, ppg_lp = _finger_filter(ecg_raw, ppg_raw, fs)
    except Exception as e:
        return PATResult(None, None, SQIResult(False, str(e)))

    # ── SQI (same checks as bp_core) ──
    if np.ptp(ecg_f) < 0.01:
        return PATResult(None, None, SQIResult(False, "ECG amplitude too low"))
    if np.std(ppg_lp - np.mean(ppg_lp)) < 1e-6:
        return PATResult(None, None, SQIResult(False, "No PPG signal"))

    r_locs, _ = find_peaks(ecg_f,
                            distance=int(fs * 0.4),
                            height=np.mean(ecg_f) + 0.5 * np.std(ecg_f))
    if len(r_locs) < 3:
        return PATResult(None, None,
                         SQIResult(False, f"Insufficient R-peaks ({len(r_locs)})",
                                   n_peaks=len(r_locs)))

    n_peaks = len(r_locs)

    # Prominence: 2% of PPG range, no hard floor
    ppg_prom = max(1e-3, 0.02 * np.ptp(ppg_lp))

    foot_locs, _ = find_peaks(-ppg_lp, distance=int(fs * 0.3), prominence=ppg_prom)
    peak_locs, _ = find_peaks( ppg_lp, distance=int(fs * 0.3), prominence=ppg_prom)

    foot_pats, peak_pats = [], []

    for rp in r_locs:
        # FOOT: wider window 200–600 ms
        lo_f = rp + int(0.20 * fs)
        hi_f = rp + int(0.60 * fs)
        cands_f = foot_locs[(foot_locs >= lo_f) & (foot_locs < hi_f)]
        if len(cands_f):
            pat = (cands_f[0] - rp) * (1000.0 / fs)
            if 200 < pat < 600:
                foot_pats.append(pat)

        # PEAK: wider window 150–550 ms
        lo_p = rp + int(0.15 * fs)
        hi_p = rp + int(0.55 * fs)
        cands_p = peak_locs[(peak_locs >= lo_p) & (peak_locs < hi_p)]
        if len(cands_p):
            pat = (cands_p[0] - rp) * (1000.0 / fs)
            if 150 < pat < 550:
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
                        f"Finger PAT not found (foot:{len(foot_pats)}, peak:{len(peak_pats)})",
                        n_peaks=n_peaks)
    else:
        sqi = SQIResult(True, "OK (finger mode)", n_peaks=n_peaks)

    return PATResult(foot_med, peak_med, sqi, foot_pats, peak_pats)
 
# ════════════════════════════════════════════
# THEME 
# ════════════════════════════════════════════
C = {
    "bg":         "#F8FAFC",
    "card":       "#FFFFFF",
    "primary":    "#1E40AF",
    "primary_lt": "#3B82F6",
    "success":    "#059669",
    "success_lt": "#10B981",
    "warn":       "#D97706",
    "warn_lt":    "#F59E0B",
    "danger":     "#DC2626",
    "danger_lt":  "#EF4444",
    "text":       "#0F172A",
    "text_sub":   "#475569",
    "text_light": "#64748B",
    "border":     "#E2E8F0",
    "border_lt":  "#CBD5E1",
    "foot":       "#7C3AED",
    "peak":       "#0891B2",
    "ble":        "#0F766E",
    "shadow":     "#00000015",
}

FH1   = ("Segoe UI", 24, "bold")
FH2   = ("Segoe UI", 16, "bold")
FH3   = ("Segoe UI", 13, "bold")
FB    = ("Segoe UI", 11)
FB_B  = ("Segoe UI", 11, "bold")
FMONO = ("Courier New", 11)
FMONO_B = ("Courier New", 11, "bold")
FSM   = ("Segoe UI", 10)
FSM_B = ("Segoe UI", 10, "bold")
# ════════════════════════════════════════════
# TRANSLATIONS
# ════════════════════════════════════════════
LANG = {
    "en": {
        "app_title":         "BP Monitor — Blood Pressure Tracker",
        "app_heading":       "BP Monitor",
        "app_subtitle":      "Blood Pressure Tracking System",
        "select_user":       "Select user:",
        "login":             "Login",
        "delete_user":       "Delete User",
        "create_user":       "Create New User",
        "new_user_title":    "Create New User",
        "name":              "Name *",
        "age":               "Age *",
        "gender":            "Gender (M/F) *",
        "name_ph":           "John Doe",
        "age_ph":            "30",
        "gender_ph":         "M",
        "save":              "✔  Save",
        "cancel":            "Cancel",
        "missing":           "Missing",
        "fill_all":          "Please fill in all fields.",
        "age_numeric":       "Age must be a number.",
        "gender_mf":         "Gender must be M or F.",
        "already_exists":    "already exists.",
        "confirm":           "Confirm",
        "delete_confirm":    "Delete '{}'?",
        "warning":           "Warning",
        "select_user_warn":  "Please select a user.",
        "active_models":     "Active Models",
        "default_model":     "⚙️ Default",
        "personal_model":    "✅ Personal",
        "total_calib":       "Total calibrations: {}",
        "select_mode":       "Select Mode",
        "calibration_mode":  "🔧  Calibration Mode",
        "measurement_mode":  "📊  Measurement Mode",
        "calib_hint":        "Calibration: train both models with reference BP  |  Measurement: CSV or live BLE",
        "back":              "← Back",
        "back_dashboard":    "← Back to Dashboard",
        "age_lbl":           "Age: {}  |  {}",
        "calib_records":     "Calibration Records",
        "calib_hint2":       "You can delete a selected record or correct its BP value.",
        "no":                "#",
        "file":              "File",
        "label":             "Label",
        "pat_foot":          "PAT-Foot(ms)",
        "pat_peak":          "PAT-Peak(ms)",
        "sbp":               "SBP",
        "dbp":               "DBP",
        "delete_selected":   "🗑  Delete Selected",
        "edit_bp":           "✏️  Edit BP Value",
        "add_calib":         "Add New Calibration",
        "label_entry":       "Label:",
        "label_ph":          "morning reading",
        "sbp_lbl":           "SBP:",
        "dbp_lbl":           "DBP:",
        "select_csv":        "📂 Select CSV & Add",
        "delete_record":     "Delete record {}?\n({}, SBP={}/{})",
        "record_deleted":    "🗑  Record {} deleted. Model re-fitted.",
        "edit_record":       "Edit Record {}",
        "record_lbl":        "Record {}: {}",
        "file_lbl":          "File: {}",
        "valid_bp":          "Enter valid values.\nSBP: 60-220, DBP: 40-140",
        "valid_sbp_dbp":     "Enter valid SBP/DBP (SBP: 60-220, DBP: 40-140)",
        "csv_ecg_ppg":       "CSV must contain 'ecg' and 'ppg' columns.",
        "sqi_failed":        "❌  SQI failed: {}",
        "signal_quality":    "Signal Quality",
        "record_saved":      "✔  {}/{} mmHg [{}] saved",
        "model_updated":     "🎯  {} model updated  R²={}",
        "model_pending":     "⏳  {}: {}",
        "record_updated":    "✏️  Record {} updated → {}/{}",
        "algorithm":         "Algorithm",
        "foot_rb":           "🟣  Foot  (onset — pulse start)",
        "peak_rb":           "🔵  Peak  (systolic — pulse peak)",
        "data_source":       "Data Source",
        "csv_source":        "📂  CSV File  (test mode)",
        "ble_source":        "📡  BLE  (ESP32 live)",
        "default_warn":      "⚠️  {}/model is using defaults — Calibration recommended",
        "ble_not_conn":      "● BLE: not connected",
        "waiting":           "Waiting...",
        "start":             "▶  Start",
        "stop":              "⏹  Stop",
        "completed":         "✔  Completed",
        "stopped":           "Stopped",
        "meas_active":       "Measurement active ●",
        "lead_off":          "⚠️ Electrode connection lost!",
        "ble_starting":      "📡  Starting BLE connection...",
        "csv_log":           "▶  CSV: {}  ({} samples ~{}s)",
        "pat_none":          "{} PAT could not be calculated",
        "error":             "Error",
        "no_selection":      "No Selection",
        "select_record":     "Please select a record from the list.",
        "calib_csv_title":   "Calibration CSV",
        "meas_csv_title":    "Measurement CSV",
        "error_calc_mode":   "📉  Error Calculation Mode",
        "error_hint":        "MAE & RMSE: load ≥3 calibration CSVs, enter reference BP, then load a verification CSV",
        "calib_files":       "Calibration Files (≥ 3 required)",
        "add_calib_file":    "➕  Add Calibration CSV",
        "clear_all":         "🗑  Clear All",
        "file_n":            "File {}",
        "ref_sbp":           "Ref SBP:",
        "ref_dbp":           "Ref DBP:",
        "verif_file":        "Verification CSV",
        "add_verif_file":    "📂  Select Verification CSV",
        "run_error_calc":    "▶  Calculate Errors",
        "err_need_3":        "Add at least 3 calibration files with valid SBP/DBP.",
        "err_need_verif":    "Select a verification CSV file first.",
        "err_bad_bp":        "Invalid BP in row {}: SBP={}, DBP={}",
        "err_no_pat":        "PAT could not be extracted from file {}.",
        "err_results":       "Error Calculation Results",
        "formula_sbp":       "SBP = {:.4f} × PAT + {:.4f}",
        "formula_dbp":       "DBP = {:.4f} × PAT + {:.4f}",
        "mae_sbp":           "MAE SBP",
        "mae_dbp":           "MAE DBP",
        "rmse_sbp":          "RMSE SBP",
        "rmse_dbp":          "RMSE DBP",
        "n_windows":         "Windows",
        "method_lbl":        "Method:",
    },

    "tr": {
        "app_title":         "BP Monitor — Kan Basıncı Takip",
        "app_heading":       "BP Monitor",
        "app_subtitle":      "Kan Basıncı Takip Sistemi",
        "select_user":       "Kullanıcı seçin:",
        "login":             "Giriş Yap",
        "delete_user":       "Kullanıcı Sil",
        "create_user":       "Yeni Kullanıcı Oluştur",
        "new_user_title":    "Yeni Kullanıcı Oluştur",
        "name":              "İsim *",
        "age":               "Yaş *",
        "gender":            "Cinsiyet (E/K) *",
        "name_ph":           "Ahmet Yılmaz",
        "age_ph":            "30",
        "gender_ph":         "E",
        "save":              "✔  Kaydet",
        "cancel":            "İptal",
        "missing":           "Eksik",
        "fill_all":          "Tüm alanları doldurun.",
        "age_numeric":       "Yaş sayısal olmalı.",
        "gender_mf":         "Cinsiyet E veya K olmalı.",
        "already_exists":    "zaten var.",
        "confirm":           "Onay",
        "delete_confirm":    "'{}' silinsin mi?",
        "warning":           "Uyarı",
        "select_user_warn":  "Lütfen kullanıcı seçin.",
        "active_models":     "Aktif Modeller",
        "default_model":     "⚙️ Varsayılan",
        "personal_model":    "✅ Kişisel",
        "total_calib":       "Toplam kalibrasyon: {}",
        "select_mode":       "Mod Seçimi",
        "calibration_mode":  "🔧  Kalibrasyon Modu",
        "measurement_mode":  "📊  Ölçüm Modu",
        "calib_hint":        "Kalibrasyon: referans BP ile her iki modeli eğit  |  Ölçüm: CSV veya BLE canlı",
        "back":              "← Geri",
        "back_dashboard":    "← Panele Dön",
        "age_lbl":           "Yaş: {}  |  {}",
        "calib_records":     "Kalibrasyon Kayıtları",
        "calib_hint2":       "Seçili kaydı silebilir veya KB değerini düzeltebilirsin.",
        "no":                "#",
        "file":              "Dosya",
        "label":             "Etiket",
        "pat_foot":          "PAT-Foot(ms)",
        "pat_peak":          "PAT-Peak(ms)",
        "sbp":               "SBP",
        "dbp":               "DBP",
        "delete_selected":   "🗑  Seçili Kaydı Sil",
        "edit_bp":           "✏️  KB Değerini Düzelt",
        "add_calib":         "Yeni Kalibrasyon Ekle",
        "label_entry":       "Etiket:",
        "label_ph":          "sabah ölçümü",
        "sbp_lbl":           "SBP:",
        "dbp_lbl":           "DBP:",
        "select_csv":        "📂 CSV Seç & Ekle",
        "delete_record":     "{}.  kayıt silinsin mi?\n({}, SBP={}/{})",
        "record_deleted":    "🗑  Kayıt {} silindi. Model yeniden fit edildi.",
        "edit_record":       "Kayıt {} Düzenle",
        "record_lbl":        "Kayıt {}: {}",
        "file_lbl":          "Dosya: {}",
        "valid_bp":          "Geçerli değer girin.\nSBP:60-220, DBP:40-140",
        "valid_sbp_dbp":     "Geçerli SBP/DBP girin (SBP:60-220, DBP:40-140)",
        "csv_ecg_ppg":       "CSV 'ecg' ve 'ppg' sütunları içermeli.",
        "sqi_failed":        "❌  SQI başarısız: {}",
        "signal_quality":    "Sinyal Kalitesi",
        "record_saved":      "✔  {}/{} mmHg [{}] kaydedildi",
        "model_updated":     "🎯  {} model güncellendi  R²={}",
        "model_pending":     "⏳  {}: {}",
        "record_updated":    "✏️  Kayıt {} güncellendi → {}/{}",
        "algorithm":         "Algoritma",
        "foot_rb":           "🟣  Foot  (onset — nabız başlangıcı)",
        "peak_rb":           "🔵  Peak  (systolic — nabız tepesi)",
        "data_source":       "Veri Kaynağı",
        "csv_source":        "📂  CSV Dosyası  (test modu)",
        "ble_source":        "📡  BLE  (ESP32 canlı)",
        "default_warn":      "⚠️  {}/model varsayılan — Kalibrasyon önerilir",
        "ble_not_conn":      "● BLE: bağlı değil",
        "waiting":           "Bekleniyor...",
        "start":             "▶  Başlat",
        "stop":              "⏹  Durdur",
        "completed":         "✔  Tamamlandı",
        "stopped":           "Durduruldu",
        "meas_active":       "Ölçüm aktif ●",
        "lead_off":          "⚠️ Elektrot bağlantısı kopuk!",
        "ble_starting":      "📡  BLE bağlantısı başlatılıyor...",
        "csv_log":           "▶  CSV: {}  ({} örnek ~{}s)",
        "pat_none":          "{} PAT hesaplanamadı",
        "error":             "Hata",
        "no_selection":      "Seçim Yok",
        "select_record":     "Listeden bir kayıt seçin.",
        "calib_csv_title":   "Kalibrasyon CSV",
        "meas_csv_title":    "Ölçüm CSV",
        "error_calc_mode":   "📉  Hata Hesaplama Modu",
        "error_hint":        "MAE & RMSE: ≥3 kalibrasyon CSV'si yükle, referans KB gir, doğrulama CSV'si seç",
        "calib_files":       "Kalibrasyon Dosyaları (en az 3)",
        "add_calib_file":    "➕  Kalibrasyon CSV Ekle",
        "clear_all":         "🗑  Tümünü Temizle",
        "file_n":            "Dosya {}",
        "ref_sbp":           "Ref SKB:",
        "ref_dbp":           "Ref DKB:",
        "verif_file":        "Doğrulama CSV",
        "add_verif_file":    "📂  Doğrulama CSV Seç",
        "run_error_calc":    "▶  Hataları Hesapla",
        "err_need_3":        "En az 3 kalibrasyon dosyası ve geçerli KB değeri ekleyin.",
        "err_need_verif":    "Önce bir doğrulama CSV dosyası seçin.",
        "err_bad_bp":        "Satır {} geçersiz KB: SKB={}, DKB={}",
        "err_no_pat":        "{} dosyasından PAT çıkarılamadı.",
        "err_results":       "Hata Hesaplama Sonuçları",
        "formula_sbp":       "SKB = {:.4f} × PAT + {:.4f}",
        "formula_dbp":       "DKB = {:.4f} × PAT + {:.4f}",
        "mae_sbp":           "MAE SKB",
        "mae_dbp":           "MAE DKB",
        "rmse_sbp":          "RMSE SKB",
        "rmse_dbp":          "RMSE DKB",
        "n_windows":         "Pencere",
        "method_lbl":        "Yöntem:",
    }
}
 
 
# ── Widget helpers ───────────────────────────
def card(p, **kw):
    """Styled card with shadow effect"""
    return tk.Frame(p, bg=C["card"], relief="flat", bd=0,
                    highlightbackground=C["border"], highlightthickness=1, **kw)

def card_shadow(p, **kw):
    """Styled card with accent left border"""
    return tk.Frame(p, bg=C["card"], relief="flat", bd=0,
                    highlightbackground=C["border_lt"], highlightthickness=1, **kw)

def scrollable_frame(parent, bg=None):
    """Create a vertically scrollable content frame."""
    outer = tk.Frame(parent, bg=bg or parent.cget("bg"))
    canvas = tk.Canvas(outer, bg=bg or parent.cget("bg"), highlightthickness=0)
    scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    frame = tk.Frame(canvas, bg=bg or parent.cget("bg"))
    window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        if canvas.winfo_width() != frame.winfo_reqwidth():
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

    frame.bind("<Configure>", _on_frame_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    return outer, frame

def lbl(p, text, font=FB, fg=None, **kw):
    bg = kw.pop("bg", None) or p.cget("bg")
    return tk.Label(p, text=text, font=font,
                    fg=fg or C["text"], bg=bg, **kw)

def btn(p, text, cmd, color=None, width=14, size="normal", **kw):
    """Improved button with better styling"""
    col = color or C["primary"]
    font_size = ("Segoe UI", 10, "bold") if size == "normal" else ("Segoe UI", 9, "bold")
    return tk.Button(p, text=text, command=cmd, bg=col, fg="white",
                     activebackground=col, font=font_size,
                     relief="flat", bd=0, cursor="hand2",
                     padx=12, pady=8, width=width, **kw)

def entry(p, **kw):
    """Improved entry field"""
    return tk.Entry(p, font=FB, relief="solid", bd=1,
                    highlightthickness=1, highlightcolor=C["primary_lt"],
                    bg="white", fg=C["text"], **kw)


# ════════════════════════════════════════════
# GRAPH HELPERS
# ════════════════════════════════════════════
def create_dual_graph(parent, title1="ECG", title2="PPG"):
    """Create dual subplot figure for ECG and PPG"""
    fig = Figure(figsize=(10, 3), dpi=80, facecolor=C["card"])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    for ax, title, color in [(ax1, title1, C["primary"]), (ax2, title2, C["peak"])]:
        ax.set_title(title, fontsize=10, fontweight='bold', color=C["text"])
        ax.set_xlabel("Time (s)", fontsize=9, color=C["text_light"])
        ax.set_ylabel("Amplitude", fontsize=9, color=C["text_light"])
        ax.tick_params(colors=C["text_light"], labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color(C["border"])
    
    fig.tight_layout(pad=2)
    
    canvas = FigureCanvasTkAgg(fig, parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, ax1, ax2, canvas

def create_triple_graph(parent, title1="ECG", title2="PPG", title3="BP"):
    """Create triple subplot figure for ECG, PPG, and BP"""
    fig = Figure(figsize=(12, 3), dpi=80, facecolor=C["card"])
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    colors = [C["primary"], C["peak"], C["success"]]
    for ax, title, color in [(ax1, title1, colors[0]), (ax2, title2, colors[1]), (ax3, title3, colors[2])]:
        ax.set_title(title, fontsize=10, fontweight='bold', color=C["text"])
        ax.set_xlabel("Time (s)" if title != title3 else "Sample #", fontsize=9, color=C["text_light"])
        ax.set_ylabel("Amplitude" if title != title3 else "mmHg", fontsize=9, color=C["text_light"])
        ax.tick_params(colors=C["text_light"], labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color(C["border"])
    
    fig.tight_layout(pad=2)
    
    canvas = FigureCanvasTkAgg(fig, parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, ax1, ax2, ax3, canvas


# ════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════
class BPApp:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.lang     = "en"   # default language
        self.db       = UserDB()
        self.current_user = None
        self.after_id     = None
        self.ble_streamer: BLEStreamer = None
        self.csv_streamer: CSVStreamer = None
        self._calib_ble_streamer: BLEStreamer = None
        self._calib_after_id = None
        self._calib_pending = None

        # ── File logging ──
        os.makedirs("logs", exist_ok=True)
        self._calib_log_path = os.path.join("logs", "calibration_log.txt")

        self._apply_title()
        self.root.geometry("1200x900")
        self.root.configure(bg=C["bg"])
        self._show_login()
 
    def t(self, key, *args):
        """Get translated string, optionally formatting with args."""
        s = LANG[self.lang].get(key, key)
        return s.format(*args) if args else s
 
    def _apply_title(self):
        self.root.title(self.t("app_title"))
 
    def _toggle_lang(self):
        self.lang = "tr" if self.lang == "en" else "en"
        self._apply_title()
        screen = getattr(self, "_current_screen", self._show_login)
        screen()
 
    def _lang_btn(self, parent):
        """Language toggle button — always top-left."""
        flag = "🇹🇷 TR" if self.lang == "en" else "🇬🇧 EN"
        b = tk.Button(parent, text=flag, command=self._toggle_lang,
                      bg=parent["bg"], fg="white",
                      font=("Segoe UI", 9, "bold"),
                      relief="flat", bd=0, cursor="hand2",
                      padx=10, pady=4, activebackground=parent["bg"])
        b.pack(side="left", padx=(0, 16))
        return b
 
    def _clear(self):
        self._stop_streams()
        self._stop_calib_streams()
        for w in self.root.winfo_children():
            w.destroy()
 
    def _stop_streams(self):
        if self.ble_streamer:
            self.ble_streamer.stop()
            self.ble_streamer = None
        self.csv_streamer = None

    def _stop_calib_streams(self):
        if self._calib_ble_streamer:
            self._calib_ble_streamer.stop()
            self._calib_ble_streamer = None
        if self._calib_after_id:
            self.root.after_cancel(self._calib_after_id)
            self._calib_after_id = None
        self._calib_pending = None
        # CSV dosyasını kapat
        if getattr(self, "_calib_raw_csv", None) is not None:
            try:
                self._calib_raw_csv.close()
            except Exception:
                pass
            self._calib_raw_csv      = None
            self._calib_raw_csv_written = 0
 
    # ════════════════════════════════
    # LOGIN SCREEN
    # ════════════════════════════════
    def _show_login(self):
        self._current_screen = self._show_login
        self._clear()
        self._stop_streams()
        self.current_user = None

        # ── Top Bar ──
        top = tk.Frame(self.root, bg=C["primary"], padx=16, pady=12)
        top.pack(fill="x")
        self._lang_btn(top)
        tk.Label(top, text=self.t("app_heading"), font=FH2,
                 fg="white", bg=C["primary"]).pack(side="left")

        # ── Main Content ──
        outer = tk.Frame(self.root, bg=C["bg"])
        outer.pack(expand=True, fill="both", padx=20, pady=40)

        # Header
        lbl(outer, self.t("app_heading"), font=FH1, fg=C["primary"]).pack(pady=(0, 4))
        lbl(outer, self.t("app_subtitle"), font=FSM, fg=C["text_light"]).pack(pady=(0, 32))

        # Login Card
        c = card(outer, padx=32, pady=28)
        c.pack(ipadx=16, ipady=16, fill="both", expand=False)

        lbl(c, self.t("select_user"), font=FH3, fg=C["text"]).pack(anchor="w", pady=(0, 12))

        self._uvar = tk.StringVar()
        combo = ttk.Combobox(c, textvariable=self._uvar,
                             values=self.db.list_users(),
                             state="readonly", font=FB, width=30)
        combo.config(postcommand=lambda: combo.config(values=self.db.list_users()))
        combo.pack(pady=(0, 20), fill="x")
        combo.bind("<Return>", lambda e: self._login())

        # Buttons
        bf = tk.Frame(c, bg=C["card"])
        bf.pack(pady=(0, 12), fill="x")
        btn(bf, self.t("login"), self._login, C["success"], width=28).pack(side="left", fill="x", expand=True, padx=(0, 8))
        btn(bf, self.t("delete_user"), self._delete_user, C["danger"], width=20).pack(side="left", fill="x", expand=False)

        # Separator
        tk.Frame(c, bg=C["border"], height=2).pack(fill="x", pady=16)

        # Create New User Button
        btn(c, self.t("create_user"), self._new_user_window,
            C["primary_lt"], width=34).pack(fill="x")
 
    def _login(self):
        name = self._uvar.get()
        if not name:
            messagebox.showwarning(self.t("warning"), self.t("select_user_warn")); return
        self.current_user = name
        self._show_dashboard()
 
    def _delete_user(self):
        name = self._uvar.get()
        if name and messagebox.askyesno(self.t("confirm"), self.t("delete_confirm", name)):
            self.db.delete_user(name)
            self._show_login()
 
    # ── New User Window ───────────────────────
    def _new_user_window(self):
        win = tk.Toplevel(self.root)
        win.title(self.t("new_user_title"))
        win.geometry("380x360")
        win.resizable(False, False)
        win.configure(bg=C["bg"])
        win.grab_set(); win.focus_force()

        # Header
        hdr = tk.Frame(win, bg=C["primary"], height=50)
        hdr.pack(fill="x")
        tk.Label(hdr, text=self.t("new_user_title"),
                 font=FH3, fg="white", bg=C["primary"]).place(x=20, y=12)

        # Body
        body = tk.Frame(win, bg=C["bg"], padx=28, pady=20)
        body.pack(fill="both", expand=True)

        fields = {}
        gender_valid = ("E","K") if self.lang == "tr" else ("M","F")
        for label_txt, key, ph in [
            (self.t("name"),   "name",   self.t("name_ph")),
            (self.t("age"),    "age",    self.t("age_ph")),
            (self.t("gender"), "gender", self.t("gender_ph")),
        ]:
            lbl(body, label_txt, font=FSM_B, fg=C["text_sub"]).pack(anchor="w", pady=(10, 3))
            e = entry(body, width=36)
            e.pack(fill="x", pady=(0, 8))
            e.insert(0, ph)
            e.config(fg=C["text_light"])

            def _fi(ev, w=e, p=ph):
                if w.get() == p:
                    w.delete(0, "end")
                    w.config(fg=C["text"])

            def _fo(ev, w=e, p=ph):
                if not w.get().strip():
                    w.insert(0, p)
                    w.config(fg=C["text_light"])

            e.bind("<FocusIn>", _fi)
            e.bind("<FocusOut>", _fo)
            fields[key] = (e, ph)

        def get(key):
            e, ph = fields[key]
            v = e.get().strip()
            # If the field still has placeholder styling AND value matches placeholder → empty
            try:
                is_placeholder = (e.cget("fg") == C["text_light"] and v == ph)
            except Exception:
                is_placeholder = (v == ph)
            return "" if is_placeholder else v

        def save():
            name, age, gender = get("name"), get("age"), get("gender")
            if not all([name, age, gender]):
                messagebox.showwarning(self.t("missing"), self.t("fill_all"), parent=win)
                return
            if not age.isdigit():
                messagebox.showwarning(self.t("error"), self.t("age_numeric"), parent=win)
                return
            if gender.upper() not in gender_valid:
                messagebox.showwarning(self.t("error"), self.t("gender_mf"), parent=win)
                return
            if not self.db.add_user(name, int(age), gender.upper()):
                messagebox.showwarning(self.t("error"), f"'{name}' {self.t('already_exists')}", parent=win)
                return
            win.destroy()
            self._show_login()

        win.bind("<Return>", lambda e: save())
        
        # Buttons
        btn_frame = tk.Frame(body, bg=C["bg"])
        btn_frame.pack(pady=(16, 0), fill="x")
        btn(btn_frame, self.t("save"), save, C["success"], width=32).pack(fill="x", pady=(0, 8))
        btn(btn_frame, self.t("cancel"), win.destroy, C["text_light"], width=32).pack(fill="x")

        e0 = fields["name"][0]
        e0.delete(0, "end")
        e0.config(fg=C["text"])
        e0.focus_set()
 
    # ════════════════════════════════
    # DASHBOARD
    # ════════════════════════════════
    def _show_dashboard(self):
        self._current_screen = self._show_dashboard
        self._clear()
        ud    = self.db.users[self.current_user]
        model = ud["model"]
        info  = ud["info"]

        # ── Top Bar ──
        top = tk.Frame(self.root, bg=C["primary"], padx=16, pady=14)
        top.pack(fill="x")
        self._lang_btn(top)
        lbl(top, f"👤  {self.current_user}", font=FH3, fg="white").pack(side="left")
        lbl(top, self.t("age_lbl", info['age'], info['gender']),
            font=FSM, fg="white", bg=C["primary"]).pack(side="left", padx=20)
        btn(top, self.t("back"), self._show_login, C["primary"], width=8, size="small").pack(side="right")

        # ── Body ──
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=20)

        # ── Active Models Card ──
        mc = card(body, padx=20, pady=16)
        mc.pack(fill="x", pady=(0, 16))
        lbl(mc, self.t("active_models"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 12))

        for method, label, color, s_a, s_b, d_a, d_b, using_def in [
            ("foot", "🟣 Foot (onset):", C["foot"],
             model.foot_s_a, model.foot_s_b, model.foot_d_a, model.foot_d_b,
             model.foot_using_defaults),
            ("peak", "🔵 Peak (systolic):", C["peak"],
             model.peak_s_a, model.peak_s_b, model.peak_d_a, model.peak_d_b,
             model.peak_using_defaults),
        ]:
            rf = tk.Frame(mc, bg=C["card"])
            rf.pack(fill="x", pady=(0, 8))
            
            # Model status
            st = self.t("default_model") if using_def else self.t("personal_model")
            sc = C["warn_lt"] if using_def else C["success_lt"]
            
            info_frame = tk.Frame(rf, bg=C["card"])
            info_frame.pack(fill="x", padx=0, pady=0)
            
            tk.Label(info_frame, text=f"{label}  {st}", font=FB_B,
                     fg=sc, bg=C["card"]).pack(side="left", anchor="w")
            
            # Coefficients
            coeff = f"SBP={s_a:+.3f}×PAT+{s_b:.1f}  |  DBP={d_a:+.3f}×PAT+{d_b:.1f}"
            tk.Label(rf, text=coeff, font=FMONO,
                     fg=color, bg=C["card"]).pack(side="left", padx=12, anchor="w")

        lbl(mc, self.t("total_calib", len(model.calibrations)),
            font=FSM, fg=C["text_light"]).pack(anchor="w", pady=(8, 0))

        # ── Mode Selection Card ──
        mf = card(body, padx=20, pady=16)
        mf.pack(fill="x", pady=(0, 16))
        lbl(mf, self.t("select_mode"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 12))
        
        bf = tk.Frame(mf, bg=C["card"])
        bf.pack(fill="x")
        btn(bf, self.t("calibration_mode"), self._show_calibration,
            C["warn_lt"], width=24).pack(side="left", fill="x", expand=True, padx=(0, 8))
        btn(bf, self.t("measurement_mode"), self._show_measurement,
            C["success_lt"], width=24).pack(side="left", fill="x", expand=True, padx=(0, 8))
        btn(bf, self.t("error_calc_mode"), self._show_error_calc,
            C["danger"], width=24).pack(side="left", fill="x", expand=True)
        
        lbl(mf, self.t("calib_hint"), font=FSM, fg=C["text_light"]).pack(pady=(12, 0), anchor="w")

 
    # ════════════════════════════════
    # CALIBRATION MODE
    # ════════════════════════════════
    def _show_calibration(self):
        self._current_screen = self._show_calibration
        self._clear()
        self._build_top_bar(self.t("calibration_mode"), C["warn_lt"])
        model: CalibModel = self.db.users[self.current_user]["model"]

        body_outer, body = scrollable_frame(self.root, bg=C["bg"])
        body_outer.pack(fill="both", expand=True, padx=20, pady=16)

        # ── Graph section (ECG/PPG) ──
        graph_frame = card(body, padx=12, pady=12)
        graph_frame.pack(fill="both", expand=True, pady=(0, 12))
        
        lbl(graph_frame, "Signal Visualization", font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 8))
        self._calib_fig, self._calib_ax1, self._calib_ax2, self._calib_canvas = create_dual_graph(graph_frame, "ECG Signal", "PPG Signal")
        self._calib_lines = []  # Store line objects for updating

        # ── Calibration Records + Controls ──
        records_frame = card(body, padx=16, pady=14)
        records_frame.pack(fill="x", pady=(0, 12))

        lbl(records_frame, self.t("calib_records"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 4))
        lbl(records_frame, self.t("calib_hint2"), font=FSM, fg=C["text_light"]).pack(anchor="w", pady=(0, 8))

        cols = ("use", "no", "file", "label", "pat_foot", "pat_peak", "sbp", "dbp")
        heads = {"use": "Reg?", "no": self.t("no"), "file": self.t("file"),
                 "label": self.t("label"), "pat_foot": self.t("pat_foot"),
                 "pat_peak": self.t("pat_peak"), "sbp": self.t("sbp"), "dbp": self.t("dbp")}
        widths = {"use": 42, "no": 30, "file": 140, "label": 90,
                  "pat_foot": 100, "pat_peak": 100, "sbp": 55, "dbp": 55}

        tree_frame = tk.Frame(records_frame, bg=C["card"])
        tree_frame.pack(fill="both", expand=False, pady=(0, 10))

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                            height=5, selectmode="browse")
        for col in cols:
            tree.heading(col, text=heads[col])
            tree.column(col, width=widths[col], anchor="center")
        tree.tag_configure("excluded", foreground=C["text_light"])
        tree.tag_configure("included", foreground=C["text"])

        sb_tree = tk.Scrollbar(tree_frame, command=tree.yview)
        tree.config(yscrollcommand=sb_tree.set)
        sb_tree.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        self._tree = tree
        self._refresh_tree(model)
        self._log_calib_records_to_file(model)

        lb = tk.Frame(records_frame, bg=C["card"])
        lb.pack(pady=(8, 0), anchor="w", fill="x")
        btn(lb, self.t("delete_selected"), lambda: self._delete_cal(model),
            C["danger"], width=16, size="small").pack(side="left", padx=(0, 8))
        btn(lb, self.t("edit_bp"), lambda: self._edit_cal(model),
            C["warn_lt"], width=16, size="small").pack(side="left", padx=(0, 8))
        btn(lb, "☑  Toggle Reg", lambda: self._toggle_reg_use(model),
            C["primary_lt"], width=14, size="small").pack(side="left", padx=(0, 8))
        btn(lb, "🔄  Refit Selected", lambda: self._refit_selected(model),
            C["ble"], width=16, size="small").pack(side="left")

        # ── Add New Calibration ──
        af = card(body, padx=16, pady=12)
        af.pack(fill="x", pady=(0, 12))

        lbl(af, self.t("add_calib"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 10))

        # Source selection
        src_row = tk.Frame(af, bg=C["card"])
        src_row.pack(fill="x", pady=(0, 10))
        lbl(src_row, "Source:", font=FSM_B, fg=C["text_sub"]).pack(side="left", padx=(0, 12))
        self._calib_source_var = tk.StringVar(value="csv")
        tk.Radiobutton(src_row, text="📂 CSV File", variable=self._calib_source_var,
                       value="csv", font=FB, bg=C["card"],
                       activebackground=C["card"], selectcolor=C["card"],
                       command=self._update_calib_source_ui).pack(side="left", padx=(0, 16))
        tk.Radiobutton(src_row, text="📡 BLE Live", variable=self._calib_source_var,
                       value="ble", font=FB, bg=C["card"],
                       activebackground=C["card"], selectcolor=C["card"],
                       command=self._update_calib_source_ui).pack(side="left")

        # Input fields
        row = tk.Frame(af, bg=C["card"])
        row.pack(fill="x")

        lbl(row, self.t("label_entry"), font=FSM_B, fg=C["text_sub"]).grid(row=0, column=0, padx=(0, 8), pady=4, sticky="w")
        self._lbl_e = entry(row, width=14)
        self._lbl_e.grid(row=0, column=1, padx=(0, 16), sticky="w")
        self._lbl_e.insert(0, self.t("label_ph"))

        lbl(row, self.t("sbp_lbl"), font=FSM_B, fg=C["text_sub"]).grid(row=0, column=2, padx=(0, 6))
        self._sbp_e = entry(row, width=8)
        self._sbp_e.grid(row=0, column=3, padx=(0, 12))

        lbl(row, self.t("dbp_lbl"), font=FSM_B, fg=C["text_sub"]).grid(row=0, column=4, padx=(0, 6))
        self._dbp_e = entry(row, width=8)
        self._dbp_e.grid(row=0, column=5, padx=(0, 16))

        self._calib_button_row = tk.Frame(af, bg=C["card"])
        self._calib_button_row.pack(fill="x", pady=(10, 0))

        # CSV button (visible only when source=csv)
        self._calib_csv_btn = btn(self._calib_button_row, self.t("select_csv"),
                                  lambda: self._add_calibration(model),
                                  C["warn"], width=16, size="small")

        # BLE Start/Stop placeholders — parented here but shown via _update_calib_source_ui
        self._calib_start_btn = btn(self._calib_button_row, self.t("start"),
                                    lambda: self._add_calibration(model),
                                    C["ble"], width=16, size="small")
        self._calib_stop_btn = btn(self._calib_button_row, self.t("stop"),
                                   self._stop_calibration,
                                   C["danger"], width=12, size="small", state="disabled")

        self._calib_status_lbl = lbl(self._calib_button_row, "", font=FSM, fg=C["text_light"])
        self._calib_status_lbl.pack(side="left", padx=(10, 0))
        self._update_calib_source_ui()

        # Log
        lc = card(body, padx=10, pady=8)
        lc.pack(fill="both", expand=True, pady=(0, 8))
        self._calib_log = tk.Text(lc, height=3, font=FSM,
                                  bg="white", relief="solid", bd=1, state="disabled",
                                  fg=C["text"])
        sb_log = tk.Scrollbar(lc, command=self._calib_log.yview)
        self._calib_log.config(yscrollcommand=sb_log.set)
        sb_log.pack(side="right", fill="y")
        self._calib_log.pack(fill="both", expand=True)

        # ── Control Buttons (bottom bar — mirrors measurement mode) ──
        self._calib_bottom_bar = tk.Frame(body, bg=C["bg"])
        self._calib_bottom_bar.pack(fill="x")

        # BLE-mode Start/Stop (hidden by default until BLE selected)
        self._calib_bb_start = btn(self._calib_bottom_bar, self.t("start"),
                                   lambda: self._add_calibration(model),
                                   C["ble"], width=18, size="small")
        self._calib_bb_stop = btn(self._calib_bottom_bar, self.t("stop"),
                                  self._stop_calibration,
                                  C["danger"], width=12, size="small", state="disabled")
        self._calib_bb_back = btn(self._calib_bottom_bar, self.t("back_dashboard"),
                                  self._show_dashboard, C["text_light"], width=18, size="small")
        self._update_calib_bottom_bar()
 
    # ── Calibration helpers ───────────────────
 
    def _refresh_tree(self, model: CalibModel):
        self._tree.delete(*self._tree.get_children())
        for i, c in enumerate(model.calibrations):
            pf = f"{c['pat_foot']:.1f}" if c.get("pat_foot") is not None else "—"
            pp = f"{c['pat_peak']:.1f}" if c.get("pat_peak") is not None else "—"
            use = c.get("use_in_regression", True)
            self._tree.insert("", "end", iid=str(i),
                              tags=("included" if use else "excluded",),
                              values=("✓" if use else "✗", i+1,
                                      os.path.basename(c.get("file","")) or "—",
                                      c.get("label","") or "—",
                                      pf, pp, f"{c['sbp']:.0f}", f"{c['dbp']:.0f}"))

    def _log_calib_records_to_file(self, model: CalibModel):
        """Write the full calibration records table to the log file."""
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._calib_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] [{self.current_user}] === Calibration Records ===\n")
                header = f"  {'#':>3}  {'File':<20}  {'Label':<16}  {'PAT-Foot(ms)':>12}  {'PAT-Peak(ms)':>12}  {'SBP':>5}  {'DBP':>5}\n"
                f.write(header)
                f.write("  " + "-" * (len(header) - 3) + "\n")
                if not model.calibrations:
                    f.write("  (no records)\n")
                for i, c in enumerate(model.calibrations):
                    pf = f"{c['pat_foot']:.1f}" if c.get("pat_foot") is not None else "—"
                    pp = f"{c['pat_peak']:.1f}" if c.get("pat_peak") is not None else "—"
                    fname = os.path.basename(c.get("file", "")) or "—"
                    label = c.get("label", "") or "—"
                    f.write(f"  {i+1:>3}  {fname:<20}  {label:<16}  {pf:>12}  {pp:>12}  {int(c['sbp']):>5}  {int(c['dbp']):>5}\n")
                f.write("\n")
        except Exception:
            pass
 
    def _selected_idx(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning(self.t("no_selection"), self.t("select_record"))
            return None
        return int(sel[0])

    def _toggle_reg_use(self, model: CalibModel):
        idx = self._selected_idx()
        if idx is None:
            return
        c = model.calibrations[idx]
        c["use_in_regression"] = not c.get("use_in_regression", True)
        self.db.save()
        self._refresh_tree(model)
        self._log_calib(f"Kayıt {idx+1} regresyon: {'✓ dahil' if c['use_in_regression'] else '✗ hariç'}")

    def _refit_selected(self, model: CalibModel):
        selected = [c for c in model.calibrations if c.get("use_in_regression", True)]
        if len(selected) < 2:
            messagebox.showwarning("Yetersiz Kayıt", "En az 2 seçili (✓) kayıt gerekli."); return

        def _linfit(xs, ys):
            x, y = np.array(xs, float), np.array(ys, float)
            a, b = np.polyfit(x, y, 1)
            ss_res = np.sum((y - (a*x + b))**2)
            ss_tot = np.sum((y - y.mean())**2)
            return float(a), float(b), float(1 - ss_res/ss_tot if ss_tot > 0 else 0)

        fp = [(c["pat_foot"], c["sbp"], c["dbp"]) for c in selected if c.get("pat_foot")]
        if len(fp) >= 2:
            pf, sf, df_ = zip(*fp); sa, sb_, r2s = _linfit(pf, sf); da, db, r2d = _linfit(pf, df_)
            model.foot_s_a, model.foot_s_b = sa, sb_
            model.foot_d_a, model.foot_d_b = da, db
            model.foot_using_defaults = False
            self._log_calib(f"🔄 Foot SBP: {sa:.4f}×PAT+{sb_:.2f} R²={r2s:.3f}")
            self._log_calib(f"🔄 Foot DBP: {da:.4f}×PAT+{db:.2f} R²={r2d:.3f}")

        pp_ = [(c["pat_peak"], c["sbp"], c["dbp"]) for c in selected if c.get("pat_peak")]
        if len(pp_) >= 2:
            pp, sp, dp = zip(*pp_); sa2, sb2, r2s2 = _linfit(pp, sp); da2, db2, r2d2 = _linfit(pp, dp)
            model.peak_s_a, model.peak_s_b = sa2, sb2
            model.peak_d_a, model.peak_d_b = da2, db2
            model.peak_using_defaults = False
            self._log_calib(f"🔄 Peak SBP: {sa2:.4f}×PAT+{sb2:.2f} R²={r2s2:.3f}")
            self._log_calib(f"🔄 Peak DBP: {da2:.4f}×PAT+{db2:.2f} R²={r2d2:.3f}")

        self.db.save()
        n, t = len(selected), len(model.calibrations)
        messagebox.showinfo("Refit Tamamlandı", f"{n}/{t} kayıt ile model yeniden fit edildi.")
        self._log_calib(f"✅ Refit: {n}/{t} kayıt")
 
    def _update_calib_source_ui(self):
        if getattr(self, '_calib_csv_btn', None) is None:
            return
        try:
            if not self._calib_csv_btn.winfo_exists():
                return
        except Exception:
            return
        for w in (self._calib_csv_btn, self._calib_start_btn,
                  self._calib_stop_btn, self._calib_status_lbl):
            try:
                w.pack_forget()
            except Exception:
                pass
        if self._calib_source_var.get() == "ble":
            ble_running = bool(getattr(self, '_calib_ble_streamer', None))
            try:
                self._calib_start_btn.config(state="disabled" if ble_running else "normal")
                self._calib_stop_btn.config(state="normal" if ble_running else "disabled")
                self._calib_start_btn.pack(side="left", padx=(0, 8))
                self._calib_stop_btn.pack(side="left",  padx=(0, 8))
                self._calib_status_lbl.pack(side="left", padx=(10, 0))
            except Exception:
                pass
        else:
            try:
                self._calib_csv_btn.pack(side="left", padx=(0, 8))
                self._calib_status_lbl.pack(side="left", padx=(10, 0))
            except Exception:
                pass
        try:
            self._calib_status_lbl.config(text="", fg=C["text_light"])
        except Exception:
            pass

    def _update_calib_bottom_bar(self):
        bb_back = getattr(self, '_calib_bb_back', None)
        if bb_back is None:
            return
        try:
            if not bb_back.winfo_exists():
                return
        except Exception:
            return
        for w in (getattr(self, '_calib_bb_start', None),
                  getattr(self, '_calib_bb_stop', None)):
            if w is not None:
                try:
                    w.pack_forget()
                except Exception:
                    pass
        try:
            bb_back.pack_forget()
            bb_back.pack(side="left", fill="x", expand=True)
        except Exception:
            pass

    def _set_calib_status(self, msg: str, fg: str = None):
        try:
            if self._calib_status_lbl.winfo_exists():
                self._calib_status_lbl.config(text=msg, fg=fg or C["text_light"])
        except Exception:
            pass

    def _stop_calibration(self):
        self._stop_calib_streams()
        self._set_calib_status(self.t("stopped"), fg=C["danger"])
        try:
            if self._calib_start_btn.winfo_exists():
                self._calib_start_btn.config(state="normal")
                self._calib_stop_btn.config(state="disabled")
        except Exception:
            pass

    def _start_calib_ble(self, model: CalibModel, sbp: float, dbp: float, label: str):
        if self._calib_ble_streamer:
            self._log_calib("❗ BLE calibration already in progress.")
            return

        self._stop_calib_streams()
        self._calib_pending = {"model": model, "sbp": sbp, "dbp": dbp, "label": label}
        self._calib_ble_streamer = BLEStreamer()
        self._calib_ble_streamer.start()
        self._calib_tick_count    = 0
        self._calib_prev_lead_off = None

        # ── CSV dosyasını aç (Start'a her basışta yeni dosya) ──
        try:
            os.makedirs("raw_records", exist_ok=True)
            safe_user  = self.current_user.replace(" ", "_")
            ts_str     = time.strftime("%Y%m%d_%H%M%S")
            self._calib_session_dt = time.strftime("%Y-%m-%d %H:%M:%S")
            csv_path   = os.path.join("raw_records", f"{safe_user}_{ts_str}.csv")
            self._calib_raw_csv         = open(csv_path, "w", newline="", encoding="utf-8")
            self._calib_raw_csv_written = 0   # kaç satır zaten yazıldı
            # başlık satırı
            self._calib_raw_csv.write("person_name,session_datetime,timestamp,ecg,ppg\n")
            self._calib_raw_csv.flush()
            self._log_calib(f"💾  Raw CSV açıldı: {csv_path}")
        except Exception as e:
            self._calib_raw_csv         = None
            self._calib_raw_csv_written = 0
            self._log_calib(f"⚠️  Raw CSV açılamadı: {e}")

        try:
            self._calib_start_btn.config(state="disabled")
            self._calib_stop_btn.config(state="normal")
        except Exception:
            pass
        self._set_calib_status(self.t("ble_starting"), fg=C["ble"])
        self._log_calib("\n📡  Starting BLE calibration...")
        self._calib_tick_ble()

    def _calib_tick_ble(self):
        if self._calib_ble_streamer is None:
            return

        # ── Status messages ──
        while True:
            msg = self._calib_ble_streamer.get_status_message()
            if msg is None:
                break
            self._log_calib(f"  [{msg}]")
            color = C["ble"] if "✔" in msg else (C["danger"] if "error" in msg.lower() or "lost" in msg.lower() else C["text_light"])
            self._set_calib_status(msg, fg=color)

        # ── Lead-off detection ──
        lead_off_now = self._calib_ble_streamer.lead_off
        prev_lead_off = getattr(self, "_calib_prev_lead_off", None)
        if lead_off_now != prev_lead_off:
            # State changed — log it
            if lead_off_now:
                self._log_calib("⚠️  Lead-off detected: electrode connection lost")
            else:
                self._log_calib("✅  Lead-off cleared: electrodes reconnected")
            self._calib_prev_lead_off = lead_off_now

        if lead_off_now:
            self._set_calib_status(self.t("lead_off"), fg=C["danger"])

        # ── Buffer preview + progress ──
        preview = self._calib_ble_streamer.peek_buffer(max_samples=2000)
        buf_len = len(preview) if preview is not None else 0
        needed = int(WIN_SEC * FS)
        pct = min(100, int(100 * buf_len / needed)) if needed > 0 else 0

        # Update status with buffer progress when not lead-off
        if not lead_off_now:
            self._set_calib_status(
                f"📡  Collecting… {buf_len}/{needed} samples ({pct}%)",
                fg=C["ble"]
            )

        # Log buffer size every ~10 ticks (5 s) to file only
        tick_count = getattr(self, "_calib_tick_count", 0) + 1
        self._calib_tick_count = tick_count
        if tick_count % 10 == 1:
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self._calib_log_path, "a", encoding="utf-8") as _f:
                    _f.write(f"[{ts}] [{self.current_user}] buffer={buf_len}/{needed}  lead_off={lead_off_now}\n")
            except Exception:
                pass

        # ── Ham örnekleri CSV'ye yaz (yalnızca yeni gelenler) ──
        if preview is not None and getattr(self, "_calib_raw_csv", None) is not None:
            try:
                total    = len(preview)
                already  = getattr(self, "_calib_raw_csv_written", 0)
                new_rows = preview.iloc[already:]
                if len(new_rows) > 0:
                    session_dt = getattr(self, "_calib_session_dt", "")
                    for _, row in new_rows.iterrows():
                        ts_val  = row.get("timestamp", "")
                        ecg_val = row.get("ecg", "")
                        ppg_val = row.get("ppg", "")
                        self._calib_raw_csv.write(
                            f"{self.current_user},{session_dt},{ts_val},{ecg_val},{ppg_val}\n"
                        )
                    self._calib_raw_csv.flush()
                    self._calib_raw_csv_written = total
            except Exception as csv_e:
                self._log_calib(f"⚠️  CSV yazma hatası: {csv_e}")

        if preview is not None and len(preview) >= 50:
            self._update_calib_graphs(preview["ecg"].values, preview["ppg"].values)

        # ── Timeout: 3 minutes ──
        elapsed = tick_count * 0.5  # seconds (500 ms per tick)
        if elapsed > 180:
            self._log_calib(f"⏰  BLE calibration timed out after {int(elapsed)}s — buffer had {buf_len} samples, needed {needed}")
            self._stop_calibration()
            messagebox.showwarning(
                "BLE Timeout",
                f"No full window received after {int(elapsed)//60} min.\n"
                f"Buffer: {buf_len}/{needed} samples.\n"
                f"Check electrode contact and try again."
            )
            return

        # ── Got a full window ──
        seg = self._calib_ble_streamer.get_window()
        if seg is not None:
            self._calib_tick_count = 0
            self._calib_prev_lead_off = None
            self._calib_ble_streamer.stop()
            self._calib_ble_streamer = None
            self._calib_after_id = None
            try:
                if self._calib_start_btn.winfo_exists():
                    self._calib_start_btn.config(state="normal")
                    self._calib_stop_btn.config(state="disabled")
            except Exception:
                pass
            self._set_calib_status(self.t("completed"), fg=C["success"])
            self._log_calib(f"✔  Received {len(seg)} samples — processing...")
            if self._calib_pending is None:
                self._log_calib("⚠️  _calib_pending is None — cannot save record")
                return
            pending = self._calib_pending
            self._calib_pending = None
            self._log_calib(f"📋  Pending: SBP={pending['sbp']} DBP={pending['dbp']} label={pending['label']}")
            self._update_calib_graphs(seg["ecg"].values, seg["ppg"].values)
            res: PATResult = calculate_pat_both(seg["ecg"].values, seg["ppg"].values)
            self._log_calib(
                f"🔬  PAT raw: foot={res.foot}  peak={res.peak}  "
                f"sqi.ok={res.sqi.ok}  sqi.n_peaks={res.sqi.n_peaks}  "
                f"sqi.reason={res.sqi.reason}"
            )
            # If PAT is 0/None but peaks were found, force sqi.ok so the
            # record is still saved with whatever PAT values came back.
            # This lets calibration proceed even with poor signal quality.
            if not res.sqi.ok and res.sqi.n_peaks >= 3:
                self._log_calib(
                    f"⚠️  SQI failed but {res.sqi.n_peaks} peaks found — "
                    f"saving record anyway (PAT may be unreliable)"
                )
                # Patch sqi so _finish_calibration doesn't bail out
                res.sqi.ok = True
                res.sqi.reason = f"Forced (peaks={res.sqi.n_peaks}, foot={res.foot}, peak={res.peak})"
            self._finish_calibration(pending["model"], res,
                                     pending["sbp"], pending["dbp"], pending["label"],
                                     file="BLE", raw_df=seg)
            return

        self._calib_after_id = self.root.after(500, self._calib_tick_ble)

    def _delete_cal(self, model: CalibModel):
        idx = self._selected_idx()
        if idx is None: return
        c = model.calibrations[idx]
        if messagebox.askyesno(self.t("confirm"),
                               self.t("delete_record", idx+1,
                                      c.get('label',''), int(c['sbp']), int(c['dbp']))):
            model.remove_calibration(idx)
            self.db.save()
            self._refresh_tree(model)
            self._log_calib_records_to_file(model)
            self._log_calib(self.t("record_deleted", idx+1))
 
    def _edit_cal(self, model: CalibModel):
        idx = self._selected_idx()
        if idx is None:
            return
        c = model.calibrations[idx]

        win = tk.Toplevel(self.root)
        win.title(self.t("edit_record", idx + 1))
        win.geometry("340x240")
        win.configure(bg=C["bg"])
        win.grab_set()
        win.focus_force()

        # Header
        hdr = tk.Frame(win, bg=C["warn_lt"], height=50)
        hdr.pack(fill="x")
        tk.Label(hdr, text=self.t("edit_record", idx + 1),
                 font=FH3, fg="white", bg=C["warn_lt"]).place(x=20, y=12)

        body = tk.Frame(win, bg=C["bg"], padx=24, pady=20)
        body.pack(fill="both", expand=True)

        lbl(body, self.t("record_lbl", idx + 1, c.get('label', '')), font=FB_B).pack(anchor="w", pady=(0, 2))
        lbl(body, self.t("file_lbl", os.path.basename(c.get('file', ''))),
            font=FSM, fg=C["text_light"]).pack(anchor="w", pady=(0, 16))

        # Input fields
        rf = tk.Frame(body, bg=C["bg"])
        rf.pack(fill="x")

        lbl(rf, self.t("sbp_lbl"), font=FSM_B, fg=C["text_sub"]).grid(row=0, column=0, padx=(0, 8), sticky="w")
        se = entry(rf, width=8)
        se.insert(0, str(int(c["sbp"])))
        se.grid(row=0, column=1, padx=(0, 16), sticky="w")

        lbl(rf, self.t("dbp_lbl"), font=FSM_B, fg=C["text_sub"]).grid(row=0, column=2, padx=(0, 8), sticky="w")
        de = entry(rf, width=8)
        de.insert(0, str(int(c["dbp"])))
        de.grid(row=0, column=3)

        def confirm():
            try:
                sbp, dbp = float(se.get()), float(de.get())
                assert 60 < sbp < 220 and 40 < dbp < 140
            except:
                messagebox.showwarning(self.t("error"), self.t("valid_bp"), parent=win)
                return
            model.update_calibration(idx, sbp, dbp)
            self.db.save()
            self._refresh_tree(model)
            self._log_calib_records_to_file(model)
            self._log_calib(self.t("record_updated", idx + 1, int(sbp), int(dbp)))
            win.destroy()

        win.bind("<Return>", lambda e: confirm())
        
        # Buttons
        btn_frame = tk.Frame(body, bg=C["bg"])
        btn_frame.pack(pady=(20, 0), fill="x")
        btn(btn_frame, self.t("save"), confirm, C["success"], width=32, size="small").pack(fill="x", pady=(0, 8))
        btn(btn_frame, self.t("cancel"), win.destroy, C["text_light"], width=32, size="small").pack(fill="x")
 
    def _add_calibration(self, model: CalibModel):
        try:
            sbp = float(self._sbp_e.get())
            dbp = float(self._dbp_e.get())
            assert 60 < sbp < 220 and 40 < dbp < 140
        except:
            messagebox.showwarning(self.t("error"), self.t("valid_sbp_dbp")); return
 
        label = self._lbl_e.get().strip() or "—"
        source = self._calib_source_var.get()
 
        if source == "csv":
            path = filedialog.askopenfilename(title=self.t("calib_csv_title"),
                                               filetypes=[("CSV","*.csv")])
            if not path: return
            try:
                df = pd.read_csv(path)
                assert {"ecg","ppg"}.issubset(df.columns)
            except:
                messagebox.showerror(self.t("error"), self.t("csv_ecg_ppg")); return
 
            self._log_calib(f"\n📂  {os.path.basename(path)}")
            res: PATResult = calculate_pat_both(df["ecg"].values, df["ppg"].values)
            self._log_calib(
                f"🔬  PAT raw: foot={res.foot}  peak={res.peak}  "
                f"sqi.ok={res.sqi.ok}  sqi.n_peaks={res.sqi.n_peaks}  "
                f"sqi.reason={res.sqi.reason}"
            )
            if not res.sqi.ok and res.sqi.n_peaks >= 3:
                self._log_calib(
                    f"⚠️  SQI failed but {res.sqi.n_peaks} peaks found — "
                    f"saving record anyway (PAT may be unreliable)"
                )
                res.sqi.ok = True
                res.sqi.reason = f"Forced (peaks={res.sqi.n_peaks}, foot={res.foot}, peak={res.peak})"

            # Plot ECG/PPG signals in calibration mode
            self._update_calib_graphs(df["ecg"].values, df["ppg"].values)

            self._finish_calibration(model, res, sbp, dbp, label, file=path, raw_df=df)
 
        else:
            self._start_calib_ble(model, sbp, dbp, label)
    def _finish_calibration(self, model: CalibModel, res: PATResult,
                             sbp: float, dbp: float, label: str,
                             file: str = "", raw_df=None):
        """Common finish logic for both CSV and BLE calibration paths."""
        try:
            self._log_calib(f"🔍  SQI check: ok={res.sqi.ok}  reason={res.sqi.reason}  peaks={res.sqi.n_peaks}")
            if not res.sqi.ok:
                self._log_calib(self.t("sqi_failed", res.sqi.reason))
                messagebox.showwarning(self.t("signal_quality"), res.sqi.reason)
                return

            pf_s = f"{res.foot:.1f} ms" if res.foot else "—"
            pp_s = f"{res.peak:.1f} ms" if res.peak else "—"
            self._log_calib(f"✔  Foot={pf_s}   Peak={pp_s}   Peaks={res.sqi.n_peaks}")

            # ── Save raw signal CSV ──
            if raw_df is not None:
                try:
                    safe_user  = self.current_user.replace(" ", "_")
                    safe_label = label.replace(" ", "_").replace("/", "-")
                    ts_str     = time.strftime("%Y%m%d_%H%M%S")
                    csv_name   = f"{safe_user}_{safe_label}_{ts_str}.csv"
                    os.makedirs("raw_records", exist_ok=True)
                    out_df = raw_df.copy()
                    out_df.insert(0, "person_name", self.current_user)
                    out_df.insert(1, "session_datetime", time.strftime("%Y-%m-%d %H:%M:%S"))
                    out_df.insert(2, "label", label)
                    out_df.insert(3, "sbp", sbp)
                    out_df.insert(4, "dbp", dbp)
                    out_df.to_csv(os.path.join("raw_records", csv_name), index=False)
                    self._log_calib(f"💾  Raw CSV: raw_records/{csv_name}")
                except Exception as csv_err:
                    self._log_calib(f"⚠️  Raw CSV kaydedilemedi: {csv_err}")

            result = model.add_calibration(res.foot, res.peak, sbp, dbp,
                                           label=label, file=file)
            self._log_calib(f"📥  add_calibration returned — total records: {len(model.calibrations)}")
            self.db.save()
            self._log_calib(f"💾  DB saved")
            self._refresh_tree(model)
            self._log_calib_records_to_file(model)
            self._log_calib(self.t("record_saved", int(sbp), int(dbp), label))

            for m_name, r in [("Foot", result["foot"]), ("Peak", result["peak"])]:
                if r["updated"]:
                    self._log_calib(self.t("model_updated", m_name, r['r2']))
                else:
                    self._log_calib(self.t("model_pending", m_name, r['reason']))
        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            self._log_calib(f"❌  _finish_calibration ERROR: {exc}")
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self._calib_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] [{self.current_user}] TRACEBACK:\n{err}\n")
            except Exception:
                pass
 
    def _log_calib(self, msg: str):
        # Update on-screen log widget
        try:
            self._calib_log.config(state="normal")
            self._calib_log.insert("end", msg+"\n")
            self._calib_log.see("end")
            self._calib_log.config(state="disabled")
        except tk.TclError:
            pass
        # Write to file log
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self._calib_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{self.current_user}] {msg}\n")
        except Exception:
            pass

    def _update_calib_graphs(self, ecg: np.ndarray, ppg: np.ndarray):
        """Update ECG/PPG graphs in calibration mode"""
        if not hasattr(self, '_calib_ax1'):
            return
        
        time_axis = np.arange(len(ecg)) / FS
        
        # Clear previous plots
        self._calib_ax1.clear()
        self._calib_ax2.clear()
        
        # Plot ECG
        self._calib_ax1.plot(time_axis, ecg, color=C["primary"], linewidth=1.5)
        self._calib_ax1.set_title("ECG Signal", fontsize=10, fontweight='bold', color=C["text"])
        self._calib_ax1.set_xlabel("Time (s)", fontsize=9, color=C["text_light"])
        self._calib_ax1.set_ylabel("Amplitude", fontsize=9, color=C["text_light"])
        self._calib_ax1.tick_params(colors=C["text_light"], labelsize=8)
        self._calib_ax1.grid(True, alpha=0.2)
        self._calib_ax1.set_facecolor("white")
        for spine in self._calib_ax1.spines.values():
            spine.set_color(C["border"])
        
        # Plot PPG
        self._calib_ax2.plot(time_axis, ppg, color=C["peak"], linewidth=1.5)
        self._calib_ax2.set_title("PPG Signal", fontsize=10, fontweight='bold', color=C["text"])
        self._calib_ax2.set_xlabel("Time (s)", fontsize=9, color=C["text_light"])
        self._calib_ax2.set_ylabel("Amplitude", fontsize=9, color=C["text_light"])
        self._calib_ax2.tick_params(colors=C["text_light"], labelsize=8)
        self._calib_ax2.grid(True, alpha=0.2)
        self._calib_ax2.set_facecolor("white")
        for spine in self._calib_ax2.spines.values():
            spine.set_color(C["border"])
        
        self._calib_fig.tight_layout(pad=2)
        self._calib_canvas.draw()

    def _update_meas_preview(self, ecg: np.ndarray, ppg: np.ndarray):
        """Plot a partial BLE preview in measurement mode while the first full window accumulates."""
        if not hasattr(self, '_meas_ax1'):
            return

        # Downsample preview if needed for responsiveness
        max_points = 1200
        step = max(1, len(ecg) // max_points)
        ecg_ds = ecg[::step]
        ppg_ds = ppg[::step]
        time_axis = np.arange(len(ecg_ds)) * step / FS

        self._meas_ax1.clear()
        self._meas_ax1.plot(time_axis, ecg_ds, color=C["primary"], linewidth=1)
        self._meas_ax1.set_title("ECG Signal", fontsize=9, fontweight='bold', color=C["text"])
        self._meas_ax1.set_xlabel("Time (s)", fontsize=8, color=C["text_light"])
        self._meas_ax1.set_ylabel("Amplitude", fontsize=8, color=C["text_light"])
        self._meas_ax1.tick_params(colors=C["text_light"], labelsize=7)
        self._meas_ax1.grid(True, alpha=0.2)
        self._meas_ax1.set_facecolor("white")
        for spine in self._meas_ax1.spines.values():
            spine.set_color(C["border"])

        self._meas_ax2.clear()
        self._meas_ax2.plot(time_axis, ppg_ds, color=C["peak"], linewidth=1)
        self._meas_ax2.set_title("PPG Signal", fontsize=9, fontweight='bold', color=C["text"])
        self._meas_ax2.set_xlabel("Time (s)", fontsize=8, color=C["text_light"])
        self._meas_ax2.set_ylabel("Amplitude", fontsize=8, color=C["text_light"])
        self._meas_ax2.tick_params(colors=C["text_light"], labelsize=7)
        self._meas_ax2.grid(True, alpha=0.2)
        self._meas_ax2.set_facecolor("white")
        for spine in self._meas_ax2.spines.values():
            spine.set_color(C["border"])

        self._meas_fig.tight_layout(pad=1.5)
        self._meas_canvas.draw()

    # ════════════════════════════════
    # MEASUREMENT MODE
    # ════════════════════════════════
    def _show_measurement(self):
        self._current_screen = self._show_measurement
        self._clear()
        model: CalibModel = self.db.users[self.current_user]["model"]
        self._build_top_bar(self.t("measurement_mode"), C["success_lt"])

        body_outer, body = scrollable_frame(self.root, bg=C["bg"])
        body_outer.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Configuration Row ──
        config_row = tk.Frame(body, bg=C["bg"])
        config_row.pack(fill="x", pady=(0, 12))

        # Algorithm Card
        af = card(config_row, padx=14, pady=10)
        af.pack(side="left", fill="both", expand=True, padx=(0, 10))
        lbl(af, self.t("algorithm"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 8))
        self._method_var = tk.StringVar(value="foot")
        tk.Radiobutton(af, text=self.t("foot_rb"),
                       variable=self._method_var, value="foot",
                       font=FB, fg=C["foot"],
                       bg=C["card"], activebackground=C["card"], selectcolor=C["card"]).pack(anchor="w", padx=4, pady=2)
        tk.Radiobutton(af, text=self.t("peak_rb"),
                       variable=self._method_var, value="peak",
                       font=FB, fg=C["peak"],
                       bg=C["card"], activebackground=C["card"], selectcolor=C["card"]).pack(anchor="w", padx=4, pady=2)

        # Data Source Card
        sf = card(config_row, padx=14, pady=10)
        sf.pack(side="right", fill="both", expand=True)
        lbl(sf, self.t("data_source"), font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 8))
        self._source_var = tk.StringVar(value="csv")
        tk.Radiobutton(sf, text=self.t("csv_source"),
                       variable=self._source_var, value="csv",
                       font=FB, bg=C["card"], activebackground=C["card"],
                       selectcolor=C["card"]).pack(anchor="w", padx=4, pady=2)
        tk.Radiobutton(sf, text=self.t("ble_source"),
                       variable=self._source_var, value="ble",
                       font=FB, bg=C["card"], activebackground=C["card"],
                       selectcolor=C["card"]).pack(anchor="w", padx=4, pady=2)

        # ── Warnings ──
        warns = []
        if model.foot_using_defaults:
            warns.append("Foot")
        if model.peak_using_defaults:
            warns.append("Peak")
        if warns:
            wf = card(body, padx=12, pady=8)
            wf.pack(fill="x", pady=(0, 10))
            lbl(wf, self.t("default_warn", "/".join(warns)), font=FSM, fg=C["warn_lt"]).pack(anchor="w")

        # ── Graphs (ECG/PPG/BP) ──
        graph_frame = card(body, padx=10, pady=10)
        graph_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        lbl(graph_frame, "Real-time Signals", font=FH3, fg=C["primary"]).pack(anchor="w", pady=(0, 8))
        self._meas_fig, self._meas_ax1, self._meas_ax2, self._meas_ax3, self._meas_canvas = create_triple_graph(graph_frame, "ECG", "PPG", "BP")
        
        # Initialize BP history for plotting
        self._sbp_history = []
        self._dbp_history = []
        self._sample_count = 0

        # ── BLE Status ──
        ble_row = tk.Frame(body, bg=C["bg"])
        ble_row.pack(fill="x", pady=(0, 8))
        self._ble_status_lbl = lbl(ble_row, self.t("ble_not_conn"), fg=C["text_light"], font=FSM)
        self._ble_status_lbl.pack(side="left")
        self._lead_off_lbl = lbl(ble_row, "", fg=C["danger"], font=FSM)
        self._lead_off_lbl.pack(side="left", padx=20)

        # ── BP Display Card ──
        dc = card(body, padx=18, pady=14)
        dc.pack(fill="x", pady=(0, 10))
        self._bp_label = tk.Label(dc, text="— / —  mmHg",
                                  font=("Segoe UI", 42, "bold"),
                                  fg=C["primary"], bg=C["card"])
        self._bp_label.pack(pady=(0, 3))
        self._pat_label = lbl(dc, "PAT: —", font=FB, fg=C["text_light"])
        self._pat_label.pack()
        self._status_lbl = lbl(dc, self.t("waiting"), font=FSM, fg=C["text_light"])
        self._status_lbl.pack(pady=(3, 0))

        # ── Progress Bar ──
        self._progress = ttk.Progressbar(body, length=600, mode="determinate")
        self._progress.pack(pady=(0, 8), fill="x")

        # ── Log Card ──
        lc = card(body, padx=10, pady=8)
        lc.pack(fill="both", expand=True, pady=(0, 8))
        self._meas_log = tk.Text(lc, height=3, font=FSM,
                                 bg="white", relief="solid", bd=1, state="disabled",
                                 fg=C["text"])
        sb2 = tk.Scrollbar(lc, command=self._meas_log.yview)
        self._meas_log.config(yscrollcommand=sb2.set)
        sb2.pack(side="right", fill="y")
        self._meas_log.pack(fill="both", expand=True)

        # ── Control Buttons ──
        bf = tk.Frame(body, bg=C["bg"])
        bf.pack(fill="x")
        self._start_btn = btn(bf, self.t("start"), self._start_measurement,
                              C["success"], width=18, size="small")
        self._start_btn.pack(side="left", padx=(0, 8), fill="x", expand=True)
        btn(bf, self.t("stop"), self._stop_measurement, C["danger"], width=12, size="small").pack(side="left", padx=(0, 8), fill="x", expand=False)
        btn(bf, self.t("back_dashboard"), self._show_dashboard, C["text_light"], width=18, size="small").pack(side="left", fill="x", expand=True)
 
    def _start_measurement(self):
        source = self._source_var.get()
        if source == "csv":
            path = filedialog.askopenfilename(title=self.t("meas_csv_title"),
                                              filetypes=[("CSV","*.csv")])
            if not path: return
            try:
                df = pd.read_csv(path)
                assert {"ecg","ppg"}.issubset(df.columns)
            except:
                messagebox.showerror(self.t("error"), self.t("csv_ecg_ppg")); return
 
            self._stop_streams()
            self.csv_streamer = CSVStreamer(df)
            self._total_w = max(1, (len(df) - WIN_SEC*FS) // (STEP_SEC*FS) + 1)
            self._win_n   = 0
            self._progress["value"] = 0
            self._log_meas(self.t("csv_log", os.path.basename(path), len(df), f"{len(df)/FS:.0f}"))
            self._start_btn.config(state="disabled")
            self._tick_csv()
        else:
            self._stop_streams()
            self.ble_streamer = BLEStreamer()
            self.ble_streamer.start()
            self._log_meas(self.t("ble_starting"))
            self._progress.config(mode="indeterminate")
            self._progress.start(50)
            self._start_btn.config(state="disabled")
            self._tick_ble()
 
    def _stop_measurement(self):
        if self.after_id:
            self.root.after_cancel(self.after_id); self.after_id = None
        self._stop_streams()
        try:
            self._progress.stop()
            self._progress.config(mode="determinate")
            self._status_lbl.config(text=self.t("stopped"), fg=C["danger"])
            self._start_btn.config(state="normal")
        except tk.TclError:
            pass
 
    def _tick_csv(self):
        if self.csv_streamer is None or self.csv_streamer.done:
            self._status_lbl.config(text=self.t("completed"), fg=C["success"])
            self._start_btn.config(state="normal"); return
        seg = self.csv_streamer.next_window()
        if seg is None:
            self._status_lbl.config(text=self.t("completed"), fg=C["success"])
            self._start_btn.config(state="normal"); return
        self._win_n += 1
        self._progress["value"] = min(100, int(100 * self._win_n / self._total_w))
        self._process_segment(seg["ecg"].values, seg["ppg"].values, self._win_n, source="csv")
        self.after_id = self.root.after(1000, self._tick_csv)
 
    def _tick_ble(self):
        if self.ble_streamer is None: return
        while True:
            msg = self.ble_streamer.get_status_message()
            if msg is None: break
            self._log_meas(f"  [{msg}]")
            color = C["ble"] if "✔" in msg else (C["danger"] if "Error" in msg or "lost" in msg else C["text_light"])
            self._ble_status_lbl.config(text=f"● BLE: {msg}", fg=color)
            # Show notification on successful connection
            if "✔" in msg and "successful" in msg:
                messagebox.showinfo("BLE Connection", "Device connected successfully!")

        if self.ble_streamer.lead_off:
            self._lead_off_lbl.config(text=self.t("lead_off"))
        else:
            self._lead_off_lbl.config(text="")

        seg = self.ble_streamer.get_window()
        if seg is not None:
            self._win_n = getattr(self, "_win_n", 0) + 1
            self._process_segment(seg["ecg"].values, seg["ppg"].values, self._win_n, source="ble")
        else:
            preview = self.ble_streamer.peek_buffer(max_samples=2000)
            if preview is not None and len(preview) >= 50:
                self._update_meas_preview(preview["ecg"].values, preview["ppg"].values)
        self.after_id = self.root.after(500, self._tick_ble)
 
    def _process_segment(self, ecg, ppg, win_n: int, source: str = "csv"):
        res: PATResult = calculate_pat_both(ecg, ppg)
        method = self._method_var.get()
        model: CalibModel = self.db.users[self.current_user]["model"]
        mcolor = C["foot"] if method == "foot" else C["peak"]
        pat = res.foot if method == "foot" else res.peak

        # Update graphs
        time_axis = np.arange(len(ecg)) / FS
        
        self._meas_ax1.clear()
        self._meas_ax1.plot(time_axis, ecg, color=C["primary"], linewidth=1.5)
        self._meas_ax1.set_title("ECG Signal", fontsize=9, fontweight='bold', color=C["text"])
        self._meas_ax1.set_xlabel("Time (s)", fontsize=8, color=C["text_light"])
        self._meas_ax1.set_ylabel("Amplitude", fontsize=8, color=C["text_light"])
        self._meas_ax1.tick_params(colors=C["text_light"], labelsize=7)
        self._meas_ax1.grid(True, alpha=0.2)
        self._meas_ax1.set_facecolor("white")
        for spine in self._meas_ax1.spines.values():
            spine.set_color(C["border"])
        
        self._meas_ax2.clear()
        self._meas_ax2.plot(time_axis, ppg, color=C["peak"], linewidth=1.5)
        self._meas_ax2.set_title("PPG Signal", fontsize=9, fontweight='bold', color=C["text"])
        self._meas_ax2.set_xlabel("Time (s)", fontsize=8, color=C["text_light"])
        self._meas_ax2.set_ylabel("Amplitude", fontsize=8, color=C["text_light"])
        self._meas_ax2.tick_params(colors=C["text_light"], labelsize=7)
        self._meas_ax2.grid(True, alpha=0.2)
        self._meas_ax2.set_facecolor("white")
        for spine in self._meas_ax2.spines.values():
            spine.set_color(C["border"])

        DEFAULT_PAT = 250.0  # ms — midpoint of default model range

        if pat is None and model.foot_using_defaults and model.peak_using_defaults:
            # No personal calibration and PAT could not be calculated —
            # fall back to default PAT so default coefficients still produce a reading.
            pat = DEFAULT_PAT
            res.sqi.reason = f"PAT unavailable — using default PAT={DEFAULT_PAT}ms"

        if pat is None:
            self._bp_label.config(text="— / —  mmHg", fg=C["danger"])
            self._pat_label.config(text="PAT: —")
            reason = res.sqi.reason if not res.sqi.ok else self.t("pat_none", method)
            self._status_lbl.config(text=f"⚠️  {reason}", fg=C["warn"])
            self._log_meas(f"  W{win_n:03d}  ✗  {reason}")
        else:
            sbp, dbp = model.predict(pat, method)
            bp_clr = C["danger"] if sbp >= 140 else (C["warn"] if sbp >= 120 else C["success"])
            using_default_pat = (res.foot is None and res.peak is None)
            pat_label_str = f"⚙️ Default PAT={pat:.0f}ms" if using_default_pat else f"PAT={pat:.1f}ms"
            f_str = (f"Foot={res.foot:.0f}ms→{model.predict(res.foot,'foot')[0]:.0f}"
                     if res.foot else f"Foot=default({pat:.0f}ms)")
            p_str = (f"Peak={res.peak:.0f}ms→{model.predict(res.peak,'peak')[0]:.0f}"
                     if res.peak else f"Peak=default({pat:.0f}ms)")
            self._bp_label.config(text=f"{sbp} / {dbp}  mmHg", fg=bp_clr)
            self._pat_label.config(
                text=f"{'🟣 Foot' if method=='foot' else '🔵 Peak'}  {pat_label_str}  |  {res.sqi.n_peaks} beats",
                fg=C["warn"] if using_default_pat else mcolor)
            status_suffix = "  (default PAT — calibrate for accuracy)" if using_default_pat else ""
            self._status_lbl.config(text=self.t("meas_active") + status_suffix, fg=C["success"] if not using_default_pat else C["warn"])
            self._log_meas(f"  W{win_n:03d}  {f_str} mmHg   {p_str} mmHg")
            
            # Add to BP history and update BP plot
            self._sbp_history.append(sbp)
            self._dbp_history.append(dbp)
            self._sample_count += 1
            
            self._meas_ax3.clear()
            if len(self._sbp_history) > 0:
                self._meas_ax3.plot(range(len(self._sbp_history)), self._sbp_history,
                                   color=C["success"], linewidth=2, marker='o', markersize=4, label='SBP')
                self._meas_ax3.plot(range(len(self._dbp_history)), self._dbp_history,
                                   color=C["danger"], linewidth=2, marker='o', markersize=4, label='DBP')
                self._meas_ax3.axhline(y=120, color=C["warn"], linestyle='--', alpha=0.5, linewidth=1)
                self._meas_ax3.axhline(y=140, color=C["danger"], linestyle='--', alpha=0.5, linewidth=1)
                self._meas_ax3.legend(fontsize=7, loc='upper right')
            else:
                self._meas_ax3.text(0.5, 0.5, "Waiting for valid BP data...",
                                    transform=self._meas_ax3.transAxes, ha='center', va='center',
                                    fontsize=10, color=C["text_light"])
            self._meas_ax3.set_title("BP Trend", fontsize=9, fontweight='bold', color=C["text"])
            self._meas_ax3.set_ylabel("BP (mmHg)", fontsize=8, color=C["text_light"])
            self._meas_ax3.set_xlabel("Measurement #", fontsize=8, color=C["text_light"])
            self._meas_ax3.tick_params(colors=C["text_light"], labelsize=7)
            self._meas_ax3.grid(True, alpha=0.2)
            self._meas_ax3.set_facecolor("white")
            for spine in self._meas_ax3.spines.values():
                spine.set_color(C["border"])
        
        self._meas_fig.tight_layout(pad=1.5)
        self._meas_canvas.draw()
 
    def _log_meas(self, msg: str):
        self._meas_log.config(state="normal")
        self._meas_log.insert("end", msg+"\n")
        self._meas_log.see("end")
        self._meas_log.config(state="disabled")
 
    # ════════════════════════════════════════════
    # ERROR CALCULATION MODE  (MAE / RMSE)
    # ════════════════════════════════════════════
    def _show_error_calc(self):
        self._current_screen = self._show_error_calc
        self._clear()
        self._build_top_bar(self.t("error_calc_mode"), C["danger"])

        # ── Persistent state — survives screen navigation ──
        if not hasattr(self, "_ec_state"):
            self._ec_state = {"method": "foot", "selected": set(),
                              "verif_path": None, "verif_ref_sbp": None, "verif_ref_dbp": None}
        st = self._ec_state
        model: CalibModel = self.db.users[self.current_user]["model"]

        self._ec_method_var = tk.StringVar(value=st["method"])
        self._ec_method_var.trace_add("write",
            lambda *_: st.update(method=self._ec_method_var.get()))

        body_outer, body = scrollable_frame(self.root, bg=C["bg"])
        body_outer.pack(fill="both", expand=True, padx=20, pady=16)

        # ── ① Algoritma ──
        step1 = card(body, padx=16, pady=14); step1.pack(fill="x", pady=(0, 12))
        h1 = tk.Frame(step1, bg=C["card"]); h1.pack(fill="x", pady=(0, 10))
        tk.Frame(h1, bg=C["primary"], width=4, height=26).pack(side="left", padx=(0, 10))
        lbl_col = tk.Frame(h1, bg=C["card"]); lbl_col.pack(side="left", fill="x", expand=True)
        lbl(lbl_col, "① " + self.t("method_lbl").rstrip(":"), font=FH3, fg=C["primary"]).pack(anchor="w")
        lbl(lbl_col, "PAT detection yöntemini seçin", font=FSM, fg=C["text_light"]).pack(anchor="w")
        mrow = tk.Frame(step1, bg=C["card"]); mrow.pack(anchor="w", padx=6)
        tk.Radiobutton(mrow, text="🟣 Foot (onset)", variable=self._ec_method_var,
                       value="foot", font=FB, fg=C["foot"], bg=C["card"],
                       activebackground=C["card"], selectcolor=C["card"]).pack(side="left", padx=(0, 24))
        tk.Radiobutton(mrow, text="🔵 Peak (systolic)", variable=self._ec_method_var,
                       value="peak", font=FB, fg=C["peak"], bg=C["card"],
                       activebackground=C["card"], selectcolor=C["card"]).pack(side="left")

        # ── ② Kalibrasyon Kayıtları ──
        step2 = card(body, padx=16, pady=14); step2.pack(fill="x", pady=(0, 12))
        h2 = tk.Frame(step2, bg=C["card"]); h2.pack(fill="x", pady=(0, 10))
        tk.Frame(h2, bg=C["warn"], width=4, height=26).pack(side="left", padx=(0, 10))
        lbl_col2 = tk.Frame(h2, bg=C["card"]); lbl_col2.pack(side="left", fill="x", expand=True)
        lbl(lbl_col2, "② Kalibrasyon Kayıtları", font=FH3, fg=C["primary"]).pack(anchor="w")
        lbl(lbl_col2, "Model fit için kullanılacak kayıtları işaretleyin (min. 2)",
            font=FSM, fg=C["text_light"]).pack(anchor="w")

        calib_list = model.calibrations
        self._ec_calib_list = calib_list  # keep reference for toggle handler

        if not calib_list:
            empty_f = tk.Frame(step2, bg="#FEF3C7",
                               highlightbackground=C["warn"], highlightthickness=1)
            empty_f.pack(fill="x", pady=(8, 0))
            lbl(empty_f, "⚠️  Henüz kalibrasyon kaydı yok — önce Kalibrasyon Modu'ndan kayıt ekleyin.",
                font=FSM_B, fg=C["warn"], bg="#FEF3C7").pack(anchor="w", padx=10, pady=10)
            self._ec_tree = None
        else:
            # ── Proper Treeview table (auto-sized columns, no clipping) ──
            tree_frame = tk.Frame(step2, bg=C["card"])
            tree_frame.pack(fill="both", expand=False, pady=(8, 4))

            cols = ("sel", "no", "label", "file", "pat_foot", "pat_peak", "sbp", "dbp")
            heads = {"sel": "✓", "no": "#", "label": "Etiket", "file": "Dosya",
                     "pat_foot": "PAT-F", "pat_peak": "PAT-P",
                     "sbp": "SBP", "dbp": "DBP"}
            widths = {"sel": 36, "no": 36, "label": 130, "file": 220,
                      "pat_foot": 72, "pat_peak": 72, "sbp": 56, "dbp": 56}
            anchors = {"sel": "center", "no": "center", "label": "w", "file": "w",
                       "pat_foot": "center", "pat_peak": "center",
                       "sbp": "center", "dbp": "center"}

            n_rows = min(len(calib_list), 8)  # show up to 8 rows without scrolling
            self._ec_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                          height=max(3, n_rows), selectmode="none")
            for col in cols:
                self._ec_tree.heading(col, text=heads[col])
                self._ec_tree.column(col, width=widths[col], anchor=anchors[col],
                                     stretch=(col in ("label", "file")))
            self._ec_tree.tag_configure("selected", background="#DCFCE7",
                                         foreground=C["text"])
            self._ec_tree.tag_configure("unselected", background=C["card"],
                                         foreground=C["text_sub"])

            sb_tree = tk.Scrollbar(tree_frame, command=self._ec_tree.yview)
            self._ec_tree.config(yscrollcommand=sb_tree.set)
            sb_tree.pack(side="right", fill="y")
            self._ec_tree.pack(fill="both", expand=True)

            # Populate the rows
            def _refresh_tree():
                self._ec_tree.delete(*self._ec_tree.get_children())
                for i, c in enumerate(calib_list):
                    pf = f"{c['pat_foot']:.0f}" if c.get("pat_foot") else "—"
                    pp = f"{c['pat_peak']:.0f}" if c.get("pat_peak") else "—"
                    is_sel = i in st["selected"]
                    self._ec_tree.insert("", "end", iid=str(i),
                        values=("✓" if is_sel else "",
                                i + 1,
                                c.get("label", "—") or "—",
                                os.path.basename(c.get("file", "")) or "—",
                                pf, pp,
                                f"{c['sbp']:.0f}",
                                f"{c['dbp']:.0f}"),
                        tags=("selected" if is_sel else "unselected",))
            _refresh_tree()
            self._ec_refresh_tree = _refresh_tree

            # Click anywhere on a row to toggle selection
            def _on_tree_click(event):
                iid = self._ec_tree.identify_row(event.y)
                if not iid:
                    return
                i = int(iid)
                if i in st["selected"]:
                    st["selected"].discard(i)
                else:
                    st["selected"].add(i)
                _refresh_tree()
                self._ec_update_sel_count()
                self._ec_update_ready()
            self._ec_tree.bind("<Button-1>", _on_tree_click)

            # Hint
            lbl(step2, "💡 Bir satıra tıklayarak seçimi açıp kapatabilirsiniz",
                font=FSM, fg=C["text_light"]).pack(anchor="w", pady=(2, 0))

        # ── Select All / Clear row + status badge ──
        sa_row = tk.Frame(step2, bg=C["card"]); sa_row.pack(fill="x", pady=(8, 0))
        def _sel_all():
            for i in range(len(calib_list)):
                st["selected"].add(i)
            if self._ec_tree is not None:
                self._ec_refresh_tree()
            self._ec_update_sel_count()
            self._ec_update_ready()
        def _sel_none():
            st["selected"].clear()
            if self._ec_tree is not None:
                self._ec_refresh_tree()
            self._ec_update_sel_count()
            self._ec_update_ready()
        btn(sa_row, "☑  Tümünü Seç", _sel_all,
            C["primary_lt"], width=14, size="small").pack(side="left", padx=(0, 8))
        btn(sa_row, "☐  Temizle", _sel_none,
            C["text_light"], width=12, size="small").pack(side="left")
        self._ec_sel_lbl = lbl(sa_row, "", font=FSM_B, fg=C["text_light"])
        self._ec_sel_lbl.pack(side="right", padx=(0, 4))
        self._ec_update_sel_count()

        # ── ③ Doğrulama Dosyası ──
        step3 = card(body, padx=16, pady=14); step3.pack(fill="x", pady=(0, 12))
        h3 = tk.Frame(step3, bg=C["card"]); h3.pack(fill="x", pady=(0, 10))
        tk.Frame(h3, bg=C["primary_lt"], width=4, height=26).pack(side="left", padx=(0, 10))
        lbl_col3 = tk.Frame(h3, bg=C["card"]); lbl_col3.pack(side="left", fill="x", expand=True)
        lbl(lbl_col3, "③ Doğrulama Dosyası (test ölçümü)", font=FH3, fg=C["primary"]).pack(anchor="w")
        lbl(lbl_col3, "Kalibrasyonda kullanılmayan bir ölçüm CSV'si — ECG & PPG sütunları gerekli",
            font=FSM, fg=C["text_light"]).pack(anchor="w")

        # File picker panel — colored border reflects state (green=loaded, gray=empty)
        verif_loaded = bool(st["verif_path"])
        vf_border = C["success"] if verif_loaded else C["border_lt"]
        vf_bg     = "#ECFDF5" if verif_loaded else "#F8FAFC"
        self._ec_verif_box = tk.Frame(step3, bg=vf_bg, relief="flat", bd=0,
                                       highlightbackground=vf_border, highlightthickness=2)
        self._ec_verif_box.pack(fill="x", pady=(4, 8))

        vf_left = tk.Frame(self._ec_verif_box, bg=vf_bg)
        vf_left.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        if verif_loaded:
            vf_text = f"✔  Yüklendi: {os.path.basename(st['verif_path'])}"
            vf_fg   = C["success"]
        else:
            vf_text = "📂  Dosya seçilmedi — sağdaki Browse düğmesine tıklayın"
            vf_fg   = C["text_light"]
        # wraplength stops very long filenames from overflowing horizontally
        self._ec_verif_lbl = lbl(vf_left, vf_text, font=FB_B, fg=vf_fg, bg=vf_bg,
                                  wraplength=480, justify="left")
        self._ec_verif_lbl.pack(anchor="w", fill="x")
        if verif_loaded:
            try:
                _sz_kb = os.path.getsize(st["verif_path"]) / 1024
                lbl(vf_left, f"   Boyut: {_sz_kb:,.1f} KB",
                    font=FSM, fg=C["text_light"], bg=vf_bg).pack(anchor="w")
            except Exception:
                pass

        btn(self._ec_verif_box, "📂  Browse", self._ec_select_verif,
            C["primary_lt"], width=14, size="small").pack(side="right", padx=10, pady=10)

        ref_row = tk.Frame(step3, bg=C["card"]); ref_row.pack(fill="x", pady=(4, 0))
        lbl(ref_row, "Referans BP (CSV'de ref sütunu yoksa):",
            font=FSM, fg=C["text_light"]).pack(side="left", padx=(0, 10))
        lbl(ref_row, "SBP", font=FSM_B, fg=C["success"]).pack(side="left", padx=(0, 4))
        self._ec_ref_sbp_var = tk.StringVar(
            value=str(int(st["verif_ref_sbp"])) if st["verif_ref_sbp"] else "")
        self._ec_ref_sbp_var.trace_add("write",
            lambda *_: (st.update(verif_ref_sbp=self._ec_ref_sbp_var.get()),
                        self._ec_update_ready()))
        entry(ref_row, width=6, textvariable=self._ec_ref_sbp_var).pack(side="left", padx=(0, 12))
        lbl(ref_row, "DBP", font=FSM_B, fg=C["danger"]).pack(side="left", padx=(0, 4))
        self._ec_ref_dbp_var = tk.StringVar(
            value=str(int(st["verif_ref_dbp"])) if st["verif_ref_dbp"] else "")
        self._ec_ref_dbp_var.trace_add("write",
            lambda *_: (st.update(verif_ref_dbp=self._ec_ref_dbp_var.get()),
                        self._ec_update_ready()))
        entry(ref_row, width=6, textvariable=self._ec_ref_dbp_var).pack(side="left")
        lbl(ref_row, "(CSV'de ref_sbp/ref_dbp sütunu varsa otomatik kullanılır)",
            font=FSM, fg=C["text_light"]).pack(side="left", padx=(12, 0))

        # ── ④ Hesapla — own section with header + live readiness indicator ──
        step4 = card(body, padx=16, pady=14); step4.pack(fill="x", pady=(0, 12))
        h4h = tk.Frame(step4, bg=C["card"]); h4h.pack(fill="x", pady=(0, 8))
        tk.Frame(h4h, bg=C["danger"], width=4, height=26).pack(side="left", padx=(0, 10))
        lbl_col4 = tk.Frame(h4h, bg=C["card"]); lbl_col4.pack(side="left", fill="x", expand=True)
        lbl(lbl_col4, "④ Hataları Hesapla", font=FH3, fg=C["danger"]).pack(anchor="w")
        lbl(lbl_col4, "Tüm adımları tamamladıktan sonra aşağıdaki düğmeye basın",
            font=FSM, fg=C["text_light"]).pack(anchor="w")

        # Live readiness banner — updates as inputs change
        self._ec_ready_box = tk.Frame(step4, bg="#FEF3C7",
                                       highlightbackground=C["warn"], highlightthickness=1)
        self._ec_ready_box.pack(fill="x", pady=(0, 10))
        self._ec_ready_lbl = lbl(self._ec_ready_box, "", font=FSM_B, fg=C["warn"],
                                  bg="#FEF3C7", wraplength=520, justify="left")
        self._ec_ready_lbl.pack(anchor="w", padx=10, pady=8)

        self._ec_run_btn = btn(step4, self.t("run_error_calc"), self._ec_run,
                                C["danger"], width=28)
        self._ec_run_btn.pack(fill="x", ipady=6)

        self._ec_update_ready()

        # ── Results ──
        self._ec_results_frame = card(body, padx=16, pady=14)
        self._ec_results_frame.pack(fill="both", expand=True, pady=(0, 10))
        h4 = tk.Frame(self._ec_results_frame, bg=C["card"]); h4.pack(fill="x", pady=(0, 8))
        tk.Frame(h4, bg=C["success"], width=4, height=26).pack(side="left", padx=(0, 10))
        lbl(h4, self.t("err_results"), font=FH3, fg=C["primary"]).pack(side="left", anchor="w")
        self._ec_results_inner = tk.Frame(self._ec_results_frame, bg=C["card"])
        self._ec_results_inner.pack(fill="both", expand=True)
        lbl(self._ec_results_inner, "Hesaplamayı çalıştırınca MAE, RMSE ve grafikler burada görünür.",
            font=FSM, fg=C["text_light"]).pack(anchor="w", padx=4)

    # ── Error calc helpers ──

    def _ec_update_sel_count(self):
        """Refresh the 'X selected — Ready/Need more' badge in step ②."""
        if not hasattr(self, "_ec_sel_lbl"):
            return
        try:
            n = len(self._ec_state.get("selected", set()))
            color = C["success"] if n >= 2 else C["warn"]
            self._ec_sel_lbl.config(
                text=f"{n} seçili  —  {'✔ Hazır' if n >= 2 else f'En az {2-n} tane daha seç'}",
                fg=color)
        except Exception:
            pass

    def _ec_update_ready(self):
        """Update the live readiness banner above the Calculate button.

        Aggregates three conditions:
          • ≥2 calibration records selected
          • verification CSV picked
          • either CSV has ref cols OR user filled SBP/DBP refs
        Only the first two are blocking — the ref fields are warned about but
        not required, since the CSV's own ref_sbp/ref_dbp columns may exist.
        """
        if not hasattr(self, "_ec_ready_lbl"):
            return
        try:
            st = self._ec_state
            n_sel = len(st.get("selected", set()))
            has_verif = bool(st.get("verif_path"))
            ref_sbp = st.get("verif_ref_sbp", "")
            ref_dbp = st.get("verif_ref_dbp", "")
            has_ref = bool(str(ref_sbp).strip() and str(ref_dbp).strip())

            problems = []
            if n_sel < 2:
                problems.append(f"② En az {2 - n_sel} kalibrasyon kaydı daha seçin")
            if not has_verif:
                problems.append("③ Doğrulama CSV dosyasını yükleyin")

            if not problems:
                # All blocking conditions met
                if has_ref:
                    txt = f"✔ Hazır  —  {n_sel} kayıt seçili · doğrulama yüklü · referans BP girildi"
                else:
                    txt = (f"✔ Hazır  —  {n_sel} kayıt seçili · doğrulama yüklü "
                           f"(referans BP boş: CSV'de ref_sbp/ref_dbp varsa otomatik kullanılır)")
                self._ec_ready_box.config(bg="#DCFCE7", highlightbackground=C["success"])
                self._ec_ready_lbl.config(text=txt, fg=C["success"], bg="#DCFCE7")
                if hasattr(self, "_ec_run_btn"):
                    self._ec_run_btn.config(state="normal", bg=C["danger"])
            else:
                txt = "Eksikler:\n  • " + "\n  • ".join(problems)
                self._ec_ready_box.config(bg="#FEF3C7", highlightbackground=C["warn"])
                self._ec_ready_lbl.config(text=txt, fg=C["warn"], bg="#FEF3C7")
                if hasattr(self, "_ec_run_btn"):
                    # Keep button clickable so the user gets a clear messagebox
                    # if they try anyway — but visually mute it so the readiness
                    # banner is the obvious thing to read first.
                    self._ec_run_btn.config(state="normal", bg=C["text_light"])
        except Exception:
            pass

    def _ec_select_verif(self):
        path = filedialog.askopenfilename(
            title=self.t("meas_csv_title"), filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self._ec_state["verif_path"] = path
        try:
            # Refresh the verification panel text + colored border
            base = os.path.basename(path)
            self._ec_verif_lbl.config(text=f"✔  Yüklendi: {base}", fg=C["success"],
                                       bg="#ECFDF5")
            self._ec_verif_box.config(bg="#ECFDF5", highlightbackground=C["success"])
            # Recolor Frames and Labels inside the box (skip the Browse button)
            def _recolor(widget):
                if isinstance(widget, (tk.Frame, tk.Label)):
                    try: widget.config(bg="#ECFDF5")
                    except Exception: pass
                    for sub in widget.winfo_children():
                        _recolor(sub)
            for child in self._ec_verif_box.winfo_children():
                _recolor(child)
        except Exception:
            pass
        self._ec_update_ready()

    def _ec_run(self):
        model: CalibModel = self.db.users[self.current_user]["model"]
        st = self._ec_state
        method = self._ec_method_var.get()

        selected_records = [model.calibrations[i]
                            for i in sorted(st["selected"])
                            if i < len(model.calibrations)]
        if len(selected_records) < 2:
            messagebox.showwarning("Yetersiz Kayıt",
                                   "Model fit için en az 2 kalibrasyon kaydı seçin.")
            return

        pat_key = "pat_foot" if method == "foot" else "pat_peak"
        calib_data = []
        for c in selected_records:
            pat = c.get(pat_key)
            if pat is None:
                continue
            calib_data.append((float(pat), float(c["sbp"]), float(c["dbp"])))

        if len(calib_data) < 2:
            messagebox.showwarning("Yetersiz Veri",
                                   "Seçili kayıtlarda geçerli PAT değeri olan en az 2 kayıt gerekli.")
            return

        if not st["verif_path"]:
            messagebox.showwarning(self.t("warning"), self.t("err_need_verif")); return
        try:
            verif_df = pd.read_csv(st["verif_path"])
            assert {"ecg", "ppg"}.issubset(verif_df.columns)
        except Exception:
            messagebox.showerror(self.t("error"), self.t("csv_ecg_ppg")); return

        pats = np.array([d[0] for d in calib_data])
        sbps = np.array([d[1] for d in calib_data])
        dbps = np.array([d[2] for d in calib_data])
        s_a, s_b = np.polyfit(pats, sbps, 1)
        d_a, d_b = np.polyfit(pats, dbps, 1)

        has_ref_cols = {"ref_sbp", "ref_dbp"}.issubset(verif_df.columns)
        user_ref_sbp = user_ref_dbp = None
        if not has_ref_cols:
            try:
                user_ref_sbp = float(st["verif_ref_sbp"])
                user_ref_dbp = float(st["verif_ref_dbp"])
                assert 60 < user_ref_sbp < 220 and 40 < user_ref_dbp < 140
            except Exception:
                messagebox.showwarning("Referans BP Gerekli",
                    "CSV'de ref_sbp/ref_dbp sütunu bulunamadı.\n"
                    "Lütfen adım ③'teki Referans SBP / DBP alanlarını doldurun.")
                return

        streamer = CSVStreamer(verif_df)
        pred_sbp_list, pred_dbp_list, pat_list = [], [], []
        ref_sbp_list, ref_dbp_list = [], []
        win_idx = 0
        while not streamer.done:
            seg = streamer.next_window()
            if seg is None:
                break
            win_idx += 1
            res: PATResult = calculate_pat_both(seg["ecg"].values, seg["ppg"].values)
            pat = res.foot if method == "foot" else res.peak
            if pat is None:
                continue
            pred_sbp_list.append(s_a * pat + s_b)
            pred_dbp_list.append(d_a * pat + d_b)
            pat_list.append(pat)
            if has_ref_cols:
                start = (win_idx - 1) * STEP_SEC * FS
                sl = verif_df.iloc[int(start):int(start + WIN_SEC * FS)]
                ref_sbp_list.append(float(sl["ref_sbp"].median()))
                ref_dbp_list.append(float(sl["ref_dbp"].median()))
            else:
                ref_sbp_list.append(user_ref_sbp)
                ref_dbp_list.append(user_ref_dbp)

        if not pred_sbp_list:
            messagebox.showerror(self.t("error"),
                                 "Doğrulama dosyasından geçerli pencere çıkarılamadı.")
            return

        pred_sbp_arr = np.array(pred_sbp_list); pred_dbp_arr = np.array(pred_dbp_list)
        ref_sbp_arr  = np.array(ref_sbp_list);  ref_dbp_arr  = np.array(ref_dbp_list)
        pat_arr      = np.array(pat_list)

        mae_sbp  = float(np.mean(np.abs(pred_sbp_arr - ref_sbp_arr)))
        mae_dbp  = float(np.mean(np.abs(pred_dbp_arr - ref_dbp_arr)))
        rmse_sbp = float(np.sqrt(np.mean((pred_sbp_arr - ref_sbp_arr)**2)))
        rmse_dbp = float(np.sqrt(np.mean((pred_dbp_arr - ref_dbp_arr)**2)))

        ref_label = ("CSV ref sütunundan" if has_ref_cols else
                     f"Kullanıcı girişi (SBP={user_ref_sbp:.0f}, DBP={user_ref_dbp:.0f})")

        self._ec_show_results(
            method, s_a, s_b, d_a, d_b,
            pats, sbps, dbps,
            pat_arr, pred_sbp_arr, pred_dbp_arr, ref_sbp_arr, ref_dbp_arr,
            mae_sbp, mae_dbp, rmse_sbp, rmse_dbp,
            has_ref_cols, ref_label=ref_label
        )

    def _ec_show_results(self, method,
                         s_a, s_b, d_a, d_b,
                         calib_pats, calib_sbps, calib_dbps,
                         verif_pats, pred_sbps, pred_dbps, ref_sbps, ref_dbps,
                         mae_sbp, mae_dbp, rmse_sbp, rmse_dbp,
                         has_ref_cols, ref_label=None):
        """Clear the results area and render metrics + plots."""
        for w in self._ec_results_inner.winfo_children():
            w.destroy()

        mcolor = C["foot"] if method == "foot" else C["peak"]
        method_label = "🟣 Foot" if method == "foot" else "🔵 Peak"
        n = len(pred_sbps)

        # ── Fitted Model row (accent bar, like calibration section headers) ──
        model_row = tk.Frame(self._ec_results_inner, bg=C["card"])
        model_row.pack(fill="x", pady=(0, 10))
        tk.Frame(model_row, bg=mcolor, width=4, height=52).pack(side="left", padx=(0, 10))
        model_txt = tk.Frame(model_row, bg=C["card"])
        model_txt.pack(side="left", fill="x", expand=True)
        lbl(model_txt, f"Fitted Model — {method_label}", font=FB_B, fg=mcolor).pack(anchor="w")
        lbl(model_txt, self.t("formula_sbp", s_a, s_b), font=FMONO_B, fg=C["success"]).pack(anchor="w")
        lbl(model_txt, self.t("formula_dbp", d_a, d_b), font=FMONO_B, fg=C["danger"]).pack(anchor="w")

        # ── Metric tiles ──
        metrics_f = tk.Frame(self._ec_results_inner, bg=C["card"])
        metrics_f.pack(fill="x", pady=(0, 8))

        def metric_tile(parent, title, value, unit, color, formula_line):
            tf = tk.Frame(parent, bg=color, padx=14, pady=10,
                          highlightbackground=C["border"], highlightthickness=1)
            tf.pack(side="left", fill="both", expand=True, padx=(0, 6))
            tk.Label(tf, text=title, font=FSM_B, fg="white", bg=color).pack()
            tk.Label(tf, text=f"{value:.2f}", font=("Segoe UI", 28, "bold"),
                     fg="white", bg=color).pack()
            tk.Label(tf, text=unit, font=FSM, fg="white", bg=color).pack()
            tk.Label(tf, text=formula_line, font=("Courier New", 8),
                     fg="white", bg=color, wraplength=160, justify="center").pack(pady=(4, 0))

        metric_tile(metrics_f, self.t("mae_sbp"),  mae_sbp,  "mmHg", C["success"],
                    "(1/N)·Σ|pred−ref|")
        metric_tile(metrics_f, self.t("mae_dbp"),  mae_dbp,  "mmHg", C["success_lt"],
                    "(1/N)·Σ|pred−ref|")
        metric_tile(metrics_f, self.t("rmse_sbp"), rmse_sbp, "mmHg", C["primary"],
                    "√[(1/N)·Σ(pred−ref)²]")
        metric_tile(metrics_f, self.t("rmse_dbp"), rmse_dbp, "mmHg", C["primary_lt"],
                    "√[(1/N)·Σ(pred−ref)²]")

        # Source / window info bar
        if ref_label:
            src_text = f"Reference: {ref_label}"
        else:
            src_text = "✔ Ground truth from CSV" if has_ref_cols else "⚠️  Reference = mean calibration BP (no ref cols)"
        info_bar = tk.Frame(self._ec_results_inner, bg="#F1F5F9",
                            highlightbackground=C["border"], highlightthickness=1)
        info_bar.pack(fill="x", pady=(0, 8))
        lbl(info_bar, f"  {self.t('n_windows')}: {n}   |   {src_text}",
            font=FSM, fg=C["text_sub"], bg="#F1F5F9").pack(anchor="w", padx=6, pady=5)

        # ── Plots ──
        plot_f = tk.Frame(self._ec_results_inner, bg=C["card"])
        plot_f.pack(fill="both", expand=True)

        fig = Figure(figsize=(13, 7), dpi=80, facecolor=C["card"])
        fig.subplots_adjust(hspace=0.42, wspace=0.32)

        ax_sbp_fit  = fig.add_subplot(231)
        ax_dbp_fit  = fig.add_subplot(232)
        ax_err      = fig.add_subplot(233)
        ax_sbp_pred = fig.add_subplot(234)
        ax_dbp_pred = fig.add_subplot(235)
        ax_pat      = fig.add_subplot(236)

        win_x = np.arange(1, n + 1)

        # ── Subplot 1: SBP calibration fit ──
        pat_line = np.linspace(calib_pats.min() - 10, calib_pats.max() + 10, 200)
        sbp_line = s_a * pat_line + s_b
        dbp_line = d_a * pat_line + d_b

        ax_sbp_fit.scatter(calib_pats, calib_sbps, color=C["success"], zorder=3, s=50, label="Calib pts")
        ax_sbp_fit.plot(pat_line, sbp_line, color=C["success_lt"], linewidth=2, label="Fit")
        ax_sbp_fit.set_title("SBP Calibration Fit", fontsize=9, fontweight="bold", color=C["text"])
        ax_sbp_fit.set_xlabel("PAT (ms)", fontsize=8, color=C["text_light"])
        ax_sbp_fit.set_ylabel("SBP (mmHg)", fontsize=8, color=C["text_light"])
        ax_sbp_fit.legend(fontsize=7)
        ax_sbp_fit.grid(True, alpha=0.2); ax_sbp_fit.set_facecolor("white")
        for sp in ax_sbp_fit.spines.values(): sp.set_color(C["border"])
        ax_sbp_fit.tick_params(colors=C["text_light"], labelsize=7)

        # ── Subplot 2: DBP calibration fit ──
        ax_dbp_fit.scatter(calib_pats, calib_dbps, color=C["danger"], zorder=3, s=50, label="Calib pts")
        ax_dbp_fit.plot(pat_line, dbp_line, color=C["danger_lt"], linewidth=2, label="Fit")
        ax_dbp_fit.set_title("DBP Calibration Fit", fontsize=9, fontweight="bold", color=C["text"])
        ax_dbp_fit.set_xlabel("PAT (ms)", fontsize=8, color=C["text_light"])
        ax_dbp_fit.set_ylabel("DBP (mmHg)", fontsize=8, color=C["text_light"])
        ax_dbp_fit.legend(fontsize=7)
        ax_dbp_fit.grid(True, alpha=0.2); ax_dbp_fit.set_facecolor("white")
        for sp in ax_dbp_fit.spines.values(): sp.set_color(C["border"])
        ax_dbp_fit.tick_params(colors=C["text_light"], labelsize=7)

        # ── Subplot 3: Absolute error per window ──
        err_sbp = np.abs(pred_sbps - ref_sbps)
        err_dbp = np.abs(pred_dbps - ref_dbps)
        ax_err.bar(win_x - 0.2, err_sbp, width=0.4, color=C["success"],   label="SBP err", alpha=0.8)
        ax_err.bar(win_x + 0.2, err_dbp, width=0.4, color=C["danger"],    label="DBP err", alpha=0.8)
        ax_err.axhline(mae_sbp,  color=C["success_lt"], linestyle="--", linewidth=1.2,
                       label=f"MAE SBP={mae_sbp:.1f}")
        ax_err.axhline(mae_dbp,  color=C["danger_lt"],  linestyle="--", linewidth=1.2,
                       label=f"MAE DBP={mae_dbp:.1f}")
        ax_err.set_title("Absolute Error per Window", fontsize=9, fontweight="bold", color=C["text"])
        ax_err.set_xlabel("Window #", fontsize=8, color=C["text_light"])
        ax_err.set_ylabel("|Error| (mmHg)", fontsize=8, color=C["text_light"])
        ax_err.legend(fontsize=7)
        ax_err.grid(True, alpha=0.2); ax_err.set_facecolor("white")
        for sp in ax_err.spines.values(): sp.set_color(C["border"])
        ax_err.tick_params(colors=C["text_light"], labelsize=7)

        # ── Subplot 4: Predicted vs Reference SBP ──
        ax_sbp_pred.plot(win_x, ref_sbps,  color=C["text_sub"], linewidth=1.4,
                         linestyle="--", label="Reference")
        ax_sbp_pred.plot(win_x, pred_sbps, color=C["success"],  linewidth=2,
                         marker="o", markersize=3, label="Predicted")
        ax_sbp_pred.axhline(120, color=C["warn"],   linestyle=":", alpha=0.5, linewidth=1)
        ax_sbp_pred.axhline(140, color=C["danger"], linestyle=":", alpha=0.5, linewidth=1)
        ax_sbp_pred.set_title("SBP: Predicted vs Reference", fontsize=9, fontweight="bold", color=C["text"])
        ax_sbp_pred.set_xlabel("Window #", fontsize=8, color=C["text_light"])
        ax_sbp_pred.set_ylabel("SBP (mmHg)", fontsize=8, color=C["text_light"])
        ax_sbp_pred.legend(fontsize=7)
        ax_sbp_pred.grid(True, alpha=0.2); ax_sbp_pred.set_facecolor("white")
        for sp in ax_sbp_pred.spines.values(): sp.set_color(C["border"])
        ax_sbp_pred.tick_params(colors=C["text_light"], labelsize=7)

        # ── Subplot 5: Predicted vs Reference DBP ──
        ax_dbp_pred.plot(win_x, ref_dbps,  color=C["text_sub"], linewidth=1.4,
                         linestyle="--", label="Reference")
        ax_dbp_pred.plot(win_x, pred_dbps, color=C["danger"],   linewidth=2,
                         marker="o", markersize=3, label="Predicted")
        ax_dbp_pred.set_title("DBP: Predicted vs Reference", fontsize=9, fontweight="bold", color=C["text"])
        ax_dbp_pred.set_xlabel("Window #", fontsize=8, color=C["text_light"])
        ax_dbp_pred.set_ylabel("DBP (mmHg)", fontsize=8, color=C["text_light"])
        ax_dbp_pred.legend(fontsize=7)
        ax_dbp_pred.grid(True, alpha=0.2); ax_dbp_pred.set_facecolor("white")
        for sp in ax_dbp_pred.spines.values(): sp.set_color(C["border"])
        ax_dbp_pred.tick_params(colors=C["text_light"], labelsize=7)

        # ── Subplot 6: PAT over verification windows ──
        ax_pat.plot(win_x, verif_pats, color=mcolor, linewidth=1.8, marker="o", markersize=3)
        ax_pat.set_title(f"PAT ({method_label}) — Verif Windows", fontsize=9, fontweight="bold", color=C["text"])
        ax_pat.set_xlabel("Window #", fontsize=8, color=C["text_light"])
        ax_pat.set_ylabel("PAT (ms)", fontsize=8, color=C["text_light"])
        ax_pat.grid(True, alpha=0.2); ax_pat.set_facecolor("white")
        for sp in ax_pat.spines.values(): sp.set_color(C["border"])
        ax_pat.tick_params(colors=C["text_light"], labelsize=7)

        canvas = FigureCanvasTkAgg(fig, plot_f)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _build_top_bar(self, title: str, color: str):
        top = tk.Frame(self.root, bg=color, padx=16, pady=14)
        top.pack(fill="x")
        self._lang_btn(top)
        lbl(top, title, font=FH3, fg="white", bg=color).pack(side="left")
        lbl(top, f"  {self.current_user}", font=FSM, fg="white", bg=color).pack(side="left")
        btn(top, self.t("back"), self._show_dashboard, color, width=8, size="small").pack(side="right")
 
 
# ════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    BPApp(root)
    root.mainloop()