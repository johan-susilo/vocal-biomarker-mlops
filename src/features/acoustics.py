import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth as pm
from parselmouth.praat import call as praat_call
from scipy.stats import skew, kurtosis, linregress
import warnings

VALIDATION_MODE = True
INTERPOLATE_F0_IN_ML = True

PITCH_FLOOR = 60.0
PITCH_CEIL  = 600.0
FORMANT_MAX = 5000.0
TIME_STEP   = 0.01
C_SOUND     = 35000  # cm/s (VTL proxy)

INTENSITY_THRESH_DB = -35.0
TRIM_PAD_S = 0.03

np.seterr(all="ignore")
pd.options.display.float_format = "{:.6f}".format
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_sound(path: Path) -> pm.Sound:
    snd = pm.Sound(str(path))
    if snd.get_number_of_channels() > 1:
        snd = snd.extract_channel(1)
    x = np.asarray(snd.values, dtype=np.float64).flatten()
    x -= np.mean(x)
    return pm.Sound(x, sampling_frequency=snd.sampling_frequency)


def preprocess_sound(snd: pm.Sound) -> pm.Sound:
    intensity = snd.to_intensity()
    times = np.arange(snd.xmin, snd.xmax, TIME_STEP)
    vals = np.array([praat_call(intensity, "Get value at time", float(t), "Cubic") for t in times])

    mask = vals > INTENSITY_THRESH_DB
    if np.any(mask):
        idx = np.where(mask)[0]
        t0 = max(float(times[idx[0]]) - TRIM_PAD_S, snd.xmin)
        t1 = min(float(times[idx[-1]]) + TRIM_PAD_S, snd.xmax)
        if t1 > t0:
            snd = snd.extract_part(t0, t1, preserve_times=False)
    return snd


def get_pitch_track(snd: pm.Sound):
    pitch = snd.to_pitch_ac(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEIL)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    t = pitch.xs()
    return t, f0


def f0_stats(f0: np.ndarray,
             validation_mode: bool = VALIDATION_MODE,
             interpolate_in_ml: bool = INTERPOLATE_F0_IN_ML):
    keys = ["f0_mean","f0_median","f0_p05","f0_p95","f0_sd","f0_iqr","f0_range","f0_skew","f0_kurtosis"]
    if f0.size == 0 or np.all(np.isnan(f0)):
        return {k: np.nan for k in keys}

    if validation_mode:
        x = f0[~np.isnan(f0)].astype(float)
        if x.size == 0:
            return {k: np.nan for k in keys}
    else:
        x = f0.copy().astype(float)
        if interpolate_in_ml:
            nans = np.isnan(x)
            if np.any(nans):
                t_idx = np.arange(len(x))
                keep = ~nans
                if np.any(keep):
                    x[nans] = np.interp(t_idx[nans], t_idx[keep], x[keep])
                else:
                    return {k: np.nan for k in keys}
        else:
            x = x[~np.isnan(x)]
            if x.size == 0:
                return {k: np.nan for k in keys}

    out = {
        "f0_mean": float(np.mean(x)),
        "f0_median": float(np.median(x)),
        "f0_p05": float(np.percentile(x, 5)),
        "f0_p95": float(np.percentile(x, 95)),
        "f0_sd": float(np.std(x)),
        "f0_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "f0_range": float(np.percentile(x, 95) - np.percentile(x, 5)),
        "f0_skew": float(skew(x, bias=True)) if x.size >= 3 else np.nan,
        "f0_kurtosis": float(kurtosis(x, bias=True)) if x.size >= 4 else np.nan,
    }
    return out


def voicing_metrics(f0: np.ndarray, t: np.ndarray):
    voiced = ~np.isnan(f0)
    pct = 100.0 * float(np.mean(voiced)) if voiced.size else np.nan
    toggles = int(np.sum(np.abs(np.diff(voiced.astype(int))) == 1)) if voiced.size >= 2 else 0
    dur = float(t[-1] - t[0]) if t.size > 1 else np.nan
    bps = (toggles / 2.0) / dur if (dur is not None and np.isfinite(dur) and dur > 0) else np.nan

    return {
        "voicing_pct": pct,
        "breaks_per_min": (bps * 60.0) if np.isfinite(bps) else np.nan,
        "breaks_per_sec": bps
    }


def safe_pitch_features(snd: pm.Sound):
    try:
        t, f0 = get_pitch_track(snd)
        return f0_stats(f0) | voicing_metrics(f0, t)
    except Exception:
        return {k: np.nan for k in [
            "f0_mean","f0_median","f0_p05","f0_p95","f0_sd","f0_iqr",
            "f0_range","f0_skew","f0_kurtosis",
            "voicing_pct","breaks_per_min","breaks_per_sec"
        ]}


def jitter_shimmer_praat(snd: pm.Sound):
    try:
        pp = praat_call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEIL)
        return {
            "jitter_local": float(praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)),
            "jitter_RAP": float(praat_call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)),
            "jitter_PPQ5": float(praat_call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)),
            "shimmer_local": float(praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)),
            "shimmer_APQ11": float(praat_call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)),
        }
    except Exception:
        return {k: np.nan for k in ["jitter_local","jitter_RAP","jitter_PPQ5","shimmer_local","shimmer_APQ11"]}


def compute_true_cpps(snd: pm.Sound) -> float:
    if snd.duration < (3.0 / PITCH_FLOOR):
        return np.nan

    try:
        pcg = praat_call(snd, "To PowerCepstrogram", PITCH_FLOOR, 0.002, 5000.0, 50.0)
    except Exception:
        return np.nan

    cpps_settings = [
        (True, 0.01, 0.001, PITCH_FLOOR, 330.0, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust"),
        (True, 0.01, 0.001, PITCH_FLOOR, 330.0, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust slow"),
        (True, 0.0, 0.0, PITCH_FLOOR, 330.0, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust"),
    ]

    for args in cpps_settings:
        try:
            cpps = float(praat_call(pcg, "Get CPPS", *args))
            if np.isfinite(cpps):
                return cpps
        except Exception:
            continue

    return np.nan


def hnr_mean_std(snd: pm.Sound):
    try:
        harm = snd.to_harmonicity_cc(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
        return {
            "hnr_mean": float(praat_call(harm, "Get mean", 0, 0)),
            "hnr_sd": float(praat_call(harm, "Get standard deviation", 0, 0)),
        }
    except Exception:
        return {"hnr_mean": np.nan, "hnr_sd": np.nan}


def compute_ltas_praat(snd: pm.Sound):
    try:
        spectrum = snd.to_spectrum()
        ltas = praat_call(snd, "To Ltas", 100.0)

        freq_bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
        band_energies_db = {}
        for i, (low, high) in enumerate(freq_bands):
            energy = float(praat_call(spectrum, "Get band energy", low, high))
            band_energies_db[f"band_{i+1}_{low}_{high}Hz_dB"] = float(10 * np.log10(energy)) if energy > 0 else -100.0

        low_energy = band_energies_db.get("band_2_250_500Hz_dB", -100.0)
        mid_energy = band_energies_db.get("band_4_1000_2000Hz_dB", -100.0)
        high_energy = band_energies_db.get("band_5_2000_4000Hz_dB", -100.0)

        slope = ((high_energy - low_energy) / (np.log2(4000) - np.log2(250))) if (low_energy > -100 and high_energy > -100) else np.nan
        tilt_low_mid = (low_energy - mid_energy) if (low_energy > -100 and mid_energy > -100) else np.nan
        tilt_mid_high = (mid_energy - high_energy) if (mid_energy > -100 and high_energy > -100) else np.nan
        tilt_low_high = (low_energy - high_energy) if (low_energy > -100 and high_energy > -100) else np.nan

        total_energy = float(praat_call(spectrum, "Get band energy", 0, 20000))
        mean_db = float(10 * np.log10(total_energy)) if total_energy > 0 else np.nan

        low_alpha = float(praat_call(ltas, "Get mean", 50, 1000, "dB"))
        high_alpha = float(praat_call(ltas, "Get mean", 1000, 5000, "dB"))

        return {
            "LTAS_mean_db": mean_db,
            "LTAS_slope_db_per_oct": float(slope) if np.isfinite(slope) else np.nan,
            "LTAS_tilt_low_mid": float(tilt_low_mid) if np.isfinite(tilt_low_mid) else np.nan,
            "LTAS_tilt_mid_high": float(tilt_mid_high) if np.isfinite(tilt_mid_high) else np.nan,
            "LTAS_tilt_low_high": float(tilt_low_high) if np.isfinite(tilt_low_high) else np.nan,
            "LTAS_alpha_ratio": float(low_alpha - high_alpha),
            **band_energies_db
        }
    except Exception:
        return {
            "LTAS_mean_db": np.nan,
            "LTAS_slope_db_per_oct": np.nan,
            "LTAS_tilt_low_mid": np.nan,
            "LTAS_tilt_mid_high": np.nan,
            "LTAS_tilt_low_high": np.nan,
            "LTAS_alpha_ratio": np.nan
        }


def compute_formant_features(snd: pm.Sound):
    try:
        formant = snd.to_formant_burg(0.01, 5, FORMANT_MAX, 0.025, 50)
        F = {}

        for i in range(1, 5):
            F[f"F{i}_mean"] = float(praat_call(formant, "Get mean", i, 0, 0, "Hertz"))

        F1 = F.get("F1_mean", np.nan)
        F4 = F.get("F4_mean", np.nan)

        F["VTL_proxy_cm"] = float(C_SOUND / (4 * F1)) if np.isfinite(F1) and F1 > 0 else np.nan
        F["Formant_dispersion"] = float(F4 - F1) if np.isfinite(F4) and np.isfinite(F1) else np.nan
        return F
    except Exception:
        out = {f"F{i}_mean": np.nan for i in range(1, 5)}
        out["VTL_proxy_cm"] = np.nan
        out["Formant_dispersion"] = np.nan
        return out


def _nan_ltas():
    return {
        "LTAS_mean_db": np.nan,
        "LTAS_slope_db_per_oct": np.nan,
        "LTAS_tilt_low_mid": np.nan,
        "LTAS_tilt_mid_high": np.nan,
        "LTAS_tilt_low_high": np.nan,
        "LTAS_alpha_ratio": np.nan,
        "band_1_0_250Hz_dB": np.nan,
        "band_2_250_500Hz_dB": np.nan,
        "band_3_500_1000Hz_dB": np.nan,
        "band_4_1000_2000Hz_dB": np.nan,
        "band_5_2000_4000Hz_dB": np.nan,
        "band_6_4000_8000Hz_dB": np.nan,
    }


def features_sustain(snd: pm.Sound):
    out = safe_pitch_features(snd)
    out.update(jitter_shimmer_praat(snd))
    out["CPPS_db"] = np.nan
    out.update(hnr_mean_std(snd))
    out.update(_nan_ltas())
    out.update(compute_formant_features(snd))
    return out


def features_read(snd: pm.Sound):
    out = safe_pitch_features(snd)
    out.update(jitter_shimmer_praat(snd))
    out["CPPS_db"] = compute_true_cpps(snd)
    out.update(hnr_mean_std(snd))
    out.update(compute_ltas_praat(snd))
    out.update(compute_formant_features(snd))
    return out


def features_glide(snd: pm.Sound):
    out = safe_pitch_features(snd)
    out.update(jitter_shimmer_praat(snd))
    out["CPPS_db"] = np.nan
    out.update(hnr_mean_std(snd))
    out.update(_nan_ltas())
    out.update(compute_formant_features(snd))

    t, f0 = get_pitch_track(snd)
    mask = ~np.isnan(f0)
    f0v = f0[mask]
    tv = t[mask]

    glide = {
        "glide_span_Hz": np.nan, "glide_rate_Hz_s": np.nan, "glide_r2": np.nan,
        "glide_up_rate": np.nan, "glide_down_rate": np.nan, "glide_asymmetry": np.nan, "voice_break_index": np.nan
    }

    if f0v.size >= 5 and tv.size == f0v.size:
        span = float(np.nanmax(f0v) - np.nanmin(f0v))
        dur = float(tv[-1] - tv[0])

        slope, intercept, r, *_ = linregress(tv, f0v)
        mid = int(np.argmax(f0v))

        up = (f0v[mid] - f0v[0]) / (tv[mid] - tv[0]) if tv[mid] > tv[0] else np.nan
        down = (f0v[-1] - f0v[mid]) / (tv[-1] - tv[mid]) if tv[-1] > tv[mid] else np.nan
        asym = up / abs(down) if np.isfinite(up) and np.isfinite(down) and down != 0 else np.nan

        jumps = int(np.sum(np.abs(np.diff(np.log2(f0v))) > 0.08)) if f0v.size >= 2 else 0

        glide.update({
            "glide_span_Hz": span,
            "glide_rate_Hz_s": float(span / dur) if dur > 0 else np.nan,
            "glide_r2": float(r**2) if np.isfinite(r) else np.nan,
            "glide_up_rate": float(up) if np.isfinite(up) else np.nan,
            "glide_down_rate": float(down) if np.isfinite(down) else np.nan,
            "glide_asymmetry": float(asym) if np.isfinite(asym) else np.nan,
            "voice_break_index": float(jumps / f0v.size) if f0v.size > 0 else np.nan
        })

    out.update(glide)
    return out