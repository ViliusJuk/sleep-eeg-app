import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import tempfile
import os

import torch
import torch.nn.functional as F

from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

ANNOT_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # sujungiam į N3
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

import numpy as np
import pandas as pd

def stage_distribution_df(y_stage, stage_names=STAGE_NAMES):
    """
    y_stage: 1D array su stadijų ID (0..4)
    Grąžina DataFrame: Stadija | Kiekis | Dalis
    """
    y_stage = np.asarray(y_stage).astype(int)

    # saugumas: paliekam tik galiojančias klases 0..4
    y_stage = y_stage[(y_stage >= 0) & (y_stage < len(stage_names))]

    counts = np.bincount(y_stage, minlength=len(stage_names))
    total = int(counts.sum())
    pct = (counts / total * 100.0) if total > 0 else np.zeros_like(counts, dtype=float)

    return pd.DataFrame({
        "Stadija": stage_names,
        "Kiekis": counts.astype(int),
        "Dalis": [f"{p:.1f} %" for p in pct],
    })



# ==========================
# Small helpers
# ==========================

def stage_name_from_id(cls_id: int) -> str:
    if 0 <= int(cls_id) < len(STAGE_NAMES):
        return STAGE_NAMES[int(cls_id)]
    return str(cls_id)


def zscore_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x: (..., T) -> zscore per paskutinį matmenį
    """
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return (x - m) / (s + eps)


def make_center_sequence(X: np.ndarray, center_idx: int, seq_len: int) -> np.ndarray:
    """
    X: (E, 1, T)
    Returns: (L, T) su centru center_idx. Kraštuose edge padding.
    """
    E = X.shape[0]
    T = X.shape[2]
    half = seq_len // 2

    idxs = np.arange(center_idx - half, center_idx - half + seq_len)
    idxs = np.clip(idxs, 0, E - 1)

    seq = X[idxs, 0, :]  # (L, T)
    return seq.astype(np.float32)


def plot_hypnogram_time(starts_sec, y, title="Hipnograma"):
    x_h = np.array(starts_sec) / 3600.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(x_h, y, where="post")
    ax.set_yticks(np.arange(len(STAGE_NAMES)))
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Laikas nuo įrašo pradžios (val.)")
    ax.set_ylabel("Miego stadija")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_epoch_signal(x_1d, sfreq, title="Epocha"):
    t = np.arange(len(x_1d)) / float(sfreq)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x_1d)
    ax.set_xlabel("Laikas (s)")
    ax.set_ylabel("Amplitudė (a.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion(cm, title="Klasifikavimo lentelė", figsize=(4.5, 3.2), font=8):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title, fontsize=font + 2)
    ax.set_xlabel("Pred", fontsize=font)
    ax.set_ylabel("True", fontsize=font)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES, fontsize=font)
    ax.set_yticklabels(STAGE_NAMES, fontsize=font)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=font)

    fig.tight_layout()
    return fig


# ==========================
# EDF loading
# ==========================

def _save_bytes_to_tmp(file_bytes: bytes, suffix=".edf"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


@st.cache_data(show_spinner=False)
def load_from_edf_pair_safe(
    psg_bytes: bytes,
    hyp_bytes: bytes,
    channel="Fpz-Cz",
    target_sfreq=100,
    epoch_sec=30,
    do_filter=False,
):
    """
    Grąžina:
      X: (E,1,T)
      y: (E,)
      starts_sec: (E,)
      ch_name, sfreq
      total_events, dropped
    """
    psg_path = _save_bytes_to_tmp(psg_bytes, suffix=".edf")
    hyp_path = _save_bytes_to_tmp(hyp_bytes, suffix=".edf")

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")

        chs = raw.ch_names
        if channel not in chs:
            candidates = [c for c in chs if ("Fpz" in c and "Cz" in c) or ("FPZ" in c and "CZ" in c)]
            if not candidates:
                raise ValueError(f"Neradau kanalo '{channel}'. Galimi kanalai: {chs}")
            channel_use = candidates[0]
        else:
            channel_use = channel

        raw.pick_channels([channel_use])
        raw.load_data()

        if target_sfreq is not None:
            raw.resample(target_sfreq)
        if do_filter:
            raw.filter(0.3, 40.0)

        sfreq = float(raw.info["sfreq"])

        ann = mne.read_annotations(hyp_path)
        raw.set_annotations(ann)

        events_all, _ = mne.events_from_annotations(
            raw,
            event_id=ANNOT_MAP,
            chunk_duration=float(epoch_sec),
        )

        if events_all.size == 0:
            raise ValueError("events_from_annotations grąžino 0 įvykių. Patikrink Hypnogram EDF.")

        total_events = int(events_all.shape[0])

        keep = events_all[:, 2] != -1
        events = events_all[keep]
        dropped = total_events - int(events.shape[0])

        if events.shape[0] == 0:
            raise ValueError("Po atmetimo (?/movement) neliko valid epochų.")

        data = raw.get_data()  # (1, N)
        T = int(epoch_sec * sfreq)

        X_list, y_list, starts_list = [], [], []
        for ev in events:
            start = int(ev[0])
            seg = data[:, start:start + T]
            if seg.shape[1] != T:
                continue
            X_list.append(seg)
            y_list.append(int(ev[2]))
            starts_list.append(start / sfreq)

        if len(X_list) == 0:
            raise ValueError("Nepavyko suformuoti nė vienos pilnos epochos.")

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        starts_sec = np.array(starts_list, dtype=np.float64)

        return X, y, starts_sec, channel_use, sfreq, total_events, dropped

    finally:
        try:
            os.remove(psg_path)
        except Exception:
            pass
        try:
            os.remove(hyp_path)
        except Exception:
            pass


# ==========================
# Model loading + inference
# ==========================

@st.cache_resource(show_spinner=False)
def load_torchscript_model_from_bytes(model_bytes: bytes):
    """
    TorchScript modelis (cnn_bilstm_full_ts.pt) -> CPU
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_bytes)
        tmp.flush()
        path = tmp.name
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def predict_full_night_seq_batched(
    model,
    X: np.ndarray,          # (E,1,T)
    seq_len: int,
    batch_size: int,
    device: str,
    progress_cb=None,
):
    """
    Prognozė visai nakčiai su seq_len (center prediction).
    Grąžina:
      y_pred: (E,)
      probs:  (E,5)
    """
    model.eval()
    E = X.shape[0]
    y_pred = np.zeros(E, dtype=np.int64)
    probs_all = np.zeros((E, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, E, batch_size):
            end = min(E, start + batch_size)

            seq_batch = []
            for i in range(start, end):
                seq = make_center_sequence(X, i, int(seq_len))  # (L,T)
                seq_batch.append(seq)

            seq_batch = np.stack(seq_batch).astype(np.float32)   # (B,L,T)
            seq_batch = zscore_np(seq_batch)                     # zscore per epochą (per T)

            xb = torch.from_numpy(seq_batch).to(device)          # (B,L,T)
            logits = model(xb)
            pb = F.softmax(logits, dim=1).cpu().numpy()

            y_pred[start:end] = pb.argmax(axis=1)
            probs_all[start:end] = pb

            if progress_cb is not None:
                progress_cb(end / E)

    return y_pred, probs_all


# ==========================
# Streamlit UI
# ==========================

st.set_page_config(page_title="Sleep EEG demo", layout="wide")
st.title("Miego EEG demonstracinė aplikacija (Sleep-EDF)")

# init session state
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "e_idx" not in st.session_state:
    st.session_state.e_idx = 0

# ---------- Sidebar (data) ----------
st.sidebar.header("1) Įkelk Sleep-EDF EDF porą")
psg = st.sidebar.file_uploader("PSG failas (-PSG.edf)", type=["edf"])
hyp = st.sidebar.file_uploader("Hypnogram failas (-Hypnogram.edf)", type=["edf"])

do_filter = st.sidebar.checkbox("Taikyti band-pass 0.3–40 Hz", value=False)
target_sfreq = st.sidebar.selectbox("Resample", [None, 100], index=1)
epoch_sec = st.sidebar.number_input("Epoch length (s)", value=30, min_value=5, max_value=60, step=5)

st.sidebar.markdown("### Demo vaizdavimas")
load_btn = st.sidebar.button("Užkrauti duomenis", type="primary", use_container_width=True)

st.sidebar.divider()

# ---------- Sidebar (model) ----------
st.sidebar.header("2) Modelio įkėlimas")
bilstm_file = st.sidebar.file_uploader("Įkelk modelio Torchscript", type=["pt"])
seq_len = st.sidebar.selectbox("Seq len (L)", [10, 20, 30, 40], index=1)  # default 20
run_bilstm_single = st.sidebar.checkbox("Rodyti prognozę pasirinktai epochai", value=True)

st.sidebar.markdown("### Visa naktis (modelis)")
batch_size = st.sidebar.slider("Batch size", 8, 256, 64, 8)
run_full_btn = st.sidebar.button("▶ Prognozuoti visą naktį", use_container_width=True)

# status
st.sidebar.markdown("### Įkėlimo statusas")
if psg is None:
    st.sidebar.info("PSG: neįkeltas")
else:
    st.sidebar.success(f"PSG: {psg.name}")
if hyp is None:
    st.sidebar.info("Hyp: neįkeltas")
else:
    st.sidebar.success(f"Hyp: {hyp.name}")
if bilstm_file is None:
    st.sidebar.info("Modelis: neįkeltas")
else:
    st.sidebar.success(f"Modelis: {bilstm_file.name}")

# ---------- Load data ----------
if load_btn:
    if (psg is None) or (hyp is None):
        st.warning("Įkelk abu failus ir tik tada spausk Užkrauti.")
    else:
        try:
            with st.spinner("Skaitau EDF ir formuoju epochas..."):
                X, y, starts_sec, ch_name, sfreq, total_events, dropped = load_from_edf_pair_safe(
                    psg.getvalue(),
                    hyp.getvalue(),
                    target_sfreq=target_sfreq,
                    epoch_sec=int(epoch_sec),
                    do_filter=do_filter,
                )

            st.session_state.loaded = True
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.starts_sec = starts_sec
            st.session_state.ch_name = ch_name
            st.session_state.sfreq = sfreq
            st.session_state.total_events = total_events
            st.session_state.dropped = dropped
            st.session_state.epoch_sec = int(epoch_sec)
            st.session_state.e_idx = 0

            # invalidate previous full-night results when data reloads
            st.session_state.pop("y_pred_full", None)
            st.session_state.pop("probs_full", None)
            st.session_state.pop("full_cache_key", None)

            st.success("Duomenys užkrauti sėkmingai!")

        except Exception as e:
            st.session_state.loaded = False
            st.error("Nepavyko užkrauti / apdoroti EDF failų.")
            st.exception(e)

# stop if not loaded
if not st.session_state.loaded:
    st.info("Įkelk PSG + Hypnogram failus kairėje ir spausk **Užkrauti duomenis**.")
    st.stop()

# ---------- Load model (optional) ----------
model_bilstm = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if bilstm_file is not None:
    try:
        model_bilstm = load_torchscript_model_from_bytes(bilstm_file.getvalue())
        # torchscript loaded on cpu; move to device if needed
        model_bilstm = model_bilstm.to(device)
        model_bilstm.eval()
    except Exception as e:
        st.error("Nepavyko užkrauti TorchScript modelio.")
        st.exception(e)
        model_bilstm = None

# ---------- Main render ----------
X = st.session_state.X
y = st.session_state.y
starts_sec = st.session_state.starts_sec
ch_name = st.session_state.ch_name
sfreq = st.session_state.sfreq
total_events = st.session_state.total_events
dropped = st.session_state.dropped
epoch_sec = st.session_state.epoch_sec

duration_min = (float(starts_sec[-1]) + float(epoch_sec)) / 60.0
unique, counts = np.unique(y, return_counts=True)
share = {stage_name_from_id(int(k)): float(v) / len(y) * 100.0 for k, v in zip(unique, counts)}

cA, cB, cC, cD = st.columns(4)
cA.metric("Kanalas", ch_name)
cB.metric("Dažnis (Hz)", f"{sfreq:.1f}")
cC.metric("Epochų skaičius", f"{X.shape[0]}")
cD.metric("Atmestos epochos (atmestos/visos)", f"{dropped} / {total_events}")

st.write(f"**Įrašo trukmė (pagal tinkamas epochas):** ~{duration_min:.0f} min")
st.write(
    "**Miego stadijų pasiskirstymas procentais:** "
    + ", ".join([f"{k} {v:.1f}%" for k, v in sorted(share.items(), key=lambda x: x[0])])
)

# window selection for visualization
mask = np.ones_like(y, dtype=bool)
window_title = "Hipnograma (visa naktis)"

Xw = X[mask]
yw = y[mask]
sw = starts_sec[mask]

if st.session_state.e_idx >= int(Xw.shape[0]):
    st.session_state.e_idx = 0

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Hipnograma")
    st.pyplot(plot_hypnogram_time(sw, yw, title=window_title))
with col2:
    st.subheader("Miego stadijų pasiskirstymas (lange)")
    u2, c2 = np.unique(yw, return_counts=True)
    total2 = len(yw)
    rows = []
    for cls_id, cnt in zip(u2, c2):
        name = stage_name_from_id(int(cls_id))
        perc = 100.0 * cnt / total2
        rows.append((name, int(cnt), f"{perc:.1f} %"))
    st.table({
        "Stadija": [r[0] for r in rows],
        "Kiekis": [r[1] for r in rows],
        "Dalis": [r[2] for r in rows],
    })
    st.bar_chart({r[0]: r[1] for r in rows})


# ---------- Full night prediction ----------
st.divider()
st.subheader("Visos nakties prognozė, metrikos ir klasifikavimo lentelė")

cache_key = (
    bilstm_file.name if bilstm_file is not None else "",
    int(seq_len),
    int(X.shape[0]),
    int(X.shape[2]),
    int(epoch_sec),
    float(sfreq),
)

if run_full_btn:
    if model_bilstm is None:
        st.warning("Neįkeltas modelio TorchScript!")
    else:
        try:
            prog = st.progress(0.0)
            with st.spinner("Skaičiuojama prognozė visai nakčiai..."):
                y_pred_full, probs_full = predict_full_night_seq_batched(
                    model=model_bilstm,
                    X=X,  # visa naktis
                    seq_len=int(seq_len),
                    batch_size=int(batch_size),
                    device=device,
                    progress_cb=lambda p: prog.progress(min(1.0, float(p))),
                )

            st.session_state.full_cache_key = cache_key
            st.session_state.y_pred_full = y_pred_full
            st.session_state.probs_full = probs_full

            st.success("Prognozė visai nakčiai paruošta!")

        except Exception as e:
            st.error("Nepavyko suskaičiuoti prognozių visai nakčiai.")
            st.exception(e)

# show results if available
has_full = ("y_pred_full" in st.session_state) and (st.session_state.get("full_cache_key") == cache_key)

if not has_full:
    st.info("Paspausk ** Prognozuoti visą naktį** sidebar'e, kad sugeneruočiau pred hipnogramą, metrikas ir klasifikavimo lentelę.")
else:
    y_pred_full = st.session_state.y_pred_full

    acc = accuracy_score(y, y_pred_full)
    f1 = f1_score(y, y_pred_full, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y, y_pred_full)

    m1, m2, m3 = st.columns(3)
    m1.metric("Tikslumas (visa naktis)", f"{acc * 100:.1f}%")
    m2.metric("Macro-F1 (visa naktis)", f"{f1:.3f}")
    m3.metric("Cohen’s κ", f"{kappa:.3f}")

    st.subheader("Miego stadijų pasiskirstymas (pagal modelio prognozę)")

    colD1, colD2 = st.columns(2)

    with colD1:
        st.markdown("**Tikra hipnograma (visa naktis)**")
        df_true = stage_distribution_df(y)
        st.dataframe(df_true, use_container_width=True)

    with colD2:
        st.markdown("**Modelio prognozė (visa naktis)**")
        df_pred = stage_distribution_df(y_pred_full)
        st.dataframe(df_pred, use_container_width=True)


    colA, colB = st.columns(2)
    with colA:
        st.pyplot(plot_hypnogram_time(starts_sec, y, title="Tikroji hipnograma (visa naktis)"))
    with colB:
        st.pyplot(plot_hypnogram_time(starts_sec, y_pred_full, title="Prognozuota hipnograma (visa naktis)"))

    cm = confusion_matrix(y, y_pred_full, labels=[0, 1, 2, 3, 4])
    

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        st.pyplot(
            plot_confusion(
                cm,
                title="Klasifikavimo lentelė (visa naktis)",
                figsize=(4.5, 3.2),
                font=8
            ),
            use_container_width=False,
        )

    # small view
    #st.pyplot(
    #    plot_confusion(cm, title="Klasifikavimo lentelė (visa naktis)", figsize=(4.5, 3.2), font=8),
    #    use_container_width=False,
    #)

