import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

ANNOT_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

def stage_name_from_id(cls_id: int) -> str:
    if 0 <= int(cls_id) < len(STAGE_NAMES):
        return STAGE_NAMES[int(cls_id)]
    return str(cls_id)

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

def zscore_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x: (..., T)  -> zscore per paskutinį matmenį (kaip train_bilstm.py)
    """
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return (x - m) / (s + eps)

def make_center_sequence(Xw: np.ndarray, center_idx: int, seq_len: int) -> np.ndarray:
    """
    Xw: (E, 1, T)
    Grąžina seq: (L, T) su centru center_idx.
    Kraštuose pad'ina kraštinėmis epochomis (edge padding).
    """
    E = Xw.shape[0]
    T = Xw.shape[2]
    half = seq_len // 2

    idxs = np.arange(center_idx - half, center_idx - half + seq_len)
    idxs = np.clip(idxs, 0, E - 1)

    seq = Xw[idxs, 0, :]          # (L, T)
    return seq.astype(np.float32)

@st.cache_resource(show_spinner=False)
def load_torchscript_model_from_bytes(model_bytes: bytes):
    """
    Kad nereikėtų rašyti į diską ir kad per-rerun neperkrautų.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_bytes)
        tmp.flush()
        path = tmp.name
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def _save_bytes_to_tmp(file_bytes: bytes, suffix=".edf"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

def load_from_edf_pair_safe(
    psg_bytes: bytes,
    hyp_bytes: bytes,
    channel="Fpz-Cz",
    target_sfreq=100,
    epoch_sec=30,
    do_filter=False,
):
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
            chunk_duration=float(epoch_sec)
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


            # ---------------- Simple CNN model (state_dict keliui) ----------------

class SimpleSleepCNN(nn.Module):
    """
    Minimalus 1D CNN demo:
    input:  [B, 1, T]
    output: [B, 5] (logits)
    """
    def __init__(self, n_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: [B,1,T]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)  # [B,128]
        return self.fc(x)


def normalize_epoch(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalizacija vienai epochai.
    x: (T,)
    """
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + eps)

def make_sequence(X: np.ndarray, start_idx: int, seq_len: int) -> np.ndarray:
    """
    X: (E,1,T)
    return: (seq_len, T)   (nuimam channel dim)
    """
    seq = X[start_idx:start_idx + seq_len, 0, :]  # (L,T)
    return seq

def normalize_sequence(seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    seq: (L,T) -> z-score per epochą (kiekviena epocha atskirai)
    """
    mu = seq.mean(axis=1, keepdims=True)
    sd = seq.std(axis=1, keepdims=True) + eps
    return (seq - mu) / sd

def first_index_of_class(y: np.ndarray, class_id: int) -> int | None:
    idx = np.where(y == class_id)[0]
    return int(idx[0]) if len(idx) else None




# ---------------- UI ----------------

st.set_page_config(page_title="Sleep EEG demo", layout="wide")
st.title("Miego EEG demonstracinė aplikacija (Sleep-EDF)")

# init session state
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "e_idx" not in st.session_state:
    st.session_state.e_idx = 0

st.sidebar.header("Įkelk Sleep-EDF EDF porą")
psg = st.sidebar.file_uploader("PSG failas (`*-PSG.edf`)", type=["edf"])
hyp = st.sidebar.file_uploader("Hypnogram failas (`*-Hypnogram.edf`)", type=["edf"])

do_filter = st.sidebar.checkbox("Taikyti band-pass 0.3–40 Hz", value=False)
target_sfreq = st.sidebar.selectbox("Resample", [None, 100], index=1)
epoch_sec = st.sidebar.number_input("Epoch length (s)", value=30, min_value=5, max_value=60, step=5)

st.sidebar.markdown("### Demo vaizdavimas")
show_full_night = st.sidebar.checkbox("Rodyti visą naktį", value=False)
start_offset_min = st.sidebar.number_input("Start offset (min)", value=60, min_value=0, max_value=600, step=10)
window_len_min = st.sidebar.number_input("Window length (min)", value=180, min_value=30, max_value=600, step=30)

st.sidebar.divider()
st.sidebar.header("Modelis (CNN+BiLSTM TorchScript)")

model_file = st.sidebar.file_uploader(
    "Įkelk modelį `.pt` (TorchScript)",
    type=["pt"]
)

seq_len = st.sidebar.selectbox("Seq len (L)", [10, 20, 30, 40], index=1)  # default 20
run_model = st.sidebar.checkbox("Rodyti modelio prognozę", value=True)


load_btn = st.sidebar.button("Užkrauti", type="primary", use_container_width=True)

st.sidebar.markdown("### Įkėlimo statusas")
if psg is None:
    st.sidebar.info("PSG: neįkeltas")
else:
    st.sidebar.success(f"PSG: {psg.name} ({psg.size} B)")
if hyp is None:
    st.sidebar.info("Hyp: neįkeltas")
else:
    st.sidebar.success(f"Hyp: {hyp.name} ({hyp.size} B)")

# Load action
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

            # store in session_state
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
            st.success("Duomenys užkrauti sėkmingai ✅")

        except Exception as e:
            st.session_state.loaded = False
            st.error("Nepavyko užkrauti / apdoroti EDF failų. Žemiau – klaidos detalės:")
            st.exception(e)

# If not loaded yet
if not st.session_state.loaded:
    st.info("Įkelk PSG + Hypnogram failus kairėje ir spausk **Užkrauti**.")
    st.stop()

# ---------------- Render (works across reruns) ----------------

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
cB.metric("sfreq (Hz)", f"{sfreq:.1f}")
cC.metric("Valid epochų", f"{X.shape[0]}")
cD.metric("Atmesta (?/movement)", f"{dropped} / {total_events}")

st.write(f"**Įrašo trukmė (pagal valid epochas):** ~{duration_min:.0f} min")
st.write("**Stadijų % (valid epochos):** " + ", ".join([f"{k} {v:.1f}%" for k, v in sorted(share.items(), key=lambda x: x[0])]))

# Window mask
if show_full_night:
    mask = np.ones_like(y, dtype=bool)
    window_title = "Hipnograma (visa naktis)"
else:
    start_s = float(start_offset_min) * 60.0
    end_s = start_s + float(window_len_min) * 60.0
    mask = (starts_sec >= start_s) & (starts_sec < end_s)
    if not np.any(mask):
        st.warning("Pasirinktas langas neturi valid epochų. Rodau visą naktį.")
        mask = np.ones_like(y, dtype=bool)
        window_title = "Hipnograma (visa naktis)"
    else:
        window_title = f"Hipnograma (langas: {start_offset_min}–{start_offset_min + window_len_min} min)"

Xw = X[mask]
yw = y[mask]
sw = starts_sec[mask]

# keep e_idx in range
if st.session_state.e_idx >= int(Xw.shape[0]):
    st.session_state.e_idx = 0

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("Hipnograma")
    st.pyplot(plot_hypnogram_time(sw, yw, title=window_title))
with col2:
    st.subheader("Miego stadijų pasiskirstymas")
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

st.divider()

# Jump buttons (now safe)
b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    if st.button("↪ First REM", use_container_width=True):
        idx = np.where(yw == 4)[0]
        if len(idx) == 0:
            st.warning("REM šiame lange nerasta.")
        else:
            st.session_state.e_idx = int(idx[0])
with b2:
    if st.button("↪ First N3", use_container_width=True):
        idx = np.where(yw == 3)[0]
        if len(idx) == 0:
            st.warning("N3 šiame lange nerasta.")
        else:
            st.session_state.e_idx = int(idx[0])

st.subheader("Epochos peržiūra")
e_idx = st.slider(
    "Pasirink epochos indeksą (lango viduje)",
    0, int(Xw.shape[0]) - 1,
    int(st.session_state.e_idx),
    1
)
st.session_state.e_idx = int(e_idx)

label_id = int(yw[e_idx])
label_name = stage_name_from_id(label_id)
epoch_start_min = float(sw[e_idx]) / 60.0
st.write(f"**Epocha (lange) #{e_idx}** → **{label_name}** (id={label_id}) | **laikas:** ~{epoch_start_min:.1f} min")

epoch = Xw[e_idx, 0, :]

# ---------------- BiLSTM inference (TorchScript, sequence mode) ----------------
st.divider()
st.subheader("CNN+BiLSTM modelio prognozė")

if bilstm_file is None:
    st.info("Įkelk `cnn_bilstm_full_ts.pt` į sidebar'ą, kad matytum BiLSTM prognozę.")
else:
    if run_bilstm:
        try:
            T = Xw.shape[2]
            expected_T = int(epoch_sec * sfreq)
            if T != expected_T:
                st.warning(f"Įtartina: Xw T={T}, bet epoch_sec*sfreq={expected_T}.")
            if T != 3000:
                st.warning(
                    f"Modelis treniruotas su T=3000 (30s @ 100Hz), pas tave T={T}. "
                    f"Rekomendacija: epoch_sec=30 ir Resample=100Hz."
                )

            model_bilstm = load_torchscript_model_from_bytes(bilstm_file.getvalue())

            # center sequence (L, T)
            seq = make_center_sequence(Xw, e_idx, int(seq_len))
            seq = zscore_np(seq)  # kaip train_bilstm.py

            x_seq = torch.from_numpy(seq).unsqueeze(0)  # (1, L, T)

            with torch.no_grad():
                logits = model_bilstm(x_seq)  # (1, 5)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_id = int(np.argmax(probs))
            pred_name = stage_name_from_id(pred_id)

            true_id = int(label_id)
            true_name = stage_name_from_id(true_id)

            m1, m2, m3 = st.columns(3)
            m1.metric("True (center)", f"{true_name} ({true_id})")
            m2.metric("Pred", f"{pred_name} ({pred_id})")
            m3.metric("Confidence", f"{float(probs[pred_id]) * 100:.1f}%")

            st.bar_chart({STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))})

        except Exception as e:
            st.error("BiLSTM inference nepavyko. Žemiau – klaidos detalės:")
            st.exception(e)

# ---------------- Epoch signal plot ----------------
st.pyplot(plot_epoch_signal(epoch, sfreq, title=f"Epochos signalas ({label_name})"))

# ---------------- CNN inference (optional demo) ----------------
if cnn_file is not None and run_cnn:
    st.divider()
    st.subheader("CNN modelio prognozė (demo)")

    device = "cpu"
    if torch.cuda.is_available():
        use_gpu = st.sidebar.checkbox("Naudoti GPU (jei yra)", value=False)
        device = "cuda" if use_gpu else "cpu"

    try:
        # cache model per filename
        if ("cnn_model_name" not in st.session_state) or (st.session_state.cnn_model_name != cnn_file.name):
            raw_bytes = cnn_file.getvalue()

            model = None
            # TorchScript
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                    tmp.write(raw_bytes)
                    tmp.flush()
                    tmp_path = tmp.name
                model = torch.jit.load(tmp_path, map_location=device)
                model.eval()
                st.session_state.cnn_model_type = "torchscript"
            except Exception:
                model = None

            # state_dict fallback (jei turi SimpleSleepCNN)
            if model is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                    tmp.write(raw_bytes)
                    tmp.flush()
                    sd_path = tmp.name
                state = torch.load(sd_path, map_location=device)

                model = SimpleSleepCNN(n_classes=len(STAGE_NAMES)).to(device)
                model.load_state_dict(state)
                model.eval()
                st.session_state.cnn_model_type = "state_dict(SimpleSleepCNN)"

            st.session_state.cnn_model = model
            st.session_state.cnn_model_name = cnn_file.name

        model = st.session_state.cnn_model

        # epoch input (B,1,T)
        x = normalize_epoch(epoch).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)

        pred_id = int(np.argmax(probs))
        pred_name = STAGE_NAMES[pred_id]

        st.write(f"**Modelis:** `{st.session_state.cnn_model_type}` | failas: `{st.session_state.cnn_model_name}`")
        st.write(f"**True:** {label_name} | **Pred:** **{pred_name}**")
        st.bar_chart({STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))})

    except Exception as e:
        st.error("CNN modelio užkrovimas / predikcija nepavyko. Klaida:")
        st.exception(e)

st.divider()
st.subheader("Modelio prognozės (paruošta plėtrai)")
st.info("Čia vėliau prijungsi apmokytą CNN / BiLSTM / Transformer ir rodysi softmax tikimybes pasirinktai epochai.")

# ---------------- Model inference (sequence) ----------------

if model_file is None:
    st.info("Jei nori prognozės – įkelk `cnn_bilstm_full_ts.pt` (TorchScript) į sidebar'ą.")
else:
    if run_model:
        try:
            # 1) Patikrink, ar T sutampa (modelis treniruotas su 30s @ 100Hz => T=3000)
            T = Xw.shape[2]
            expected_T = int(epoch_sec * sfreq)
            if T != expected_T:
                st.warning(f"Įtartina: Xw T={T}, bet epoch_sec*sfreq={expected_T}.")
            if T != 3000:
                st.warning(f"Modelis greičiausiai treniruotas su T=3000, pas tave T={T}. "
                           f"Rekomendacija: epoch_sec=30 ir Resample=100Hz.")

            # 2) Užkraunam modelį (cache)
            model = load_torchscript_model_from_bytes(model_file.getvalue())

            # 3) Sudarom seką aplink pasirinktą epochą
            seq = make_center_sequence(Xw, e_idx, int(seq_len))      # (L, T)
            seq = zscore_np(seq)                                     # zscore per epochą (kaip train'e)

            # 4) Tensor: (B, L, T)
            x = torch.from_numpy(seq).unsqueeze(0)                   # (1, L, T)

            with torch.no_grad():
                logits = model(x)                                    # (1, 5)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_id = int(np.argmax(probs))
            pred_name = stage_name_from_id(pred_id)
            true_id = int(label_id)
            true_name = stage_name_from_id(true_id)

            st.subheader("Modelio prognozė (CNN+BiLSTM)")
            m1, m2, m3 = st.columns(3)
            m1.metric("True (center)", f"{true_name} ({true_id})")
            m2.metric("Pred", f"{pred_name} ({pred_id})")
            m3.metric("Confidence", f"{float(probs[pred_id]) * 100:.1f}%")

            # Softmax bar chart
            prob_dict = {STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))}
            st.bar_chart(prob_dict)

        except Exception as e:
            st.error("Modelio inference nepavyko. Žemiau – klaidos detalės:")
            st.exception(e)

st.pyplot(plot_epoch_signal(epoch, sfreq, title=f"Epochos signalas ({label_name})"))

st.divider()
st.subheader("CNN modelio prognozė (demo)")

model_file = st.sidebar.file_uploader(
    "Įkelk CNN modelį (TorchScript .pt/.ts arba state_dict .pth)",
    type=["pt", "ts", "pth"]
)

device = "cpu"
if torch.cuda.is_available():
    use_gpu = st.sidebar.checkbox("Naudoti GPU (jei yra)", value=False)
    device = "cuda" if use_gpu else "cpu"

if model_file is None:
    st.info("Įkelk modelį šone, kad matytum prognozes. Rekomenduojama: TorchScript (.pt/.ts).")
else:
    try:
        # 1) Užkraunam modelį į session_state (kad nereikėtų krauti per kiekvieną rerun)
        if ("model_name" not in st.session_state) or (st.session_state.model_name != model_file.name):
            raw_bytes = model_file.getvalue()

            # bandome TorchScript
            model = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                    tmp.write(raw_bytes)
                    tmp.flush()
                    tmp_path = tmp.name
                model = torch.jit.load(tmp_path, map_location=device)
                model.eval()
                st.session_state.model_type = "torchscript"
            except Exception:
                model = None

            # jei ne TorchScript, bandome state_dict su SimpleSleepCNN
            if model is None:
                state = torch.load(
                    tempfile.NamedTemporaryFile(delete=False, suffix=".pth").name,
                    map_location=device
                )
                # ^ viršuj dar neįrašėm bytes, todėl darom teisingai:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                    tmp.write(raw_bytes)
                    tmp.flush()
                    sd_path = tmp.name
                state = torch.load(sd_path, map_location=device)

                model = SimpleSleepCNN(n_classes=len(STAGE_NAMES)).to(device)
                model.load_state_dict(state)
                model.eval()
                st.session_state.model_type = "state_dict(SimpleSleepCNN)"

            st.session_state.model = model
            st.session_state.model_name = model_file.name

        model = st.session_state.model

        # 2) Paruošiam input epochą (B,1,T)
        x = normalize_epoch(epoch).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]

        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)

        pred_id = int(np.argmax(probs))
        pred_name = STAGE_NAMES[pred_id]

        st.write(f"**Modelis:** `{st.session_state.model_type}` | failas: `{st.session_state.model_name}`")
        st.write(f"**True:** {label_name} | **Pred:** **{pred_name}**")

        # 3) Softmax bar chart
        prob_dict = {STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))}
        st.bar_chart(prob_dict)

        # 4) (bonus) prognozės visam langui + accuracy
        if st.checkbox("Skaičiuoti prognozes visam langui (lėčiau)", value=False):
            Xw_norm = Xw[:, 0, :].astype(np.float32)
            # normalizuojam per epochą
            mu = Xw_norm.mean(axis=1, keepdims=True)
            sd = Xw_norm.std(axis=1, keepdims=True) + 1e-8
            Xw_norm = (Xw_norm - mu) / sd

            batch = st.slider("Batch size", 16, 256, 64, 16)
            preds = []
            with torch.no_grad():
                for i in range(0, Xw_norm.shape[0], batch):
                    xb = torch.from_numpy(Xw_norm[i:i+batch]).unsqueeze(1).to(device)  # [B,1,T]
                    pb = torch.softmax(model(xb), dim=-1).argmax(dim=-1).cpu().numpy()
                    preds.append(pb)
            y_pred = np.concatenate(preds, axis=0)

            acc = float(np.mean(y_pred == yw[:len(y_pred)]))
            st.write(f"**Window accuracy:** {acc*100:.1f}%")

    except Exception as e:
        st.error("Modelio užkrovimas / predikcija nepavyko. Klaida:")
        st.exception(e)


st.divider()
st.subheader("Modelio prognozės (paruošta plėtrai)")
st.info("Čia vėliau prijungsi apmokytą CNN / BiLSTM / Transformer ir rodysi softmax tikimybes pasirinktai epochai.")
