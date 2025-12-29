import numpy as np

# Parametrai
FS = 100          # 100 Hz
EPOCH_SEC = 30    # 30 sekundžių
T = FS * EPOCH_SEC
N_EPOCHS = 200    # kiek dirbtinių epochų generuosim

CLASSES = ["W", "N1", "N2", "N3", "REM"]  # tik info, indeksai 0..4

def generate_epoch(stage_idx: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sugeneruoja dirbtinę EEG epochą konkrečiai stadijai.
    Čia nieko moksliško – tiesiog 'panašūs' pattern'ai,
    kad būtų įdomiau negu visiškas random.
    """
    t = np.linspace(0, EPOCH_SEC, T, endpoint=False)

    # bazinis triukšmas
    x = 0.3 * rng.standard_normal(T)

    if stage_idx == 0:
        # W: daugiau greitesnių svyravimų (alpha / beta feeling)
        x += 0.5 * np.sin(2 * np.pi * 10 * t)  # ~10 Hz
    elif stage_idx == 1:
        # N1: šiek tiek lėtesnis, mažesnė amplitudė
        x += 0.3 * np.sin(2 * np.pi * 6 * t)
    elif stage_idx == 2:
        # N2: pridėkim "miego verpstes" imitaciją (~12 Hz burst'ai)
        spindle = (np.sin(2 * np.pi * 12 * t) *
                   np.exp(-((t - 10) ** 2) / 4.0))
        x += 0.4 * spindle
    elif stage_idx == 3:
        # N3: lėtos bangos (delta)
        x += 0.8 * np.sin(2 * np.pi * 1.0 * t)
    elif stage_idx == 4:
        # REM: panašu į budrumą, bet su random "akių judesių" impulsais
        x += 0.4 * np.sin(2 * np.pi * 8 * t)
        # keli impulsai
        for center in [5, 15, 25]:
            x += 0.8 * np.exp(-((t - center) ** 2) / 0.02)

    return x.astype(np.float32)


def main():
    rng = np.random.default_rng(42)

    # Pasidarom dirbtinį hipnogramą (seką)
    # pvz.: W -> N1 -> N2 -> N3 -> N2 -> REM -> W -> ...
    pattern = [0, 1, 2, 3, 2, 4]
    y = []
    X = []

    for i in range(N_EPOCHS):
        stage_idx = pattern[i % len(pattern)]
        epoch = generate_epoch(stage_idx, rng)
        X.append(epoch)
        y.append(stage_idx)

    X = np.stack(X, axis=0)          # (N, 3000)
    X = X[:, None, :]                # (N, 1, 3000)
    y = np.array(y, dtype=np.int64)  # (N,)

    print("X shape:", X.shape, "y shape:", y.shape)
    print("Class counts:", np.bincount(y, minlength=len(CLASSES)))

    # Išsaugom kaip Sleep-EDF stiliaus npz
    out_name = "dummy_sleep.npz"
    np.savez(out_name, X=X, y=y)
    print(f"Saugoma į {out_name}")


if __name__ == "__main__":
    main()
