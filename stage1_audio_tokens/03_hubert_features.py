"""03 — HuBERT: waveform → semantic feature → discrete unit(k-means).

배우는 것 (논문: HuBERT, arXiv:2106.07447):
- SSL 인코더는 라벨 없이 대량 음성으로 학습. "내용/발음" 중심 표현(semantic)을 만든다.
- HuBERT 자체는 연속 hidden state를 뱉는다. 여기에 k-means를 씌우면 discrete unit이 된다.
  (GLM-4-Voice 등 semantic tokenizer의 기본 아이디어.)
- semantic feature는 '복원'이 목표가 아니라 '이해'가 목표 → 디코딩으로 원음이 안 돌아온다.
- 레이어마다 성격이 다르다(하위=음향, 중상위=음소/내용). 흔히 6~9번째 레이어를 쓴다.

실행: python 03_hubert_features.py
출력: outputs/03_hubert_units.npy (프레임별 discrete unit)
"""
import numpy as np
import torch

from _common import OUT_DIR, banner, load_wav, pick_device, sample_path


def main() -> None:
    banner("03 · HuBERT — semantic feature + discrete unit")
    from sklearn.cluster import KMeans
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    device = pick_device()
    name = "facebook/hubert-base-ls960"
    print(f"loading {name} ...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    model = HubertModel.from_pretrained(name).to(device).eval()

    wav, sr = load_wav(sample_path(), target_sr=16000)
    inputs = fe(wav, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: (embeddings, layer1..layer12)
    n_layers = len(hidden_states) - 1
    feat_last = out.last_hidden_state[0].cpu().numpy()  # (frames, dim)
    frames, dim = feat_last.shape
    dur = len(wav) / sr
    frame_rate = frames / dur

    print(f"hidden states: {len(hidden_states)}개 (embedding + {n_layers} layers)")
    print(f"feature shape: {feat_last.shape} = (frames={frames}, dim={dim})")
    print(f"frame rate ≈ {frame_rate:.1f}Hz  ({1000/frame_rate:.0f}ms/frame)")
    print("→ HuBERT-base는 20ms/frame = 50Hz. Stage 1의 다른 표현들과 frame rate를 비교해보라.")

    # 중간 레이어(내용 중심)로 discrete unit 만들기
    layer = 9
    feat = hidden_states[layer][0].cpu().numpy()
    n_clusters = 100
    print(f"\nk-means (layer {layer}, k={n_clusters}) → discrete unit 생성 ...")
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit(feat)
    units = km.labels_
    print(f"units shape: {units.shape}  (프레임마다 정수 하나 = discrete semantic token)")
    print(f"units 예시 (첫 30 frame): {units[:30].tolist()}")
    # 연속 반복 제거(run-length) → 실제 음소열에 가까운 짧은 시퀀스
    dedup = units[np.insert(np.diff(units) != 0, 0, True)]
    print(f"연속중복 제거 후 길이: {len(units)} → {len(dedup)} "
          f"(내용이 유지되는 구간은 같은 unit이 반복됨)")

    out_path = OUT_DIR / "03_hubert_units.npy"
    np.save(out_path, units)
    print(f"[saved] {out_path}")
    print("\n핵심: 이 unit들은 '내용'을 담지만, 이것만으로 원음(음색·화자)을 복원할 수 없다.")
    print("      → 그래서 semantic(HuBERT) vs acoustic(EnCodec)을 04에서 나란히 비교한다.")


if __name__ == "__main__":
    main()
