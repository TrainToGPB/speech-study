"""VIZ — semantic-acoustic decoupling (Notion/논문 Fig 2 의 통제 실험 버전).

VoxCPM의 핵심 주장: 명시적 supervision 없이도 skeleton(TSLM+FSQ)은 내용(semantic),
residual(RALM)은 음색(acoustic)을 담아 **암묵적으로 분리**된다.

논문 Fig 2는 여러 화자의 zero-shot 복제 latent을 t-SNE로 보여준다. 공식 실습용
tiny 데이터셋(librispeech dummy)엔 화자가 1명뿐이라, 여기선 더 깔끔한 **통제 실험**
으로 같은 주장을 검증한다:

  내용(문장)을 고정하고 **목소리만** N개로 바꾼다(text-only 생성은 seed마다 다른
  목소리를 낸다). 각 목소리 오디오를 prefill해 오디오 위치의
    - skeleton hidden(enc_out, FSQ 후)   ← 내용 축
    - residual hidden(residual_lm 출력)  ← 음색 축
  을 뽑아 목소리별로 색칠한다. 내용이 고정이므로:
    skeleton은 목소리와 무관하게 겹쳐야(speaker-agnostic),
    residual은 목소리별로 갈려야(speaker-specific) 한다.

정량 검증으로 목소리 라벨에 대한 silhouette score를 두 표현에서 계산한다
(residual ≫ skeleton 이면 주장 성립). run.py 파이프라인과 독립.

    python viz_decoupling_tsne.py
"""
import numpy as np
import torch

from common import (OUT, load_model, model_dtype, rule, save_wav, savefig, seed_everything)

CONTENT = "The quick brown fox jumps over the lazy dog."  # 내용 고정
N_VOICES = 6                 # 생성할 목소리 수(같은 내용)
MAX_PATCH_PER_UTT = 60       # 발화당 최대 오디오 patch


@torch.inference_mode()
def extract(tts, device, dtype, prompt_text, audio_np, sr):
    """오디오 하나를 prefill해 오디오 위치의 (skeleton, residual) hidden을 반환."""
    vae = tts.audio_vae
    patch_len = tts.patch_size * vae.chunk_size

    wav = torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).unsqueeze(0)
    if wav.shape[1] % patch_len != 0:
        wav = torch.nn.functional.pad(wav, (patch_len - wav.shape[1] % patch_len, 0))
    z = vae.encode(wav.to(device).to(torch.float32), sr)
    prompt_feat = z.view(vae.latent_dim, -1, tts.patch_size).permute(1, 2, 0).contiguous()  # (La,P,D)

    text_ids = tts.text_tokenizer(prompt_text.strip() or "audio")
    text_token = torch.LongTensor(text_ids + [tts.audio_start_token])
    L_text, L_audio = text_token.shape[0], prompt_feat.shape[0]
    P, D = tts.patch_size, vae.latent_dim

    text_token = torch.cat([text_token, torch.zeros(L_audio, dtype=torch.long)]).unsqueeze(0).to(device)
    feat = torch.cat([torch.zeros(L_text, P, D), prompt_feat.cpu()], dim=0).unsqueeze(0).to(device).to(dtype)
    text_mask = torch.cat([torch.ones(L_text), torch.zeros(L_audio)]).unsqueeze(0).to(device)
    audio_mask = torch.cat([torch.zeros(L_text), torch.ones(L_audio)]).unsqueeze(0).to(device)
    tmask, amask = text_mask.unsqueeze(-1).to(dtype), audio_mask.unsqueeze(-1).to(dtype)

    feat_embed = tts.enc_to_lm_proj(tts.feat_encoder(feat))
    scale_emb = tts.config.lm_config.scale_emb if tts.config.lm_config.use_mup else 1.0
    combined = tmask * (tts.base_lm.embed_tokens(text_token) * scale_emb) + amask * feat_embed
    enc_raw, _ = tts.base_lm(inputs_embeds=combined, is_causal=True)
    enc_out = tts.fsq_layer(enc_raw) * amask + enc_raw * tmask
    res_raw, _ = tts.residual_lm(inputs_embeds=enc_out + amask * feat_embed, is_causal=True)

    skel = enc_out[0, L_text:].float().cpu().numpy()   # (La, 1024) skeleton
    res = res_raw[0, L_text:].float().cpu().numpy()     # (La, 1024) residual
    if skel.shape[0] > MAX_PATCH_PER_UTT:
        idx = np.linspace(0, skel.shape[0] - 1, MAX_PATCH_PER_UTT).astype(int)
        skel, res = skel[idx], res[idx]
    return skel, res


def main() -> None:
    rule("VIZ · semantic-acoustic decoupling (내용 고정 · 목소리 N개, skeleton vs residual)")
    seed_everything()
    model, tts, device = load_model()
    dtype = model_dtype(tts)

    print(f"내용 고정: {CONTENT!r}")
    print(f"목소리 {N_VOICES}개를 text-only 생성(seed별 다른 음색) 후 각각 prefill로 hidden 추출...")

    skels, resids, labels = [], [], []
    for v in range(N_VOICES):
        torch.manual_seed(1000 + v)  # 설치된 voxcpm엔 generate(seed=) 없음 → 전역 RNG로 목소리 고정
        wav = np.asarray(
            model.generate(text=CONTENT, cfg_value=2.0, inference_timesteps=10, normalize=True, denoise=False),
            dtype=np.float32,
        )
        save_wav(OUT / f"tsne_voice_{v}.wav", wav, tts.sample_rate)
        sk, rs = extract(tts, device, dtype, CONTENT, wav, tts.sample_rate)
        skels.append(sk); resids.append(rs); labels.extend([v] * sk.shape[0])
        print(f"  voice {v}: {len(wav) / tts.sample_rate:.2f}s → {sk.shape[0]} patch")

    skels = np.concatenate(skels, 0)
    resids = np.concatenate(resids, 0)
    labels = np.array(labels)
    print(f"수집 포인트: {len(labels)}개 (각 {skels.shape[1]}차원)")

    # 정량 검증: 목소리 라벨에 대한 silhouette (raw hidden 기준). residual ≫ skeleton 기대
    from sklearn.metrics import silhouette_score

    sil_skel = float(silhouette_score(skels, labels))
    sil_res = float(silhouette_score(resids, labels))
    print(f"\n[정량] 목소리 라벨 silhouette (높을수록 목소리로 잘 갈림):")
    print(f"  skeleton(TSLM+FSQ) = {sil_skel:+.3f}   (낮아야 함 = 내용 축, 목소리 무관)")
    print(f"  residual(RALM)     = {sil_res:+.3f}   (높아야 함 = 음색 축, 목소리별 분리)")
    verdict = "성립 ✔ (residual이 목소리로 더 잘 갈림)" if sil_res > sil_skel else "약함 — seed/N 조정 여지"
    print(f"  → 분리 주장: {verdict}")

    from sklearn.manifold import TSNE

    def tsne(x):
        perp = max(5, min(30, (len(x) - 1) // 3))
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(x)

    emb_skel, emb_res = tsne(skels), tsne(resids)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab10")
        cols = [cmap(int(l) % 10) for l in labels]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.4))
        ax[0].scatter(emb_skel[:, 0], emb_skel[:, 1], c=cols, s=8, alpha=0.6)
        ax[0].set(title=f"TSLM+FSQ skeleton  (silhouette {sil_skel:+.2f})\n내용 축 → 목소리끼리 겹침")
        ax[1].scatter(emb_res[:, 0], emb_res[:, 1], c=cols, s=8, alpha=0.6)
        ax[1].set(title=f"RALM residual  (silhouette {sil_res:+.2f})\n음색 축 → 목소리별 군집")
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(i % 10), label=f"voice {i}") for i in range(N_VOICES)]
        fig.legend(handles=handles, loc="lower center", ncol=N_VOICES, fontsize=8)
        fig.suptitle("semantic-acoustic decoupling (통제 실험): 같은 내용, 다른 목소리")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        savefig(fig, "viz_decoupling_tsne.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 시각화 생략: {e}")


if __name__ == "__main__":
    main()
