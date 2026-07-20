"""STEP 2 — 텍스트 BPE 토큰화 + 조건 시퀀스 조립.

VoxCPM은 텍스트는 여전히 **BPE 토큰**으로(LLM과 동일), 오디오는 STEP 1의
**연속 latent patch**로 넣는다. 둘을 하나의 시퀀스로 이어 붙이되, 각 위치가
텍스트인지 오디오인지 mask로 구분한다(voice cloning: 참조 텍스트+오디오가 앞,
합성할 목표 텍스트가 그 안에 포함).

VoxCPMModel._generate_with_prompt_cache의 조립을 그대로 재현한다:

  text_token = BPE(prompt_text + target_text) + [audio_start=101]     # 길이 L_text
  audio_feat = [0]*L_text  ++  참조 latent patch(길이 L_audio)          # (L_text+L_audio, P, D)
  text_mask  = [1]*L_text  ++ [0]*L_audio      (텍스트 위치)
  audio_mask = [0]*L_text  ++ [1]*L_audio      (오디오 위치)

같은 위치에서 text_embed(텍스트) 또는 feat_embed(오디오) 중 하나만 살아남아
backbone에 들어간다(STEP 3). audio_start_token(101)이 "이제 오디오다"라는 스위치.
"""
import torch

from common import build_context, rule


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "prompt_audio_feat" not in ctx:  # STEP 1이 참조 latent을 채워야 한다
        import importlib
        ctx = importlib.import_module("01_audio_vae").main(ctx)

    tts, device, dtype = ctx["tts"], ctx["device"], ctx["dtype"]
    rule("STEP 2 · 텍스트 BPE 토큰화 + 조건 시퀀스 조립 (text ++ audio, mask로 구분)")

    prompt_text, target_text = ctx["prompt_text"], ctx["target_text"]
    prompt_audio_feat = ctx["prompt_audio_feat"]  # (L_audio, P, D)

    # voice cloning: 참조 전사 + 목표 문장을 이어 토큰화(공백 하나로 구분해 가독성 확보).
    text = prompt_text.strip() + " " + target_text
    subwords = tts.text_tokenizer.tokenize(text)     # BPE 서브워드 문자열
    text_ids = tts.text_tokenizer(text)              # 서브워드 → id

    text_token = torch.LongTensor(text_ids + [tts.audio_start_token])  # +audio_start(101)
    L_text = text_token.shape[0]
    L_audio = prompt_audio_feat.shape[0]
    P, D = tts.patch_size, tts.audio_vae.latent_dim

    # 텍스트 구간엔 0 오디오, 오디오 구간엔 0 텍스트를 채워 길이를 맞춘다
    text_token = torch.cat([text_token, torch.zeros(L_audio, dtype=torch.long)])
    audio_feat = torch.cat([torch.zeros(L_text, P, D), prompt_audio_feat], dim=0)  # (L_text+L_audio, P, D)
    text_mask = torch.cat([torch.ones(L_text), torch.zeros(L_audio)]).to(torch.int32)
    audio_mask = torch.cat([torch.zeros(L_text), torch.ones(L_audio)]).to(torch.int32)

    print(f"prompt_text    : {prompt_text[:60]!r}{'...' if len(prompt_text) > 60 else ''}")
    print(f"target_text    : {target_text!r}")
    print(f"BPE 서브워드    : {len(subwords)}개 — 예시 {[s.replace(chr(9601), '·') for s in subwords[:10]]}")
    print(f"text_ids       : {len(text_ids)}개 + audio_start({tts.audio_start_token}) → 텍스트 토큰 {L_text}개")
    print(f"참조 latent     : {L_audio} patch (STEP 1 결과)")
    print(f"조립 시퀀스     : 총 {L_text + L_audio} 위치 = 텍스트 {L_text} + 오디오 {L_audio}")
    print(f"  text_token   : {tuple(text_token.shape)}  (오디오 구간은 0 패딩)")
    print(f"  audio_feat   : {tuple(audio_feat.shape)}  (텍스트 구간은 0 패딩)")
    print(f"  text_mask    : {int(text_mask.sum())}개 1  | audio_mask: {int(audio_mask.sum())}개 1  (합 = 전체)")

    # 배치화 + device/dtype (STEP 3~4가 그대로 backbone에 넣는다)
    ctx["text_token"] = text_token.unsqueeze(0).to(device)
    ctx["text_mask"] = text_mask.unsqueeze(0).to(device)
    ctx["audio_feat_seq"] = audio_feat.unsqueeze(0).to(device).to(dtype)
    ctx["audio_mask"] = audio_mask.unsqueeze(0).to(device)
    ctx["L_text"], ctx["L_audio"] = L_text, L_audio
    ctx["subwords"] = subwords
    return ctx


if __name__ == "__main__":
    main()
