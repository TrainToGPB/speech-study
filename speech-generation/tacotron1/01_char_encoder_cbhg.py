"""STEP 1 — char embedding → encoder pre-net → CBHG (인코더).

문자열이 인코더 표현으로 바뀌는 경로를 shape로 따라간다. CBHG의 conv bank는
width 1..K를 병렬로 쌓아(uni~K-gram) 다양한 문맥을 뽑고 highway+BiGRU로 섞는다.
random init이라 값은 의미 없고 shape·구조만 검증한다. (B=1이라 BatchNorm은 eval로)
"""
import torch
import torch.nn as nn

from common import HP, VOCAB, build_context, rule
from modules import CBHG, Prenet


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    hp: HP = ctx["hp"]
    dev = ctx["device"]
    rule("STEP 1 · char → embed → pre-net → CBHG (인코더)")

    torch.manual_seed(0)
    embed = nn.Embedding(len(VOCAB), hp.embed).to(dev)
    prenet = Prenet(hp.embed, hp.prenet).to(dev).eval()
    cbhg = CBHG(hp.prenet[-1], K=hp.k_enc, gru=hp.enc_gru).to(dev).eval()

    char_ids = ctx["char_ids"].to(dev)
    with torch.no_grad():
        e = embed(char_ids)                   # (1, L, 256)
        p = prenet(e)                         # (1, L, 128)
        bank = cbhg.bank(p.transpose(1, 2))   # (1, K*128, L)
        enc_out = cbhg(p)                     # (1, L, 256)

    L = char_ids.size(1)
    ok = []

    def check(name, got, want):
        good = tuple(got) == tuple(want)
        ok.append(good)
        tail = "✅" if good else f"❌ (기대 {tuple(want)})"
        print(f"{name:15s}: {tuple(got)}  {tail}")

    print(f"입력 문자열    : {ctx['text']!r}  (L={L}자)")
    check("char embed", e.shape, (1, L, hp.embed))
    check("pre-net out", p.shape, (1, L, hp.prenet[-1]))
    check("conv bank", bank.shape, (1, hp.k_enc * hp.conv_ch, L))
    check("CBHG enc_out", enc_out.shape, (1, L, hp.enc_out))
    print(f"conv bank      : width 1..{hp.k_enc} → {hp.k_enc}개×{hp.conv_ch}ch "
          f"= {hp.k_enc * hp.conv_ch} (uni~{hp.k_enc}-gram 문맥)")
    print(f"enc_out dim    : BiGRU {hp.enc_gru}×2 = {hp.enc_out}")
    print(f"검증           : {'모두 통과 ✅' if all(ok) else '실패 ❌'}")

    ctx["embed"], ctx["prenet_out"], ctx["enc_out"] = e, p, enc_out
    return ctx


if __name__ == "__main__":
    main()
