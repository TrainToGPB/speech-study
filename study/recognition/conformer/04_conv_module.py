"""STEP 4 — Convolution module (국소 시간 구조 담당).

self-attention이 전역 관계를 본다면, conv module은 인접 프레임의 국소 패턴을 잡는다.
  residual = x
  x = layer_norm(x)                          # pre-LN
  x = xᵀ                                     # (B, C, T)로 축 교환
  x = pointwise_conv1(x)   (1024 → 2048)     # 채널 확장
  x = GLU(x)               (2048 → 1024)     # a * sigmoid(b): 절반은 게이트
  x = depthwise_conv(x)    (k=31, groups=1024)  # 채널별 1D conv = ±15 frame(±300ms) 국소 수용영역
  x = batch_norm(x); x = swish(x)
  x = pointwise_conv2(x)   (1024 → 1024)
  x = xᵀ                                     # (B, T, C)로 복귀
  out = residual + x

GLU 게이트가 "어떤 채널을 통과시킬지" 스스로 조절하고, depthwise conv가 시간축 국소성을 준다.
"""
import torch

from common import build_context, rule


def main(ctx: dict | None = None) -> dict:
    ctx = ctx or build_context()
    if "after_mhsa" not in ctx:
        import importlib
        ctx = importlib.import_module("03_mhsa_relpos").main(ctx)

    layer = ctx["layer"]
    x = ctx["after_mhsa"]
    cm = layer.conv_module
    rule("STEP 4 · Convolution module (pointwise→GLU→depthwise→BN→swish→pointwise)")

    with torch.no_grad():
        residual = x
        xln = cm.layer_norm(x)
        h = xln.transpose(1, 2)                 # (1, 1024, T)
        pw1 = cm.pointwise_conv1(h)             # (1, 2048, T)
        glu = cm.glu(pw1)                        # (1, 1024, T)  = a * sigmoid(b)
        dw = cm.depthwise_conv(glu)             # (1, 1024, T)  채널별 k31 conv
        bn = cm.batch_norm(dw)
        act = cm.activation(bn)                 # swish
        pw2 = cm.pointwise_conv2(act)           # (1, 1024, T)
        conv = pw2.transpose(1, 2)              # (1, T, 1024)
        out = residual + conv

        a_half, b_half = pw1.chunk(2, dim=1)    # GLU 내부: a, gate=sigmoid(b)
        gate = torch.sigmoid(b_half)

    ctx["conv_glu_gate"] = gate
    ctx["after_conv"] = out

    k = cm.depthwise_conv.kernel_size[0]
    print(f"입력 x         : {tuple(x.shape)}")
    print(f"pointwise_conv1: {tuple(pw1.shape)}  ← 1024→2048 (GLU용 2배 확장)")
    print(f"GLU 후         : {tuple(glu.shape)}  gate 평균 {gate.mean():.3f} (0~1, 채널 통과율)")
    print(f"depthwise(k{k})  : {tuple(dw.shape)}  groups=1024, 수용영역 ±{k // 2} frame (~±{k // 2 * 20}ms)")
    print(f"BN·swish·pw2   : {tuple(conv.shape)}")
    print(f"out = res+conv : {tuple(out.shape)}  ‖·‖ {residual.norm(dim=-1).mean():.2f} → {out.norm(dim=-1).mean():.2f}")
    print(f"기여도         : ‖Δ‖/‖res‖ 평균 {((out - residual).norm(dim=-1) / residual.norm(dim=-1)).mean():.3f}")

    return ctx


if __name__ == "__main__":
    main()
