"""Tacotron1 서브모듈을 from-scratch로 조립.

논문(1703.10135) Table 1/2 기준. 공식 HF/torchaudio 구현이 없어 여기서 직접 만든다.
가중치는 random init이므로 forward 출력의 acoustic 의미는 없고, 스텝 스크립트는
shape·수식·불변식만 검증한다. 핵심 3블록:

- CBHG   : conv bank(width 1..K) + highway + BiGRU. 인코더와 post-net이 공유.
- Attention : content-based(Bahdanau) — decoder query와 인코더 상태의 tanh energy.
- Decoder : autoregressive, 스텝당 r개 mel 프레임 방출(reduction factor).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Prenet(nn.Module):
    """FC-ReLU-Dropout 스택. Tacotron은 추론 때도 prenet dropout을 켜둔다."""

    def __init__(self, in_dim: int, sizes=(256, 128), dropout: float = 0.5):
        super().__init__()
        dims = [in_dim] + list(sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.dropout = dropout

    def forward(self, x):
        for fc in self.layers:
            x = nn.functional.dropout(torch.relu(fc(x)), self.dropout, training=True)
        return x


class Highway(nn.Module):
    """y = H(x)·T(x) + x·(1-T(x)). T bias를 -1로 초기화해 초반엔 carry 쪽으로."""

    def __init__(self, dim: int):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.T.bias.data.fill_(-1.0)

    def forward(self, x):
        h = torch.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        return h * t + x * (1 - t)


class ConvBank(nn.Module):
    """K개의 Conv1d(width 1..K) + BN을 채널로 concat. width k = k-gram 문맥."""

    def __init__(self, in_dim: int, K: int, ch: int = 128):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(in_dim, ch, k, padding=k // 2), nn.BatchNorm1d(ch))
            for k in range(1, K + 1)
        )
        self.K, self.ch = K, ch

    def forward(self, x):  # x: (B, in_dim, T)
        T = x.size(-1)
        outs = [torch.relu(c(x))[:, :, :T] for c in self.convs]  # even-k padding 1 trim
        return torch.cat(outs, dim=1)  # (B, K*ch, T)


class CBHG(nn.Module):
    """1-D conv bank + highway + BiGRU. residual add 위해 proj[-1] == in_dim 이어야 함."""

    def __init__(self, in_dim: int, K: int = 16, ch: int = 128,
                 proj=(128, 128), n_highway: int = 4, gru: int = 128):
        super().__init__()
        assert proj[-1] == in_dim, f"CBHG residual: proj[-1]({proj[-1]}) != in_dim({in_dim})"
        self.bank = ConvBank(in_dim, K, ch)
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.proj1 = nn.Sequential(nn.Conv1d(K * ch, proj[0], 3, padding=1), nn.BatchNorm1d(proj[0]))
        self.proj2 = nn.Sequential(nn.Conv1d(proj[0], proj[1], 3, padding=1), nn.BatchNorm1d(proj[1]))
        self.highways = nn.ModuleList(Highway(in_dim) for _ in range(n_highway))
        self.gru = nn.GRU(in_dim, gru, batch_first=True, bidirectional=True)

    def forward(self, x):  # x: (B, T, in_dim)
        T = x.size(1)
        y = x.transpose(1, 2)          # (B, in_dim, T)
        y = self.bank(y)               # (B, K*ch, T)
        y = self.pool(y)[:, :, :T]     # (B, K*ch, T)
        y = torch.relu(self.proj1(y))  # (B, proj0, T)
        y = self.proj2(y)              # (B, proj1, T)
        y = y.transpose(1, 2) + x      # residual (proj1 == in_dim)
        for h in self.highways:
            y = h(y)
        y, _ = self.gru(y)             # (B, T, 2*gru)
        return y


class BahdanauAttention(nn.Module):
    """content-based tanh attention. energy = v·tanh(W·query + V·keys)."""

    def __init__(self, enc_dim: int, query_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(query_dim, attn_dim, bias=False)
        self.V = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=True)

    def forward(self, query, keys):  # query (B, query_dim), keys (B, T, enc_dim)
        e = self.v(torch.tanh(self.W(query).unsqueeze(1) + self.V(keys))).squeeze(-1)  # (B, T)
        align = torch.softmax(e, dim=-1)
        context = torch.bmm(align.unsqueeze(1), keys).squeeze(1)  # (B, enc_dim)
        return context, align


class Decoder(nn.Module):
    """autoregressive attention decoder. 스텝당 r개 mel 프레임을 방출한다."""

    def __init__(self, enc_dim: int = 256, mel: int = 80, r: int = 2,
                 prenet=(256, 128), attn_dim: int = 128, gru: int = 256):
        super().__init__()
        self.mel, self.r = mel, r
        self.prenet = Prenet(mel, prenet)
        self.attn_rnn = nn.GRUCell(prenet[-1] + enc_dim, gru)
        self.attn = BahdanauAttention(enc_dim, gru, attn_dim)
        self.dec_rnn = nn.ModuleList([nn.GRUCell(gru + enc_dim, gru), nn.GRUCell(gru, gru)])
        self.proj = nn.Linear(gru + enc_dim, mel * r)

    def forward(self, enc_out, n_steps: int):
        B = enc_out.size(0)
        prev = enc_out.new_zeros(B, self.mel)  # <GO> frame
        h_attn = enc_out.new_zeros(B, self.attn_rnn.hidden_size)
        h_dec = [enc_out.new_zeros(B, c.hidden_size) for c in self.dec_rnn]
        ctx_vec = enc_out.new_zeros(B, enc_out.size(-1))
        mels, aligns = [], []
        for _ in range(n_steps):
            p = self.prenet(prev)                                   # (B, prenet[-1])
            h_attn = self.attn_rnn(torch.cat([p, ctx_vec], -1), h_attn)
            ctx_vec, align = self.attn(h_attn, enc_out)             # (B, enc_dim), (B, T_enc)
            x = torch.cat([h_attn, ctx_vec], -1)
            h_dec[0] = self.dec_rnn[0](x, h_dec[0])
            inp = h_dec[0]
            for i in range(1, len(self.dec_rnn)):                   # vertical residual
                h_dec[i] = self.dec_rnn[i](inp, h_dec[i])
                inp = inp + h_dec[i]
            frames = self.proj(torch.cat([inp, ctx_vec], -1)).view(B, self.r, self.mel)
            mels.append(frames)
            aligns.append(align)
            prev = frames[:, -1]                                    # 마지막 프레임 되먹임
        return torch.cat(mels, 1), torch.stack(aligns, 1)           # (B, n*r, mel), (B, n, T_enc)


class PostNet(nn.Module):
    """디코더 mel 시퀀스 → linear-scale spectrogram. CBHG(K=8) + linear proj."""

    def __init__(self, mel: int = 80, K: int = 8, linear_bins: int = 513, gru: int = 128):
        super().__init__()
        self.cbhg = CBHG(mel, K=K, proj=(256, mel), gru=gru)
        self.to_linear = nn.Linear(2 * gru, linear_bins)

    def forward(self, mels):  # (B, T, mel)
        return self.to_linear(self.cbhg(mels))  # (B, T, linear_bins)
