# Tacotron1 Hands-On Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conformer-style, step-by-step hands-on for Tacotron1 — from-scratch CBHG / attention-decoder / post-net with tensor-shape tracing, a real Griffin-Lim audio demo, and a pretrained end-to-end synth.

**Architecture:** `common.py` builds a shared `ctx` (device, config, toy text, real audio) once; `modules.py` holds from-scratch `nn.Module`s; numbered `0N_*.py` step scripts each `main(ctx)` and chain via `importlib`; `run.py` runs them in order. No official HF Tacotron1 exists, so correctness is checked by paper-spec shape/invariant `assert`s (not bit-exact), plus audible output for Griffin-Lim (04) and pretrained synth (05).

**Tech Stack:** PyTorch 2.13 + torchaudio 2.11 (work venv, Python 3.12, MPS), matplotlib, soundfile, numpy; Coqui TTS (`coqui-tts`) for the pretrained step only.

## Global Constraints

- Device always via `shared.env.get_device()` (cuda→mps→cpu); code runs on home(CUDA) and work(MPS).
- No bit-exact HF reference — verify by paper-spec shape/invariant `assert` (✅/❌ prints), conformer-style.
- Core steps (Tasks 1–6) deps: `torch, torchaudio, matplotlib, soundfile, numpy` only.
- Pretrained step (Task 7) dep `coqui-tts` installed via `download.py`, NOT `requirements.txt` (its torch pin can break the core venv); lazy-import, graceful failure logged to LOG.
- Follow conformer scaffolding: `build_context()`/ctx-chaining, `rule()`, `savefig()`, `use_korean_font()`; Korean 음슴체 commentary in prints.
- Paper hyperparams (verbatim): char embed `256`; enc pre-net `FC-256→FC-128, dropout 0.5`; enc CBHG `K=16`, conv `128`ch, maxpool `width 2 stride 1`, proj `3-128 ReLU → 3-128 linear`, highway `4×128`, BiGRU `128` (→out 256); content-based Bahdanau attention; decoder pre-net `FC-256→FC-128`; attention RNN `1×GRU 256`; decoder RNN `2×residual GRU 256`; mel `80`; `r=2`; post-net CBHG `K=8`, proj `3-256→3-80`; L1 loss (mel+linear, equal weight); Griffin-Lim `50 iter`, magnitude power `1.2`.
- 04 Griffin-Lim uses the sample's native sr (16 kHz), no resample to 24 kHz.

---

### Task 1: `common.py` — shared ctx, config, utils

**Files:**
- Create: `speech-generation/tacotron1/common.py`

**Interfaces:**
- Produces: `HP` (dataclass: `embed=256, k_enc=16, k_post=8, mel=80, r=2, gru=256, prenet=(256,128), attn_dim=128, n_fft=1024, hop=256, win=1024`), `get_device()`, `seed_everything()`, `rule(title)`, `use_korean_font()`, `savefig(fig,name)`, `save_wav(wav,sr,name)`, `char_encode(text)->LongTensor (1,L)`, `VOCAB`, `load_sample()->(audio np.float32, sr, text)`, `build_context()->dict{device,hp,text,char_ids,audio,sr,ref_text}`, `OUT`.
- Reuse conformer's `load_sample` (librispeech_asr_dummy, `Audio(decode=False)`+soundfile, synthetic-sine fallback).

- [ ] **Step 1: Write `common.py`** with `HP` dataclass, `sys.path` insert to import `shared.env.get_device`, `VOCAB` (`" abcdefghijklmnopqrstuvwxyz.,!?'-"` + pad idx 0), `char_encode`, `load_sample` (copied from conformer), `save_wav` (via `torchaudio.save`), `build_context` returning the ctx dict, `rule/use_korean_font/savefig` (copied from conformer), and `use_korean_font()` at import.

- [ ] **Step 2: Verify import + context**

Run: `.venv/bin/python -c "import common; c=common.build_context(); print(c['device'], c['hp'], c['char_ids'].shape, len(c['audio'])/c['sr'])"`
Expected: prints device (`mps`), HP values, `torch.Size([1, L])`, audio seconds. No traceback.

- [ ] **Step 3: Commit** — `git add speech-generation/tacotron1/common.py && git commit -m "tacotron1: 공통 ctx·config·유틸 (common.py)"`

---

### Task 2: `modules.py` CBHG + Pre-net, and `01_char_encoder_cbhg.py`

**Files:**
- Create: `speech-generation/tacotron1/modules.py`
- Create: `speech-generation/tacotron1/01_char_encoder_cbhg.py`

**Interfaces:**
- Produces: `Prenet(in_dim, sizes=(256,128), dropout=0.5)`; `CBHG(in_dim, K, proj_dims, out_gru=128)` with `forward(x:(B,T,in_dim))->(B,T,2*out_gru)`; sub-parts `ConvBank`, `Highway`. `main(ctx)` sets `ctx["enc_out"]:(B,L,256)`, `ctx["embed"], ctx["prenet_out"]`.

- [ ] **Step 1: Implement `modules.py` Prenet + CBHG** (real code):

```python
import torch
import torch.nn as nn


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=(256, 128), dropout=0.5):
        super().__init__()
        dims = [in_dim] + list(sizes)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.dropout = dropout

    def forward(self, x):
        for fc in self.layers:
            x = nn.functional.dropout(torch.relu(fc(x)), self.dropout, training=True)
        return x  # dropout always on (Tacotron keeps prenet dropout at inference)


class Highway(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        self.T.bias.data.fill_(-1.0)  # bias gate toward carry at init

    def forward(self, x):
        h = torch.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        return h * t + x * (1 - t)


class ConvBank(nn.Module):
    """K parallel Conv1d (width 1..K) + BN, concatenated on channel."""
    def __init__(self, in_dim, K, ch=128):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(in_dim, ch, k, padding=k // 2), nn.BatchNorm1d(ch))
            for k in range(1, K + 1)
        )
        self.K, self.ch = K, ch

    def forward(self, x):  # x: (B, in_dim, T)
        T = x.size(-1)
        outs = [torch.relu(c(x))[:, :, :T] for c in self.convs]  # trim even-k padding
        return torch.cat(outs, dim=1)  # (B, K*ch, T)


class CBHG(nn.Module):
    def __init__(self, in_dim, K=16, ch=128, proj=(128, 128), n_highway=4, gru=128):
        super().__init__()
        self.bank = ConvBank(in_dim, K, ch)
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.proj1 = nn.Sequential(nn.Conv1d(K * ch, proj[0], 3, padding=1),
                                   nn.BatchNorm1d(proj[0]))
        self.proj2 = nn.Sequential(nn.Conv1d(proj[0], proj[1], 3, padding=1),
                                   nn.BatchNorm1d(proj[1]))
        self.pre_highway = nn.Linear(proj[1], in_dim) if proj[1] != in_dim else None
        self.highways = nn.ModuleList(Highway(in_dim) for _ in range(n_highway))
        self.gru = nn.GRU(in_dim, gru, batch_first=True, bidirectional=True)

    def forward(self, x):  # x: (B, T, in_dim)
        T = x.size(1)
        y = x.transpose(1, 2)               # (B, in_dim, T)
        y = self.bank(y)                    # (B, K*ch, T)
        y = self.pool(y)[:, :, :T]          # (B, K*ch, T)
        y = torch.relu(self.proj1(y))       # (B, proj0, T)
        y = self.proj2(y)                   # (B, proj1, T)
        y = y.transpose(1, 2) + x           # residual (needs proj1 == in_dim)
        if self.pre_highway is not None:
            y = self.pre_highway(y)
        for h in self.highways:
            y = h(y)
        y, _ = self.gru(y)                  # (B, T, 2*gru)
        return y
```

- [ ] **Step 2: Implement `01_char_encoder_cbhg.py`** — `main(ctx)`: build `nn.Embedding(len(VOCAB), 256)`, `Prenet(256,(256,128))`, `CBHG(128, K=16)`; run on `ctx["char_ids"]`; print + assert each stage shape:
  - `embed (1,L,256)`, `prenet_out (1,L,128)`, convbank channels `16*128=2048`, `enc_out (1,L,256)`.
  - Store `ctx["enc_out"]`. `rule("STEP 1 · char→embed→prenet→CBHG (encoder)")`. Asserts print `✅/❌`.

- [ ] **Step 3: Run** — `.venv/bin/python 01_char_encoder_cbhg.py`
Expected: shape lines + all asserts ✅ (`enc_out` last dim = 256).

- [ ] **Step 4: Commit** — `git add modules.py 01_char_encoder_cbhg.py && git commit -m "tacotron1: CBHG 인코더 + 01 shape 추적"`

---

### Task 3: Attention + Decoder in `modules.py`, and `02_attention_decoder.py`

**Files:**
- Modify: `speech-generation/tacotron1/modules.py` (append)
- Create: `speech-generation/tacotron1/02_attention_decoder.py`

**Interfaces:**
- Produces: `BahdanauAttention(enc_dim, query_dim, attn_dim)` → `forward(query,(keys))->(context:(B,enc_dim), align:(B,T_enc))`; `Decoder(enc_dim=256, mel=80, r=2, prenet=(256,128), attn_dim=128, gru=256)` → `forward(enc_out, n_steps)->(mels:(B, n_steps*r, 80), aligns:(B, n_steps, T_enc))`. `main(ctx)` sets `ctx["mels"], ctx["aligns"]`.

- [ ] **Step 1: Append attention + decoder** (real code):

```python
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, query_dim, attn_dim=128):
        super().__init__()
        self.W = nn.Linear(query_dim, attn_dim, bias=False)
        self.V = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=True)

    def forward(self, query, keys):  # query (B,query_dim), keys (B,T,enc_dim)
        e = self.v(torch.tanh(self.W(query).unsqueeze(1) + self.V(keys))).squeeze(-1)
        align = torch.softmax(e, dim=-1)            # (B, T)
        context = torch.bmm(align.unsqueeze(1), keys).squeeze(1)  # (B, enc_dim)
        return context, align


class Decoder(nn.Module):
    def __init__(self, enc_dim=256, mel=80, r=2, prenet=(256, 128), attn_dim=128, gru=256):
        super().__init__()
        self.mel, self.r = mel, r
        self.prenet = Prenet(mel, prenet)
        self.attn_rnn = nn.GRUCell(prenet[-1] + enc_dim, gru)
        self.attn = BahdanauAttention(enc_dim, gru, attn_dim)
        self.dec_rnn = nn.ModuleList([nn.GRUCell(gru + enc_dim, gru), nn.GRUCell(gru, gru)])
        self.proj = nn.Linear(gru + enc_dim, mel * r)

    def forward(self, enc_out, n_steps):
        B = enc_out.size(0)
        go = enc_out.new_zeros(B, self.mel)
        h_attn = enc_out.new_zeros(B, self.attn_rnn.hidden_size)
        h_dec = [enc_out.new_zeros(B, c.hidden_size) for c in self.dec_rnn]
        ctx_vec = enc_out.new_zeros(B, enc_out.size(-1))
        mels, aligns, prev = [], [], go
        for _ in range(n_steps):
            p = self.prenet(prev)
            h_attn = self.attn_rnn(torch.cat([p, ctx_vec], -1), h_attn)
            ctx_vec, align = self.attn(h_attn, enc_out)
            x = torch.cat([h_attn, ctx_vec], -1)
            for i, cell in enumerate(self.dec_rnn):
                h_dec[i] = cell(x if i == 0 else h_dec[i - 1], h_dec[i]) + (0 if i == 0 else h_dec[i])
                x = h_dec[i]
            frames = self.proj(torch.cat([h_dec[-1], ctx_vec], -1))  # (B, mel*r)
            mels.append(frames.view(B, self.r, self.mel))
            aligns.append(align)
            prev = frames.view(B, self.r, self.mel)[:, -1]  # feed last of r frames
        return torch.cat(mels, 1), torch.stack(aligns, 1)
```

- [ ] **Step 2: Implement `02_attention_decoder.py`** — `main(ctx)`: needs `ctx["enc_out"]` (chain to `01` via importlib if absent). Choose `n_steps = ceil(T_mel_toy / r)` for a toy `T_mel_toy=50`. Run `Decoder`; print + assert:
  - `mels (1, n_steps*r, 80)`, `aligns (1, n_steps, T_enc)`, `align.sum(-1)≈1` (✅), `n_steps == ceil(50/2)=25` (✅).
  - Save alignment heatmap `outputs/alignment_randinit.png` (note in caption: random init → diffuse). Store `ctx["mels"], ctx["aligns"]`.

- [ ] **Step 3: Run** — `.venv/bin/python 02_attention_decoder.py`
Expected: shape lines, `행합=1.000 ✅`, `steps=25 ✅`, png saved.

- [ ] **Step 4: Commit** — `git add modules.py 02_attention_decoder.py && git commit -m "tacotron1: content-based attention 디코더 + r-frame 방출 (02)"`

---

### Task 4: PostNet in `modules.py`, and `03_postnet_and_loss.py`

**Files:**
- Modify: `speech-generation/tacotron1/modules.py` (append `PostNet`)
- Create: `speech-generation/tacotron1/03_postnet_and_loss.py`

**Interfaces:**
- Produces: `PostNet(mel=80, K=8, linear_bins=n_fft//2+1)` wrapping a `CBHG(mel, K=8, proj=(256,80))` + `nn.Linear(2*128, linear_bins)` → `forward(mels:(B,T,80))->(B,T,linear_bins)`. `main(ctx)` sets `ctx["linear"], ctx["loss"]`.

- [ ] **Step 1: Append `PostNet`** (real code):

```python
class PostNet(nn.Module):
    def __init__(self, mel=80, K=8, linear_bins=513):
        super().__init__()
        self.cbhg = CBHG(mel, K=K, proj=(256, mel), gru=128)
        self.to_linear = nn.Linear(2 * 128, linear_bins)

    def forward(self, mels):           # (B, T, mel)
        return self.to_linear(self.cbhg(mels))  # (B, T, linear_bins)
```

- [ ] **Step 2: Implement `03_postnet_and_loss.py`** — `main(ctx)`: chain to `02`; `linear_bins = hp.n_fft//2+1` (513 for n_fft=1024). Run PostNet on `ctx["mels"]`; make toy targets same shapes; `L1 = l1(mels,tgt_mel) + l1(linear,tgt_lin)`; print + assert `linear (1,T,513)`, `loss` is finite scalar. Store `ctx["linear"], ctx["loss"]`.

- [ ] **Step 3: Run** — `.venv/bin/python 03_postnet_and_loss.py`
Expected: `linear (1,T,513) ✅`, `L1 loss = <finite>`.

- [ ] **Step 4: Commit** — `git add modules.py 03_postnet_and_loss.py && git commit -m "tacotron1: post-net(mel→linear) + L1 손실 (03)"`

---

### Task 5: `04_griffinlim_demo.py` — real audio reconstruction

**Files:**
- Create: `speech-generation/tacotron1/04_griffinlim_demo.py`

**Interfaces:**
- Consumes: `ctx["audio"], ctx["sr"], hp.n_fft/hop/win`.
- Produces: `outputs/original.wav`, `outputs/griffinlim.wav`, `outputs/griffinlim_spec.png`; prints per-iter spectral convergence.

- [ ] **Step 1: Implement `04_griffinlim_demo.py`** — `main(ctx)`:
  - `spec = torchaudio.transforms.Spectrogram(n_fft, win, hop, power=1.0)(audio_tensor)` (magnitude).
  - For `n_iter in (10,30,60)`: `gl = torchaudio.transforms.GriffinLim(n_fft, 32, win, hop, power=1.0, n_iter=n_iter)`; `rec = gl(spec)`; compute **spectral convergence** `‖ |STFT(rec)| - spec ‖_F / ‖spec‖_F`; print (lower = better, should drop with more iters).
  - Save `original.wav` (native sr) and `griffinlim.wav` (60-iter). Save mag-spectrogram compare png (orig vs 60-iter reconstruction). Use `save_wav`, `savefig`.
  - Print 음슴체 commentary: magnitude만 알고 위상 추정으로 복원 → iter↑ 수렴.

- [ ] **Step 2: Run** — `.venv/bin/python 04_griffinlim_demo.py`
Expected: 3 spectral-convergence lines (decreasing), `outputs/original.wav`+`griffinlim.wav`+png saved. Listen: intelligible.

- [ ] **Step 3: Commit** — `git add 04_griffinlim_demo.py && git commit -m "tacotron1: Griffin-Lim 실제 오디오 복원 데모 (04)"`

---

### Task 6: `run.py` wiring + `requirements.txt`

**Files:**
- Modify: `speech-generation/tacotron1/run.py` (replace stub)
- Modify: `speech-generation/tacotron1/requirements.txt`

**Interfaces:**
- Consumes: step `main(ctx)` chain. Produces: `python run.py` runs 01→04 (05 only with `--pretrained`).

- [ ] **Step 1: Replace `run.py`** — `build_context()` once, then `for name in ["01_char_encoder_cbhg","02_attention_decoder","03_postnet_and_loss","04_griffinlim_demo"]: ctx = importlib.import_module(name).main(ctx)`; if `--pretrained` in argv (and import ok) append `05_pretrained_synth`. `rule()` header/footer like conformer.

- [ ] **Step 2: Update `requirements.txt`** — core only: `matplotlib`, `soundfile`, `numpy` (drop the unused `transformers` pin — Tacotron core needs none). Comment that `coqui-tts` is installed separately by `download.py`.

- [ ] **Step 3: Run full core** — `.venv/bin/python run.py`
Expected: 01→04 all run, asserts ✅, `outputs/` has pngs + wavs.

- [ ] **Step 4: Commit** — `git add run.py requirements.txt && git commit -m "tacotron1: run.py 01~04 관통 + requirements 정리"`

---

### Task 7: `download.py` + `05_pretrained_synth.py` (pretrained, work-primary)

**Files:**
- Modify: `speech-generation/tacotron1/download.py` (replace stub)
- Create: `speech-generation/tacotron1/05_pretrained_synth.py`

**Interfaces:**
- Consumes: nothing from ctx (text arg). Produces: `outputs/tts_<model>.wav` or a clean skip log.

- [ ] **Step 1: `download.py`** — `pip install coqui-tts` (via `subprocess`, into current venv) + preload a model; print guidance. Idempotent; wrapped in try/except with a clear message.

- [ ] **Step 2: `05_pretrained_synth.py`** — lazy `from TTS.api import TTS`; try model list in order `["tts_models/en/ljspeech/tacotron-DCA", "tts_models/en/ljspeech/tacotron2-DDC"]`, first that loads wins; synth a fixed sentence → `outputs/tts_<tag>.wav`; log which model + vocoder used; on `ImportError`/download failure print `[skip] coqui-tts 없음 → download.py 실행` and append note to LOG 막힌점. Never raise into `run.py`.

- [ ] **Step 3: Run (work only)** — `.venv/bin/python download.py && .venv/bin/python 05_pretrained_synth.py`
Expected: either `outputs/tts_*.wav` saved (model+vocoder logged) OR a clean skip message. No crash.

- [ ] **Step 4: Commit** — `git add download.py 05_pretrained_synth.py && git commit -m "tacotron1: Coqui 사전학습 end-to-end 합성 (05) + download.py"`

---

### Task 8: LOG.md results + INDEX

**Files:**
- Modify: `speech-generation/tacotron1/LOG.md` (📊 결과, status→wip)
- Regenerate: `INDEX.md` (via skill `build_index.py`)

- [ ] **Step 1:** Fill LOG 📊 결과 with observed shapes / GL convergence numbers / which pretrained model ran (or skipped); set `status: wip`; note any 막힌 점 (e.g. Coqui failure, MPS quirks).
- [ ] **Step 2:** `python3 .claude/skills/speech-study/scripts/build_index.py` to sync INDEX.
- [ ] **Step 3: Commit** — `git add speech-generation/tacotron1/LOG.md INDEX.md && git commit -m "tacotron1: 실행 결과 기록 + INDEX 갱신"`

---

## Self-Review

- **Spec coverage:** LOG plan items map — modules.py+CBHG→T2, attention/decoder/r→T3, postnet/L1→T4, Griffin-Lim real audio→T5, run wiring→T6, Coqui pretrained+download split→T7, LOG/INDEX→T8, common/config→T1. ✅
- **Placeholder scan:** module code is concrete; step scripts specify exact asserts, shapes, run commands, expected output. Print/viz boilerplate is described by its asserts (repo-native verification), not left as "TODO". ✅
- **Type consistency:** `CBHG(in_dim,K,proj,gru)` reused by encoder (proj=(128,128)) and PostNet (proj=(256,80)); `enc_out` last dim `2*gru=256` consumed by `Decoder(enc_dim=256)` and `BahdanauAttention(enc_dim=256)`; `linear_bins=n_fft//2+1=513` consistent T4↔T5. ✅
- **Known risk to watch during exec:** CBHG residual requires `proj[-1]==in_dim`; encoder ok (128==128), PostNet uses `proj=(256,80)` with `in_dim=80` (80==80) ✅. Decoder residual GRU wiring is a faithful reconstruction, not a canonical checkpoint — verify shapes, not weights.
