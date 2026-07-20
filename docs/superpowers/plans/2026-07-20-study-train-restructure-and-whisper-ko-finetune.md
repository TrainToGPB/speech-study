# study/train 재구조화 + Whisper 한국어 미니 파인튜닝 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 레포를 `study/`(논문 해부) + `train/`(학습 실무) 3-depth 구조로 재편하고, 첫 `train/` 실험으로 Whisper-base를 한국어에 full fine-tune 해 CER이 실제로 떨어지는 걸 M5/MPS에서 확인한다.

**Architecture:** (A) 세 `speech-*` 토픽 폴더를 `study/<topic>`로 이동(`speech-` 제거), 모든 실험을 `<section>/<topic>/<name>` 3-depth로 통일, 그에 맞춰 `parents[3]`·`build_index.py`·`new_experiment.py`·문서를 갱신. (B) `train/recognition/whisper-finetune-ko`를 기존 step-by-step 컨벤션으로 스캐폴드하고 `01_data→02_baseline→03_train→04_evaluate`로 전개.

**Tech Stack:** Python 3.12, PyTorch(MPS), HuggingFace `transformers==4.46.3` / `datasets>=3.0,<4` / `evaluate` / `jiwer` / `accelerate`, `soundfile`. Whisper-base, Zeroth-Korean.

## Global Constraints

- 모든 실험 경로는 **`<section>/<topic>/<name>`** 3-depth. 실험 코드에서 repo root = `pathlib.Path(__file__).resolve().parents[3]`.
- 디렉터리 이동은 **git 히스토리 보존**(rename 감지). `outputs/` PNG는 물리 보존, `.venv/`는 **삭제**(이동 시 절대경로 깨짐, git 제외·재현 비엄격 → `setup_env.py`로 재생성).
- `INDEX.md`는 **자동 생성물** — 손으로 편집 금지, 항상 `build_index.py`로 재생성.
- `docs/superpowers/plans|specs/`의 **기존 dated 문서는 수정하지 않는다**(시점 기록).
- `train/` 실험 의존성: **`datasets>=3.0,<4`**(Zeroth FLAC를 soundfile로 디코딩 → torchcodec 회피), **`transformers==4.46.3`**.
- MPS 학습 인자: `fp16=False`, `bf16=False`, `dataloader_pin_memory=False`, `dataloader_num_workers=0`, `remove_unused_columns=False`, 환경변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- 지표: **CER**(주), WER(참고). 커밋 메시지는 한국어·토픽 prefix, 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- 작업 브랜치: `bard-channel/curriculum-restructure-papers`. push/merge는 사용자 요청 시에만.

---

## 파일 구조 (이 계획이 만드는/고치는 것)

**Phase 1 (재구조화)**
- 이동: `speech-generation|recognition|representation/*` → `study/<topic>/*`
- 수정: 5× `common.py`/`run.py`(`parents[2]→[3]`), `shared/env.py`(도크스트링), `.claude/skills/speech-study/scripts/build_index.py`, `.../new_experiment.py`, `.../assets/LOG_template.md`, `.../SKILL.md`, `README.md`, 5× `LOG.md`, `study/recognition/conformer/01_frontend.py`
- 재생성: `INDEX.md`

**Phase 2 (실험)** — `train/recognition/whisper-finetune-ko/`
- `common.py`(로더·collator·CER), `download.py`, `01_data.py`, `02_baseline.py`, `03_train.py`, `04_evaluate.py`, `run.py`, `requirements.txt`, `LOG.md`, `outputs/`(git 제외)

---

# Phase 1 — 재구조화

### Task 1: 토픽 폴더를 `study/`로 이동 (히스토리 보존, venv 삭제)

**Files:**
- Move: `speech-generation` → `study/generation`, `speech-recognition` → `study/recognition`, `speech-representation` → `study/representation`
- Delete: 이동된 4× `.venv/`

- [ ] **Step 1: 이동 전 상태 스냅샷**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git ls-files speech-generation speech-recognition speech-representation | wc -l   # 65 예상
ls speech-recognition/whisper/outputs | wc -l                                     # >0 (PNG 보존 대상)
```
Expected: 65개 tracked, whisper outputs에 PNG 존재.

- [ ] **Step 2: 폴더째 이동(artifact 포함) 후 venv 삭제**

Run:
```bash
mkdir -p study
mv speech-generation     study/generation
mv speech-recognition    study/recognition
mv speech-representation study/representation
find study -type d -name .venv -prune -exec rm -rf {} +
git add -A
```

- [ ] **Step 3: rename·보존·정리 검증**

Run:
```bash
git status --short | grep -E '^R' | wc -l          # rename 다수
ls study/recognition/whisper/outputs | wc -l        # PNG 여전히 존재(>0)
find study -name .venv | wc -l                       # 0 (삭제됨)
ls speech-generation 2>/dev/null; echo "old gen gone? $?"   # 없음
git log --follow --oneline -- study/recognition/whisper/LOG.md | head -3  # 과거 커밋 보임
```
Expected: rename으로 잡힘, outputs 보존, `.venv` 0개, 옛 폴더 사라짐, `--follow`로 히스토리 연결.

- [ ] **Step 4: Commit**

```bash
git commit -q -m "$(cat <<'EOF'
레포 구조: speech-* 토픽을 study/ 하위로 이동 (speech- prefix 제거)

- speech-generation/recognition/representation → study/generation/recognition/representation
- git rename으로 히스토리 보존, outputs PNG 보존, 실험별 .venv 삭제(재생성 대상)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: import 깊이 `parents[2]` → `parents[3]`

이동으로 실험이 3-depth가 됐으므로 repo root 계산을 한 단계 올린다.

**Files (Modify):**
- `study/generation/tacotron1/common.py`
- `study/generation/voxcpm/run.py`
- `study/recognition/conformer/common.py`
- `study/recognition/whisper/common.py`
- `study/representation/wav2vec2/common.py`
- `shared/env.py` (도크스트링 예시)

**Interfaces:**
- Produces: 각 실험에서 `from shared.env import get_device`가 다시 동작.

- [ ] **Step 1: 다섯 실험 파일 일괄 치환**

각 파일에서 정확히 한 줄:
```
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
```
→
```
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
```

Run(일괄):
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
grep -rl "resolve().parents\[2\]" study | while read f; do
  sed -i '' 's/resolve()\.parents\[2\]/resolve().parents[3]/' "$f"
done
grep -rn "parents\[" study | grep -v __pycache__   # 전부 [3] 확인
```
Expected: `study/**` 안의 `parents[...]`가 모두 `[3]`.

- [ ] **Step 2: `shared/env.py` 도크스트링 갱신**

`shared/env.py`의 도크스트링 예시 라인 `sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))` → `parents[3]`. (실제 함수 로직엔 `parents[]`가 없으니 도크스트링만.)

- [ ] **Step 3: 각 실험에서 import 성공 검증** (venv 불필요 — 순수 경로 계산)

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
for f in study/generation/tacotron1/common.py study/recognition/whisper/common.py study/representation/wav2vec2/common.py; do
  d=$(dirname "$f")
  python3 -c "import pathlib,sys; sys.path.append(str(pathlib.Path('$f').resolve().parents[3])); import importlib; importlib.import_module('shared.env'); print('OK $f ->', pathlib.Path('$f').resolve().parents[3].name)"
done
```
Expected: 세 줄 모두 `OK ... -> sanddollar`(repo root 폴더명).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -q -m "$(cat <<'EOF'
레포 구조: 3-depth 이동에 맞춰 repo-root 경로 parents[2]→[3]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `build_index.py`를 3-depth·section 인지로 재작성

**Files:**
- Modify(전체 교체): `.claude/skills/speech-study/scripts/build_index.py`

**Interfaces:**
- Produces: `INDEX.md`에 `구분`(section) 컬럼, `<section>/<topic>/<name>` 스캔, `train` 섹션은 `🛠 실습` 유형.

- [ ] **Step 1: 파일 전체를 아래로 교체**

```python
#!/usr/bin/env python3
"""LOG.md 프론트매터를 스캔해 루트 INDEX.md를 재생성한다.

표준 라이브러리만 사용 — 시스템 파이썬으로 실행 가능.
실험은 항상 <section>/<topic>/<name>/LOG.md 세 단계 깊이에 있다고 가정한다.
section=최상위(study/train 등), topic=그 하위, name=실험 폴더.
"""
import subprocess
import sys
from pathlib import Path

# 실험 섹션이 아닌 최상위 디렉토리
IGNORE_TOPLEVEL = {".git", ".claude", ".omc", "docs", "shared", ".github", "outputs"}

STATUS_EMOJI = {
    "planned": "🌱 planned",
    "wip": "🚧 wip",
    "done": "✅ done",
    "parked": "🧊 parked",
}


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return Path(out.decode().strip())
    except Exception:
        return Path.cwd()


def _parse_val(val: str):
    val = val.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        return [x.strip() for x in inner.split(",") if x.strip()] if inner else []
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    return val


def parse_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    data: dict = {}
    parent = None
    for raw in block.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, val = raw.strip().partition(":")
        key = key.strip()
        if indent == 0:
            if val.strip() == "":
                data[key] = {}
                parent = key
            else:
                data[key] = _parse_val(val)
                parent = None
        elif parent is not None:
            data[parent][key] = _parse_val(val)
    return data


def _kind(section: str, fm: dict) -> str:
    if section == "train":
        return "🛠 실습"
    return "💡 아이디어" if "idea" in fm else "📄 논문"


def collect(root: Path) -> list[dict]:
    rows = []
    for section_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if section_dir.name in IGNORE_TOPLEVEL or section_dir.name.startswith("."):
            continue
        for topic_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            for exp_dir in sorted(p for p in topic_dir.iterdir() if p.is_dir()):
                log = exp_dir / "LOG.md"
                if not log.exists():
                    continue
                fm = parse_frontmatter(log.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "section": section_dir.name,
                        "topic": fm.get("topic", topic_dir.name),
                        "name": exp_dir.name,
                        "rel": log.relative_to(root).as_posix(),
                        "kind": _kind(section_dir.name, fm),
                        "status": STATUS_EMOJI.get(
                            str(fm.get("status", "planned")), fm.get("status", "")
                        ),
                        "env": ", ".join(fm.get("env_tested", []) or []) or "–",
                        "updated": fm.get("updated", ""),
                    }
                )
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        "# 실험 인덱스",
        "",
        "> `build_index.py`가 각 실험의 `LOG.md` 프론트매터를 스캔해 자동 생성합니다.",
        "> 직접 편집하지 말고 로그를 고친 뒤 스킬의 `index` 명령을 쓰세요.",
        "",
        f"총 {len(rows)}개 실험.",
        "",
        "| 구분 | 주제 | 실험 | 유형 | 상태 | 검증환경 | 갱신일 |",
        "|------|------|------|------|------|----------|--------|",
    ]
    for r in sorted(rows, key=lambda x: (x["section"], x["topic"], x["name"])):
        lines.append(
            f"| {r['section']} | {r['topic']} | [{r['name']}]({r['rel']}) | {r['kind']} | "
            f"{r['status']} | {r['env']} | {r['updated']} |"
        )
    if not rows:
        lines.append("| _아직 없음_ | | | | | | |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    rows = collect(root)
    (root / "INDEX.md").write_text(render(rows), encoding="utf-8")
    print(f"INDEX.md 갱신: {len(rows)}개 실험")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 실행 후 INDEX 검증** (아직 train 실험 없음 → study 5개)

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
python3 .claude/skills/speech-study/scripts/build_index.py
cat INDEX.md
```
Expected: `총 5개 실험`, 모든 행 `구분=study`, 주제=`generation|recognition|representation`(prefix 없음), 링크 경로 `study/.../LOG.md`.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/speech-study/scripts/build_index.py INDEX.md
git commit -q -m "$(cat <<'EOF'
스킬: build_index를 <section>/<topic>/<name> 3-depth로 재작성

- section(study/train) 컬럼 추가, train은 🛠 실습 유형
- INDEX 재생성

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `new_experiment.py` + `LOG_template.md`를 3-part 경로로

**Files:**
- Modify: `.claude/skills/speech-study/scripts/new_experiment.py`
- Modify: `.claude/skills/speech-study/assets/LOG_template.md`

**Interfaces:**
- Consumes: `build_index.py`(Task 3) — 스캐폴드 후 자동 호출.
- Produces: `new_experiment.py <section>/<topic>/<name>`로 실험 생성, RUN_STUB이 `parents[3]` 사용.

- [ ] **Step 1: `RUN_STUB`의 depth 수정**

`new_experiment.py`에서:
```
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
```
→
```
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
```

- [ ] **Step 2: 경로 파싱을 3-part로 (main 내부 교체)**

기존 블록:
```python
    parts = args.path.strip("/").split("/")
    if len(parts) != 2:
        print("경로는 <topic>/<name> 두 단계여야 합니다.", file=sys.stderr)
        return 1
    topic, name = parts

    root = repo_root()
    exp = root / topic / name
```
→
```python
    parts = args.path.strip("/").split("/")
    if len(parts) != 3:
        print("경로는 <section>/<topic>/<name> 세 단계여야 합니다 "
              "(예: train/recognition/whisper-finetune-ko).", file=sys.stderr)
        return 1
    section, topic, name = parts

    root = repo_root()
    exp = root / section / topic / name
```

- [ ] **Step 3: 템플릿 치환·안내문에 `PATH`/`SECTION` 반영**

`log = (...)` 치환 체인에서 `.replace("{{TITLE}}", title)` 다음 줄에 두 줄 추가하고 이후 라인은 유지:
```python
        .replace("{{TITLE}}", title)
        .replace("{{PATH}}", f"{section}/{topic}/{name}")
        .replace("{{SECTION}}", section)
        .replace("{{TOPIC}}", topic)
```
그리고 "다음 단계" 출력 두 줄을 교체:
```python
    print(f"  1. {section}/{topic}/{name}/LOG.md 의 🎯 목표를 채우기")
    print(f"  2. cd {section}/{topic}/{name} && python {SKILL_DIR}/scripts/setup_env.py home  # 또는 work")
```

- [ ] **Step 4: `LOG_template.md`의 cd 경로를 `{{PATH}}`로**

`assets/LOG_template.md`에서:
```
cd {{TOPIC}}/{{NAME}}
```
→
```
cd {{PATH}}
```
(`topic: {{TOPIC}}` 프론트매터 줄은 그대로 둔다.)

- [ ] **Step 5: dry 검증(폴더 미생성 확인용) + 실인자 오류 처리 확인**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
python3 .claude/skills/speech-study/scripts/new_experiment.py bad/two 2>&1 | head -1   # 3단계 아님 → 오류 메시지
python3 - <<'PY'
import pathlib
t = pathlib.Path(".claude/skills/speech-study/assets/LOG_template.md").read_text()
assert "cd {{PATH}}" in t, "template cd not updated"
s = pathlib.Path(".claude/skills/speech-study/scripts/new_experiment.py").read_text()
assert "parents[3]" in s and "len(parts) != 3" in s
print("template/script OK")
PY
```
Expected: 2-part 경로는 "세 단계여야" 오류; `template/script OK`.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/speech-study/scripts/new_experiment.py .claude/skills/speech-study/assets/LOG_template.md
git commit -q -m "$(cat <<'EOF'
스킬: new_experiment/LOG_template을 <section>/<topic>/<name> 3-part로

- 경로 파싱 3단계, RUN_STUB parents[3], 템플릿 cd {{PATH}}

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: LOG 프론트매터·본문·wiki-link 경로 갱신

**Files (Modify):**
- `study/generation/tacotron1/LOG.md`, `study/generation/voxcpm/LOG.md`
- `study/recognition/conformer/LOG.md`, `study/recognition/whisper/LOG.md`
- `study/representation/wav2vec2/LOG.md`
- `study/recognition/conformer/01_frontend.py` (도크스트링 wiki-link)

- [ ] **Step 1: `topic:` 프론트매터에서 `speech-` 제거**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
sed -i '' 's/^topic: speech-generation$/topic: generation/'     study/generation/*/LOG.md
sed -i '' 's/^topic: speech-recognition$/topic: recognition/'   study/recognition/*/LOG.md
sed -i '' 's/^topic: speech-representation$/topic: representation/' study/representation/*/LOG.md
```

- [ ] **Step 2: 본문 `cd` 경로에 `study/` 붙이고 prefix 제거**

Run:
```bash
sed -i '' 's#cd speech-generation/#cd study/generation/#'   study/generation/*/LOG.md
sed -i '' 's#cd speech-recognition/#cd study/recognition/#' study/recognition/*/LOG.md
```
(wav2vec2 LOG의 `cd speech-representation/...`도 처리:)
```bash
sed -i '' 's#cd speech-representation/#cd study/representation/#' study/representation/*/LOG.md
```

- [ ] **Step 3: wiki-link `[[speech-<topic>/...]]` → `[[study/<topic>/...]]`**

Run(LOG + 스텝 스크립트 도크스트링 모두):
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
grep -rl "\[\[speech-" study | while read f; do
  sed -i '' \
    -e 's#\[\[speech-generation/#[[study/generation/#g' \
    -e 's#\[\[speech-recognition/#[[study/recognition/#g' \
    -e 's#\[\[speech-representation/#[[study/representation/#g' "$f"
done
grep -rn "\[\[speech-" study | grep -v __pycache__ ; echo "남은 옛 링크? (위 비어야 정상)"
```
Expected: 남은 `[[speech-` 없음.

- [ ] **Step 4: 잔여 하드코딩 경로 스캔**

Run:
```bash
grep -rn "speech-generation\|speech-recognition\|speech-representation" study shared README.md .claude 2>/dev/null | grep -v __pycache__
echo "위 결과 비어야 정상(docs/ 기존 계획서는 의도적으로 제외)"
```
Expected: 비어 있음.

- [ ] **Step 5: INDEX 재생성 후 topic 확인**

Run:
```bash
python3 .claude/skills/speech-study/scripts/build_index.py
grep -E "generation|recognition|representation" INDEX.md
```
Expected: 5행, 주제가 prefix 없는 이름으로.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -q -m "$(cat <<'EOF'
레포 구조: LOG 프론트매터/본문/wiki-link 경로를 study/ 3-depth로 갱신

- topic: speech-* prefix 제거, cd 경로 study/ 반영, [[speech-*]] → [[study/*]]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `SKILL.md` + `README.md` 재작성

**Files (Modify):** `.claude/skills/speech-study/SKILL.md`, `README.md`

- [ ] **Step 1: `SKILL.md` 컨벤션·명령 예시 갱신**

`## 저장소 컨벤션` 트리 블록을 아래로 교체:
```
speech-study/
├── INDEX.md                       # 실험 레지스트리 (build_index.py가 자동 생성)
├── shared/env.py                  # device 자동감지 헬퍼 (cuda→mps→cpu)
├── study/                         # 논문·메커니즘 해부
│   └── <topic>/<paper-or-idea>/   # 실험 단위
│       ├── LOG.md                 # 프론트매터 + 본문 (INDEX의 단일 소스)
│       ├── requirements.txt       # torch 제외 의존성
│       ├── download.py            # 대용량 자산 다운로드 (자산은 git 제외)
│       ├── run.py                 # 실험 코드
│       ├── .venv/  · outputs/     # git 제외
└── train/                         # 학습·전처리 실무 실습
    └── <topic>/<name>/            # 같은 골격
```
`- **조직 단위 ...**` 줄 아래에 한 줄 추가: `- **최상위 구분**: \`study/\`(해부) · \`train/\`(학습 실무). 경로는 항상 \`<section>/<topic>/<name>\` 3단계.`
그리고 `### 1) new` 예시 커맨드를 3-part로:
```bash
python .claude/skills/speech-study/scripts/new_experiment.py <section>/<topic>/<name> \
  --title "<사람이 읽는 제목>" \
  [--idea] [--paper-link <url>] [--notion-link <url>]
```
예시 줄: `- \`<section>/<topic>/<name>\` 예: \`train/recognition/whisper-finetune-ko\`, \`study/audio-tokenization/encodec\`.`
`### 2) setup`의 `cd <topic>/<name>` → `cd <section>/<topic>/<name>`.
로그 템플릿 문단의 키 목록에 `topic`은 유지(변경 없음).

- [ ] **Step 2: `README.md` 구조·예시 갱신**

`## 구조`의 코드블록을 아래로 교체:
```
study/<topic>/<paper-or-idea>/   # 논문·메커니즘 해부 (예: recognition/whisper)
train/<topic>/<name>/            # 학습·전처리 실무 실습 (예: recognition/whisper-finetune-ko)
├── LOG.md                 # 목표·실행법·결과·막힌 점 (INDEX의 단일 소스)
├── requirements.txt       # torch 제외 의존성
├── download.py            # 대용량 자산 다운로드 (자산 자체는 git 제외)
├── run.py                 # 실험 코드
├── .venv/                 # 실험 전용 (git 제외)
└── outputs/               # 결과물·가중치 (git 제외)

shared/env.py              # device 자동감지 (cuda→mps→cpu)
```
`## 워크플로`의 스크립트 예시 3곳 경로를 3-part로:
```bash
# 새 실험
python .claude/skills/speech-study/scripts/new_experiment.py \
    train/recognition/whisper-finetune-ko --title "Whisper 한국어 파인튜닝"

# 환경 셋업 (실험 폴더 안에서)
cd train/recognition/whisper-finetune-ko
python ../../../.claude/skills/speech-study/scripts/setup_env.py work
```
(setup 예시의 상대경로가 `../../../`로 한 단계 깊어짐에 유의.)

- [ ] **Step 3: 링크/경로 정합성 확인**

Run:
```bash
grep -n "study/\|train/" README.md .claude/skills/speech-study/SKILL.md | head
grep -n "speech-generation\|speech-recognition\|speech-representation" README.md .claude/skills/speech-study/SKILL.md ; echo "옛 경로 남았나? (비어야 정상)"
```
Expected: study/·train/ 예시 존재, 옛 `speech-<topic>` 없음.

- [ ] **Step 4: Commit**

```bash
git add README.md .claude/skills/speech-study/SKILL.md
git commit -q -m "$(cat <<'EOF'
문서: SKILL/README를 study·train 3-depth 구조로 갱신

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

# Phase 2 — train/recognition/whisper-finetune-ko

### Task 7: 실험 스캐폴드 + 의존성 + LOG 목표

**Files:**
- Create(스캐폴드): `train/recognition/whisper-finetune-ko/{LOG.md,requirements.txt,download.py,run.py,outputs/}`
- Modify: 위 `requirements.txt`, `LOG.md`

- [ ] **Step 1: 스캐폴드 실행** (Task 4의 새 `new_experiment.py` 사용 — 실검증 겸함)

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
python3 .claude/skills/speech-study/scripts/new_experiment.py \
  train/recognition/whisper-finetune-ko \
  --title "Whisper 한국어 미니 파인튜닝 (CER 전/후, M5·MPS)" --idea
ls train/recognition/whisper-finetune-ko
```
Expected: `LOG.md requirements.txt download.py run.py outputs/` 생성, "INDEX.md 갱신: 6개 실험" 출력.

- [ ] **Step 2: `requirements.txt` 교체**

파일 전체:
```
# torch/torchaudio는 setup_env.py가 환경(work=MPS)에 맞게 설치합니다.
transformers==4.46.3
datasets>=3.0,<4      # Zeroth FLAC를 soundfile로 디코딩(torchcodec 회피). datasets 4는 torchcodec 요구.
evaluate
jiwer                 # CER/WER 백엔드
accelerate            # HF Trainer 필수
soundfile             # FLAC 디코딩
```

- [ ] **Step 3: `LOG.md` 프론트매터·목표·실행법 작성**

`LOG.md`에서 프론트매터 `topic: recognition` 확인(스캐폴드가 채움), 아래 본문으로 🎯/⚙️ 섹션 교체:
```markdown
## 🎯 목표
GPU 서버 없이 **M5(32GB·MPS)** 에서 기존 multilingual `openai/whisper-base`를
한국어(**Zeroth-Korean** 소량)에 full fine-tune 해서, **CER이 실제로 떨어지는지**를
전/후로 확인한다. 학습 자체가 목적이 아니라 "데이터 파이프라인 → 학습 루프 → 지표 개선"의
한 사이클을 손으로 도는 게 목표. 지표는 한국어 특성상 WER보다 **CER**을 주로 본다.

## ⚙️ 실행 방법
```bash
cd train/recognition/whisper-finetune-ko
python ../../../.claude/skills/speech-study/scripts/setup_env.py work
python download.py       # whisper-base + Zeroth subset + CER metric 캐시
python 01_data.py        # 전처리 한 샘플 해부 (mel·label)
python 02_baseline.py    # 파인튜닝 전 CER
python 03_train.py       # full fine-tune (MPS, max_steps 소량)
python 04_evaluate.py    # 파인튜닝 후 CER + 전/후 비교
# run.py 로 01→04 일괄 실행도 가능
```
모델 `openai/whisper-base`, 데이터 `Bingsu/zeroth-korean`. MPS 안전 설정(fp32,
num_workers 0)은 `common.py`에 모음.
```

- [ ] **Step 4: Commit** (venv·outputs는 git 제외)

```bash
git add train/recognition/whisper-finetune-ko/LOG.md \
        train/recognition/whisper-finetune-ko/requirements.txt \
        train/recognition/whisper-finetune-ko/download.py \
        train/recognition/whisper-finetune-ko/run.py \
        INDEX.md
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: 실험 스캐폴드 + 의존성 + 목표 (planned)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: 환경 셋업 (venv 생성·설치·device 검증)

**Files:** `train/recognition/whisper-finetune-ko/.venv/`(git 제외 — 커밋 없음)

- [ ] **Step 1: setup 실행**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
python3 ../../../.claude/skills/speech-study/scripts/setup_env.py work
```
Expected: `.venv` 생성, torch/transformers/datasets/... 설치, 마지막 검증에 `mps available True`.

- [ ] **Step 2: 핵심 import 스모크**

Run:
```bash
.venv/bin/python -c "import transformers, datasets, evaluate, jiwer, soundfile, accelerate; print('deps OK', transformers.__version__, datasets.__version__)"
```
Expected: `deps OK 4.46.3 3.x.y`. (커밋 없음 — venv는 git 제외.)

---

### Task 9: `common.py` — 로더·collator·CER

**Files:** Create `train/recognition/whisper-finetune-ko/common.py`

**Interfaces (Produces):**
- `get_device() -> str`(shared 재수출), `MODEL_ID`, `LANG`, `TASK`, `DATASET_ID`, `OUT`, `CKPT_DIR`
- `load_processor()`, `load_model(from_dir=None) -> (model, device)`, `load_splits(n_train, n_eval) -> (ds_train, ds_eval)`
- `make_prepare(processor) -> callable`, `DataCollatorSpeechSeq2SeqWithPadding`, `load_cer()`, `transcribe(model, processor, device, example) -> str`, `evaluate_cer(model, processor, device, ds) -> (cer, pairs)`

- [ ] **Step 1: 파일 작성**

```python
"""Whisper 한국어 파인튜닝 공통 유틸 (M5·MPS 안전 설정 포함).

모델 openai/whisper-base 를 Zeroth-Korean 소량에 full fine-tune 한다.
MPS에서 안전하도록: fp32, MPS fallback 허용, tokenizers 병렬 off.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # 미지원 op는 CPU 폴백
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from shared.env import get_device  # noqa: E402

MODEL_ID = "openai/whisper-base"     # multilingual (한국어 약함 → 개선폭 큼)
DATASET_ID = "Bingsu/zeroth-korean"  # FLAC, CC-BY-4.0
LANG = "korean"
TASK = "transcribe"
SEED = 0

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
CKPT_DIR = OUT / "whisper-base-ko"


def seed_everything() -> None:
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_processor():
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(MODEL_ID, language=LANG, task=TASK)


def load_model(from_dir: str | None = None):
    """(model, device). from_dir 지정 시 파인튜닝 체크포인트에서 로드."""
    from transformers import WhisperForConditionalGeneration

    device = get_device()
    src = from_dir or MODEL_ID
    model = WhisperForConditionalGeneration.from_pretrained(src).to(device)
    model.generation_config.language = LANG
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model, device


def load_splits(n_train: int = 1000, n_eval: int = 200):
    """Zeroth train/test 를 소량 subset 으로. 16kHz mono 보장."""
    from datasets import Audio, load_dataset

    ds_train = load_dataset(DATASET_ID, split="train")
    ds_eval = load_dataset(DATASET_ID, split="test")
    ds_train = ds_train.shuffle(seed=SEED).select(range(min(n_train, len(ds_train))))
    ds_eval = ds_eval.select(range(min(n_eval, len(ds_eval))))
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_eval = ds_eval.cast_column("audio", Audio(sampling_rate=16000))
    return ds_train, ds_eval


def make_prepare(processor):
    """example → {input_features, labels}. datasets.map 용."""

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """input_features 와 labels 를 각각 패딩. label 패딩은 -100 마스크."""

    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # collator가 앞에 BOS를 다시 붙이므로 있으면 하나 제거
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_cer():
    import evaluate

    return evaluate.load("cer")


def transcribe(model, processor, device, example) -> str:
    """단일 example → 예측 문자열 (한국어 강제)."""
    import torch

    audio = example["audio"]
    feats = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feats, language=LANG, task=TASK, max_new_tokens=225)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def evaluate_cer(model, processor, device, ds):
    """(cer, pairs) — pairs는 (정답, 예측) 리스트. 02/04가 공유한다."""
    cer = load_cer()
    refs, hyps = [], []
    for ex in ds:
        refs.append(ex["text"])
        hyps.append(transcribe(model, processor, device, ex))
    return cer.compute(predictions=hyps, references=refs), list(zip(refs, hyps))


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
```

- [ ] **Step 2: import + 컬럼 스모크** (Zeroth 컬럼명이 `audio`/`text`인지 실확인)

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python -c "
import common
from datasets import load_dataset
d = load_dataset(common.DATASET_ID, split='test')
print('columns:', d.column_names)
assert 'audio' in d.column_names and 'text' in d.column_names, d.column_names
print('common OK')
"
```
Expected: `columns: [...'audio'...'text'...]`, `common OK`.
> 컬럼명이 다르면(`sentence` 등) `make_prepare`/`transcribe`/`load_splits`의 `batch["text"]`를 실제 이름으로 바꾸고 재검증.

- [ ] **Step 3: Commit**

```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git add train/recognition/whisper-finetune-ko/common.py
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: common.py (로더·collator·CER, MPS 안전 설정)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `download.py` — 자산 warm-up

**Files:** Modify `train/recognition/whisper-finetune-ko/download.py`

- [ ] **Step 1: 파일 교체**

```python
"""whisper-base 가중치 + Zeroth-Korean subset + CER metric 을 HF 캐시에 미리 받는다.

실제 파일은 ~/.cache/huggingface(공유 캐시)에 저장된다.
"""
from common import DATASET_ID, MODEL_ID, load_cer


def main() -> None:
    print(f"1) 모델 캐시: {MODEL_ID}")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained(MODEL_ID)
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    print(f"2) 데이터 캐시: {DATASET_ID} (train/test)")
    from datasets import load_dataset

    dtr = load_dataset(DATASET_ID, split="train")
    dte = load_dataset(DATASET_ID, split="test")
    print(f"   train {len(dtr)} · test {len(dte)} · columns {dtr.column_names}")
    print(f"   예시 전사: {dte[0].get('text', dte[0])!r}"[:100])

    print("3) CER metric 캐시")
    load_cer()
    print("완료. 캐시 위치: ~/.cache/huggingface")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python download.py
```
Expected: 모델·데이터 다운로드, `train N · test M · columns [...]`, 예시 전사 한국어 출력, `완료`.

- [ ] **Step 3: Commit**

```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git add train/recognition/whisper-finetune-ko/download.py
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: download.py (모델·Zeroth·CER 캐시)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `01_data.py` — 전처리 한 샘플 해부

**Files:** Create `train/recognition/whisper-finetune-ko/01_data.py`

- [ ] **Step 1: 파일 작성**

```python
"""STEP 1 — 한 샘플로 전처리 전 과정을 눈으로 본다.

raw waveform → 16k → log-Mel (1,80,3000) → (텍스트) tokenizer label ids → decode 복원.
LLM tokenization과의 대응: 오디오는 feature_extractor가, 텍스트는 tokenizer가 담당.
한국어는 띄어쓰기 편차가 커서 WER이 흔들린다 → 문자 단위 CER을 주 지표로 쓴다.
"""
import numpy as np

from common import LANG, TASK, load_processor, load_splits, rule, seed_everything


def main() -> None:
    seed_everything()
    processor = load_processor()
    _, ds_eval = load_splits(n_train=1, n_eval=8)
    ex = ds_eval[0]
    audio, text = ex["audio"], ex["text"]

    rule("STEP 1a · raw waveform")
    arr = np.asarray(audio["array"], dtype=np.float32)
    print(f"sr={audio['sampling_rate']}Hz  len={len(arr)}  "
          f"dur={len(arr)/audio['sampling_rate']:.2f}s  전사={text!r}")

    rule("STEP 1b · log-Mel feature")
    feat = processor.feature_extractor(
        arr, sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features
    print(f"input_features shape={tuple(feat.shape)}  (1, 80 mel, 3000 frame=30s·100Hz)")
    print(f"값 범위 min={feat.min():.2f} max={feat.max():.2f}")

    rule("STEP 1c · 텍스트 → label ids → 복원")
    ids = processor.tokenizer(text).input_ids
    back = processor.tokenizer.decode(ids)
    print(f"label ids({len(ids)})[:12] = {ids[:12]}")
    print(f"decode 복원 = {back!r}")
    print(f"special 제외 = {processor.tokenizer.decode(ids, skip_special_tokens=True)!r}")

    rule("메모")
    print(f"- 학습 시 디코더 입력은 '<|startoftranscript|><|{LANG[:2]}|><|{TASK}|>...' prefix로 시작")
    print("- collator가 label의 앞 BOS를 하나 잘라내고 -100으로 패딩 마스킹")
    print("- 지표: CER(주)·WER(참고) — 한국어는 CER이 표준")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 검증**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python 01_data.py
```
Expected: `input_features shape=(1, 80, 3000)`, label ids 출력, decode 복원이 원문과 일치(special 제외).

- [ ] **Step 3: Commit**

```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git add train/recognition/whisper-finetune-ko/01_data.py
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: 01_data 전처리 한 샘플 해부(mel·label·CER 설명)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: `02_baseline.py` — 파인튜닝 전 CER

**Files:** Create `train/recognition/whisper-finetune-ko/02_baseline.py`

**Interfaces (Consumes):** `common.evaluate_cer` (Task 9).

- [ ] **Step 1: 파일 작성**

```python
"""STEP 2 — 사전학습 whisper-base가 한국어 eval에서 얼마나 틀리는지(baseline CER)."""
from common import evaluate_cer, load_model, load_processor, load_splits, rule, seed_everything


def main() -> None:
    seed_everything()
    processor = load_processor()
    model, device = load_model()
    _, ds_eval = load_splits(n_train=1, n_eval=100)

    rule(f"STEP 2 · baseline CER (whisper-base, n={len(ds_eval)}, device={device})")
    score, pairs = evaluate_cer(model, processor, device, ds_eval)
    print(f"baseline CER = {score:.4f}")
    rule("예시 (정답 → 예측)")
    for ref, hyp in pairs[:5]:
        print(f"  정답: {ref}")
        print(f"  예측: {hyp}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 검증**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python 02_baseline.py
```
Expected: `baseline CER = 0.xxxx`(base는 한국어 약함 → 상당히 높을 수 있음), (정답→예측) 5쌍 출력. (MPS에서 100개 생성 수 분 소요 가능.)

- [ ] **Step 3: Commit**

```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git add train/recognition/whisper-finetune-ko/02_baseline.py
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: 02_baseline 파인튜닝 전 CER 측정

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: `03_train.py` — full fine-tune (MPS-safe)

**Files:** Create `train/recognition/whisper-finetune-ko/03_train.py`

**Interfaces (Consumes):** `common.make_prepare`, `DataCollatorSpeechSeq2SeqWithPadding`, `load_model`, `load_processor`, `load_splits`, `CKPT_DIR`.

- [ ] **Step 1: 파일 작성**

```python
"""STEP 3 — whisper-base를 Zeroth-Korean 소량에 full fine-tune (M5·MPS).

평가는 학습 루프 밖(02/04)에서 수동으로 — MPS에서 Trainer 내부 generate 리스크를
피하고 루프를 단순·견고하게 유지한다. 저장은 끝에서 1회.
"""
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from common import (
    CKPT_DIR,
    DataCollatorSpeechSeq2SeqWithPadding,
    load_model,
    load_processor,
    load_splits,
    make_prepare,
    rule,
    seed_everything,
)

N_TRAIN = 1000
MAX_STEPS = 200
BATCH = 8


def main() -> None:
    seed_everything()
    processor = load_processor()
    model, device = load_model()
    model.config.use_cache = False  # 학습 시 캐시 off

    ds_train, _ = load_splits(n_train=N_TRAIN, n_eval=1)
    prepare = make_prepare(processor)
    ds_train = ds_train.map(prepare, remove_columns=ds_train.column_names, num_proc=1)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    rule(f"STEP 3 · train (n={len(ds_train)}, steps={MAX_STEPS}, batch={BATCH}, device={device})")
    args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=20,
        max_steps=MAX_STEPS,
        fp16=False,
        bf16=False,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=ds_train,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(CKPT_DIR))
    processor.save_pretrained(str(CKPT_DIR))
    print(f"\n저장 완료: {CKPT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 검증** (본 학습 — 수 분~수십 분)

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python 03_train.py
ls outputs/whisper-base-ko
```
Expected: loss가 로그로 출력되며 감소 경향, `저장 완료`, `outputs/whisper-base-ko/`에 `model.safetensors`·`config.json`·processor 파일들.
> OOM/과속도 시: `BATCH=4` 또는 `MAX_STEPS` 축소. 그래도 막히면 `common.MODEL_ID="openai/whisper-tiny"`로 폴백하고 LOG 막힌 점에 기록.

- [ ] **Step 3: Commit** (코드만 — outputs는 git 제외)

```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
git add train/recognition/whisper-finetune-ko/03_train.py
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: 03_train full fine-tune (MPS-safe, 끝에 저장)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: `04_evaluate.py` + `run.py` + LOG 결과 기록

**Files:**
- Create `train/recognition/whisper-finetune-ko/04_evaluate.py`
- Modify `train/recognition/whisper-finetune-ko/run.py`
- Modify `train/recognition/whisper-finetune-ko/LOG.md` (결과·status·env)

**Interfaces (Consumes):** `common.evaluate_cer`, `common.load_model(from_dir=str(CKPT_DIR))`.

- [ ] **Step 1: `04_evaluate.py` 작성 (capstone)**

```python
"""STEP 4 — 파인튜닝 후 CER, 그리고 전/후 비교 (capstone).

같은 eval subset에서 base와 fine-tuned의 CER을 각각 재고, (정답/base/ft) triple을 본다.
"""
from common import (
    CKPT_DIR,
    evaluate_cer,
    load_model,
    load_processor,
    load_splits,
    rule,
    seed_everything,
)


def main() -> None:
    seed_everything()
    _, ds_eval = load_splits(n_train=1, n_eval=100)

    base_proc = load_processor()
    base_model, device = load_model()
    ft_proc = load_processor()
    ft_model, _ = load_model(from_dir=str(CKPT_DIR))

    rule(f"STEP 4 · CER 전/후 (n={len(ds_eval)}, device={device})")
    base_cer, base_pairs = evaluate_cer(base_model, base_proc, device, ds_eval)
    ft_cer, ft_pairs = evaluate_cer(ft_model, ft_proc, device, ds_eval)
    delta = base_cer - ft_cer
    print(f"baseline CER   = {base_cer:.4f}")
    print(f"fine-tuned CER = {ft_cer:.4f}")
    print(f"개선(Δ)        = {delta:+.4f}  ({'좋아짐' if delta > 0 else '나빠짐'})")

    rule("예시 (정답 / base / fine-tuned)")
    for (ref, b_hyp), (_, f_hyp) in list(zip(base_pairs, ft_pairs))[:5]:
        print(f"  정답      : {ref}")
        print(f"  base      : {b_hyp}")
        print(f"  finetuned : {f_hyp}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `04_evaluate.py` 실행 검증**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar/train/recognition/whisper-finetune-ko
.venv/bin/python 04_evaluate.py
```
Expected: `fine-tuned CER < baseline CER`(Δ 양수), (정답/base/finetuned) 5 triple에서 한국어 전사가 눈에 띄게 개선.
> Δ가 0 이하이면: `03_train`의 `MAX_STEPS`↑(예 400)·`learning_rate` 조정 후 재학습, 또는 `N_TRAIN`↑. LOG 막힌 점에 기록.

- [ ] **Step 3: `run.py` 오케스트레이션으로 교체**

```python
"""01→02→03→04 일괄 실행. 개별 스텝도 각각 단독 실행 가능하다.

주의: 03(학습)·02·04(생성 평가)는 MPS에서 수 분~수십 분 걸린다.
"""
import runpy


def main() -> None:
    for step in ("01_data", "02_baseline", "03_train", "04_evaluate"):
        print(f"\n########## {step} ##########")
        runpy.run_path(f"{step}.py", run_name="__main__")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `LOG.md`에 결과 기록 + 프론트매터 갱신**

`LOG.md` 프론트매터 `status: planned`→`wip`, `env_tested: []`→`[work]`, `updated:`를 오늘(2026-07-20)로. `## 📊 결과 / 관찰` 섹션에 실제 수치로 한 문단 추가(예):
```markdown
## 📊 결과 / 관찰
2026-07-20 · work(M5, MPS)에서 01~04 완주. whisper-base + Zeroth-Korean
train {N_TRAIN}·eval 100, max_steps {MAX_STEPS}, fp32.
- baseline CER = <02 값> → fine-tuned CER = <04 값> (Δ +<개선>).
- 예시 triple에서 <관찰: 조사/받침/숫자 등 어디가 좋아졌는지 한 줄>.
- 학습 시간 <분> (batch {BATCH}, MPS).
```
`## 🧱 막힌 점 / TODO`에 실제 겪은 이슈(있으면)와 다음 시도(steps↑, small 승급 등) 기록.

- [ ] **Step 5: INDEX 재생성 + 최종 검증**

Run:
```bash
cd /Users/bard/orca/workspaces/speech-study/sanddollar
python3 .claude/skills/speech-study/scripts/build_index.py
grep "whisper-finetune-ko" INDEX.md    # 구분=train, 유형=🛠 실습, 상태=🚧 wip, 검증환경=work
```
Expected: `총 6개 실험`, whisper-finetune-ko 행이 train/🛠 실습/wip/work.

- [ ] **Step 6: Commit**

```bash
git add train/recognition/whisper-finetune-ko/04_evaluate.py \
        train/recognition/whisper-finetune-ko/run.py \
        train/recognition/whisper-finetune-ko/LOG.md INDEX.md
git commit -q -m "$(cat <<'EOF'
whisper-finetune-ko: 04_evaluate(CER 전/후)+run 오케스트레이션+결과 기록(wip)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## 완료 기준

- `study/`(5) + `train/`(1) 6개 실험이 3-depth로 정리되고 `INDEX.md`가 구분/유형까지 정확.
- 이동한 실험에서 `from shared.env import get_device`가 동작(`parents[3]`), 스킬 스크립트가 3-part 경로로 작동.
- `whisper-finetune-ko`가 M5/MPS에서 01~04 완주하고 **fine-tuned CER < baseline CER**.
- 옛 `speech-<topic>` 경로 참조가 (docs/ 기존 계획서 제외) 코드·문서에 남지 않음.
