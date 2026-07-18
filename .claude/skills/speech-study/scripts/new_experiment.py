#!/usr/bin/env python3
"""새 실험 폴더를 표준 골격으로 스캐폴딩하고 INDEX를 갱신한다.

사용법:
  new_experiment.py <topic>/<name> --title "제목" [--idea]
                    [--paper-link URL] [--notion-link URL]

예:
  new_experiment.py audio-tokenization/encodec --title "EnCodec 실습" \
      --paper-link https://arxiv.org/abs/2210.13438
  new_experiment.py sandbox/streaming-vad --title "스트리밍 VAD 아이디어" --idea
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = SKILL_DIR / "assets" / "LOG_template.md"


def repo_root() -> Path:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
        )
        return Path(out.decode().strip())
    except Exception:
        return Path.cwd()


REQUIREMENTS_STUB = """\
# torch/torchaudio는 setup_env.py가 환경(home=CUDA / work=MPS)에 맞게 설치합니다.
# 여기엔 넣지 마세요. 나머지 의존성만 나열하세요.
transformers==4.46.3   # home(CUDA cu121+torch2.5) 안전 핀. work(mac)에서 최신이 필요하면 조정.
"""

DOWNLOAD_STUB = '''\
"""대용량 자산(가중치·데이터셋·샘플 오디오) 다운로드 스크립트.

실제 파일은 .gitignore로 git에서 제외됩니다. 여기엔 URL/로더만 남기세요.
그러면 어느 머신에서든 이 스크립트만 돌려 자산을 복원할 수 있습니다.
"""
# 예시:
# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="facebook/encodec_24khz", filename="pytorch_model.bin",
#                 local_dir="outputs/weights")

if __name__ == "__main__":
    print("다운로드할 자산을 이 스크립트에 추가하세요.")
'''

RUN_STUB = '''\
import pathlib
import sys

# 레포 루트를 경로에 추가해 shared/env.py를 임포트
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.env import get_device

device = get_device()
print(f"device: {device}")

# TODO: 실험 코드
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="<topic>/<name> 형식")
    ap.add_argument("--title", default=None)
    ap.add_argument("--idea", action="store_true", help="논문이 아닌 아이디어 실험")
    ap.add_argument("--paper-link", default="")
    ap.add_argument("--notion-link", default="")
    args = ap.parse_args()

    parts = args.path.strip("/").split("/")
    if len(parts) != 2:
        print("경로는 <topic>/<name> 두 단계여야 합니다.", file=sys.stderr)
        return 1
    topic, name = parts

    root = repo_root()
    exp = root / topic / name
    if exp.exists():
        print(f"이미 존재합니다: {exp.relative_to(root)}", file=sys.stderr)
        return 1

    title = args.title or name
    today = dt.date.today().isoformat()
    kind_line = f"idea: {name}" if args.idea else f"paper: {name}"

    log = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("{{TITLE}}", title)
        .replace("{{TOPIC}}", topic)
        .replace("{{NAME}}", name)
        .replace("{{KIND_LINE}}", kind_line)
        .replace("{{DATE}}", today)
        .replace("{{PAPER_LINK}}", args.paper_link)
        .replace("{{NOTION_LINK}}", args.notion_link)
    )

    (exp / "outputs").mkdir(parents=True)
    (exp / "outputs" / ".gitkeep").write_text("", encoding="utf-8")
    (exp / "LOG.md").write_text(log, encoding="utf-8")
    (exp / "requirements.txt").write_text(REQUIREMENTS_STUB, encoding="utf-8")
    (exp / "download.py").write_text(DOWNLOAD_STUB, encoding="utf-8")
    (exp / "run.py").write_text(RUN_STUB, encoding="utf-8")

    print(f"생성됨: {exp.relative_to(root)}/")
    for f in ("LOG.md", "requirements.txt", "download.py", "run.py", "outputs/"):
        print(f"  - {f}")

    # INDEX 갱신
    subprocess.run([sys.executable, str(SKILL_DIR / "scripts" / "build_index.py")], check=False)

    print("\n다음 단계:")
    print(f"  1. {topic}/{name}/LOG.md 의 🎯 목표를 채우기")
    print(f"  2. cd {topic}/{name} && python {SKILL_DIR}/scripts/setup_env.py home  # 또는 work")
    print("  3. run.py 에 실험 코드 작성")
    return 0


if __name__ == "__main__":
    sys.exit(main())
