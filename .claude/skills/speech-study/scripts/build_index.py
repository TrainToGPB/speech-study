#!/usr/bin/env python3
"""LOG.md 프론트매터를 스캔해 루트 INDEX.md를 재생성한다.

표준 라이브러리만 사용 — 시스템 파이썬으로 실행 가능.
실험은 항상 <topic>/<name>/LOG.md 두 단계 깊이에 있다고 가정한다.
"""
import subprocess
import sys
from pathlib import Path

# 실험 폴더가 아닌 최상위 디렉토리
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


def collect(root: Path) -> list[dict]:
    rows = []
    for topic_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if topic_dir.name in IGNORE_TOPLEVEL or topic_dir.name.startswith("."):
            continue
        for exp_dir in sorted(p for p in topic_dir.iterdir() if p.is_dir()):
            log = exp_dir / "LOG.md"
            if not log.exists():
                continue
            fm = parse_frontmatter(log.read_text(encoding="utf-8"))
            kind = "💡 아이디어" if "idea" in fm else "📄 논문"
            rows.append(
                {
                    "topic": fm.get("topic", topic_dir.name),
                    "name": exp_dir.name,
                    "rel": log.relative_to(root).as_posix(),
                    "kind": kind,
                    "status": STATUS_EMOJI.get(str(fm.get("status", "planned")), fm.get("status", "")),
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
        "| 주제 | 실험 | 유형 | 상태 | 검증환경 | 갱신일 |",
        "|------|------|------|------|----------|--------|",
    ]
    for r in sorted(rows, key=lambda x: (x["topic"], x["name"])):
        lines.append(
            f"| {r['topic']} | [{r['name']}]({r['rel']}) | {r['kind']} | "
            f"{r['status']} | {r['env']} | {r['updated']} |"
        )
    if not rows:
        lines.append("| _아직 없음_ | | | | | |")
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
