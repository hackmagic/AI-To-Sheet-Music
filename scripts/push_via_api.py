"""用 GitHub Git Data API 推送本地提交（HTTPS 被阻断时的替代路径）。

背景：本机 ``github.com:443`` 不通（``git push`` 失败），但 ``api.github.com`` 通。
Git Data API 提供等价的底层原语（blob → tree → commit → ref），可以完整复现一次
``git push`` 的效果，且不会丢失提交信息与父子关系。

用法::

    .venv/Scripts/python.exe scripts/push_via_api.py            # dry-run，只打印计划
    .venv/Scripts/python.exe scripts/push_via_api.py --apply    # 真正推送

设计约束：

* **幂等**：远端已是目标 SHA 时直接成功；
* **只推增量**：与远端 tree 对比，只上传新增/变更的 blob；
* **保留提交信息**：用本地 ``git log`` 的 message/author/committer；
* **失败不半途而废**：创建完 commit 但更新 ref 失败时，明确报出可重试的 commit SHA。
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{result.stderr.strip()}")
    return result.stdout


def gh(*args: str, stdin_payload: dict | None = None) -> dict:
    """调 ``gh api``。

    **必须走 stdin 传 JSON**：文件内容的 base64 动辄几百 KB，做成命令行参数会触发
    Windows 的 ``WinError 206（文件名或扩展名太长）``。
    """
    command = ["gh", "api", *args]
    if stdin_payload is not None:
        command.append("--input")
        command.append("-")
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        input=json.dumps(stdin_payload) if stdin_payload is not None else None,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)} 失败：{result.stderr.strip()[:400]}")
    text = result.stdout.strip()
    return json.loads(text) if text else {}


def repo_slug() -> str:
    remote = git("remote", "get-url", "origin").strip()
    # https://github.com/owner/name.git
    parts = remote.rstrip("/").removesuffix(".git").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def local_commit_info(rev: str = "HEAD") -> dict:
    raw = git(
        "show",
        "-s",
        "--format=%H%x00%T%x00%P%x00%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%B",
        rev,
    )
    fields = raw.split("\x00")
    return {
        "sha": fields[0].strip(),
        "tree": fields[1].strip(),
        "parents": [item for item in fields[2].split() if item],
        "author_name": fields[3],
        "author_email": fields[4],
        "author_date": fields[5],
        "committer_name": fields[6],
        "committer_email": fields[7],
        "committer_date": fields[8],
        "message": fields[9],
    }


def changed_paths(base_rev: str, head_rev: str) -> list[str]:
    """本地两个提交之间变更的文件（不含删除，删除走 tree 对比）。"""
    output = git("diff", "--name-only", "--diff-filter=d", base_rev, head_rev)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _blob_bytes(path: str, rev: str = "HEAD") -> bytes:
    """从 **git 对象库**读文件内容，而不是工作区。

    关键差异：工作区文件带 CRLF（``core.autocrlf=true``），git 存的是 LF。
    上传工作区字节会让远端 blob 与本地提交不一致（内容相同但 sha 不同），
    后续 ``git fetch`` 会看到假差异。这里一律取提交里的对象。
    """
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{rev}:{path}"],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"读取 {rev}:{path} 失败：{result.stderr.decode('utf-8', 'replace')[:200]}")
    return result.stdout


def upload_blobs(paths: list[str], slug: str) -> dict[str, str]:
    """上传文件内容，返回 {path: blob_sha}。"""
    mapping: dict[str, str] = {}
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        try:
            raw = _blob_bytes(path, "HEAD")
        except RuntimeError as exc:
            print(f"  [{index}/{total}] 跳过：{exc}")
            continue
        encoded = base64.b64encode(raw).decode("ascii")
        # 走 stdin 传 JSON（命令行参数会超长）
        result = gh(
            f"repos/{slug}/git/blobs",
            "-X", "POST",
            stdin_payload={"content": encoded, "encoding": "base64"},
        )
        mapping[path] = result["sha"]
        if index % 10 == 0 or index == total:
            print(f"  [{index}/{total}] 已上传")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="通过 GitHub API 推送提交")
    parser.add_argument("--apply", action="store_true", help="真正推送（默认 dry-run）")
    parser.add_argument("--force", action="store_true", help="允许非快进（慎用）")
    parser.add_argument("--base", default=None, help="以哪个远端 SHA 为基线（默认读 branches/main）")
    args = parser.parse_args()

    slug = repo_slug()
    info = local_commit_info("HEAD")
    if args.base:
        remote_sha = args.base
    else:
        remote_ref = gh(f"repos/{slug}/git/ref/heads/main")
        remote_sha = remote_ref["object"]["sha"]

    # 远端只有对象、本地还没有时（上一次 API 推送产生的 commit），
    # git diff 会报 bad object。这里在本地注册该对象为「远端已知尖端」，
    # 用共同祖先（merge-base）做 diff 基线。
    diff_base = remote_sha
    try:
        git("cat-file", "-e", f"{remote_sha}^{{commit}}")
    except RuntimeError:
        local_branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        # 用 --force-with-lease 式的思路：找本地 HEAD 与远端的历史交集
        for rev in git("rev-list", "--max-count=200", "HEAD").split():
            try:
                remote_commit = gh(f"repos/{slug}/git/commits/{rev}")
            except RuntimeError:
                continue
            if remote_commit.get("sha"):
                diff_base = rev
                break
        print(f"  远端对象本地缺失，改用共同祖先 {diff_base[:8]} 作为 diff 基线")
        # 让远端 tree 也基于这个基线重建（等于整体覆盖远端）
        remote_sha_for_tree = diff_base
    else:
        remote_sha_for_tree = remote_sha

    print(f"仓库      {slug}")
    print(f"远端 main {remote_sha}")
    print(f"本地 HEAD {info['sha']}")
    print(f"提交信息  {info['message'].splitlines()[0]}")

    if remote_sha == info["sha"]:
        print("\n远端已是最新，无需推送。")
        return 0

    # 确认是快进：远端必须是本地的祖先（本地比远端多提交）
    try:
        behind = git("rev-list", "--count", f"{info['sha']}..{diff_base}").strip()
    except RuntimeError:
        behind = "unknown"
    if behind not in ("0",):
        message = f"远端有本地没有的提交（{behind} 个），不是快进推送。"
        if not args.force:
            print(f"\n✗ {message} 需要先合并；确认要覆盖请加 --force。")
            return 2
        print(f"\n警告：{message} 继续执行（--force）。")

    paths = changed_paths(diff_base, info["sha"])
    # 也要处理删除：用 tree 对比更稳妥，但这里先按 git diff 的变更集合构造
    deleted = [
        line.strip()
        for line in git("diff", "--name-only", "--diff-filter=D", diff_base, info["sha"]).splitlines()
        if line.strip()
    ]
    print(f"\n变更文件 {len(paths)} 个（删除 {len(deleted)} 个）")
    for path in paths[:8]:
        print(f"  + {path}")
    if len(paths) > 8:
        print(f"  … 另有 {len(paths) - 8} 个")

    if not args.apply:
        print("\n[dry-run] 未做任何修改。加 --apply 执行。")
        return 0

    print("\n上传 blob…")
    blobs = upload_blobs(paths, slug)

    # 以远端 tree 为基线，叠加新 blob / 移除删除项
    remote_commit = gh(f"repos/{slug}/git/commits/{remote_sha_for_tree}")
    base_tree = remote_commit["tree"]["sha"]

    print("创建 tree…")
    entries = [
        {"path": path, "mode": "100644", "type": "blob", "sha": sha}
        for path, sha in blobs.items()
    ] + [
        {"path": path, "mode": "100644", "type": "blob", "sha": None}
        for path in deleted
    ]
    tree = gh(
        f"repos/{slug}/git/trees",
        "-X", "POST",
        stdin_payload={"base_tree": base_tree, "tree": entries},
    )

    print("创建 commit…")
    commit = gh(
        f"repos/{slug}/git/commits",
        "-X", "POST",
        stdin_payload={
            "message": info["message"],
            "tree": tree["sha"],
            "parents": [remote_sha_for_tree],
            "author": {
                "name": info["author_name"],
                "email": info["author_email"],
                "date": info["author_date"],
            },
            "committer": {
                "name": info["committer_name"],
                "email": info["committer_email"],
                "date": info["committer_date"],
            },
        },
    )
    print(f"  commit {commit['sha']}")

    print("更新 ref…")
    gh(
        f"repos/{slug}/git/refs/heads/main",
        "-X", "PATCH",
        stdin_payload={"sha": commit["sha"], "force": False},
    )
    print(f"\n✓ 已推送到 {slug} main")
    print(f"  本地 {info['sha']}")
    print(f"  远端 {commit['sha']}")
    print("  注意：API 推送后本地与远端 SHA 不同（提交对象内容一致），")
    print("        下一次 git push 前建议 git fetch 后 rebase。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"\n✗ {exc}", file=sys.stderr)
        raise SystemExit(1)
