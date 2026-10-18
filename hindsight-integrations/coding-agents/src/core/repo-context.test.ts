import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  branchFromHead,
  normalizeRemoteUrl,
  repoContextOf,
  repositoryFromConfig,
} from "./repo-context";

describe("normalizeRemoteUrl", () => {
  it.each([
    ["git@github.com:acme/payments.git", "github.com/acme/payments"],
    ["https://github.com/acme/payments.git", "github.com/acme/payments"],
    ["https://github.com/acme/payments", "github.com/acme/payments"],
    ["https://github.com/acme/payments/", "github.com/acme/payments"],
    ["https://user:s3cret@GitHub.com/acme/payments.git", "github.com/acme/payments"],
    ["https://x-access-token:ghp_abc@github.com/acme/payments.git/", "github.com/acme/payments"],
    ["ssh://git@github.com/acme/payments.git", "github.com/acme/payments"],
    ["ssh://git@GitLab.Example.com:2222/Team/Payments.git", "gitlab.example.com/Team/Payments"],
    ["git://github.com/acme/payments.git", "github.com/acme/payments"],
    ["http://gitea.local:3000/acme/payments?ref=main", "gitea.local/acme/payments"],
    ["github.com:acme/payments", "github.com/acme/payments"],
    ["file:///srv/git/payments.git", "/srv/git/payments"],
    ["/srv/git/payments.git", "/srv/git/payments"],
  ])("%s -> %s", (raw, expected) => {
    expect(normalizeRemoteUrl(raw)).toBe(expected);
  });

  it("never keeps credentials", () => {
    expect(normalizeRemoteUrl("https://alice:hunter2@github.com/acme/x")).not.toMatch(
      /alice|hunter2/
    );
  });

  it("returns null when nothing identifying is left", () => {
    expect(normalizeRemoteUrl("")).toBeNull();
    expect(normalizeRemoteUrl("   ")).toBeNull();
    expect(normalizeRemoteUrl("file:///")).toBeNull();
  });
});

describe("repositoryFromConfig", () => {
  it("prefers origin over an earlier remote", () => {
    const config = [
      "[core]",
      "\tbare = false",
      '[remote "upstream"]',
      "\turl = git@github.com:upstream/payments.git",
      '[remote "origin"]',
      "\turl = git@github.com:acme/payments.git",
      "\tfetch = +refs/heads/*:refs/remotes/origin/*",
    ].join("\n");
    expect(repositoryFromConfig(config)).toBe("github.com/acme/payments");
  });

  it("falls back to the first remote, and to null with none", () => {
    expect(repositoryFromConfig('[remote "fork"]\n  url = https://github.com/me/payments\n')).toBe(
      "github.com/me/payments"
    );
    expect(repositoryFromConfig("[core]\n  url = https://example.com/not-a-remote\n")).toBeNull();
  });
});

describe("branchFromHead", () => {
  it("reads a symbolic HEAD, including slashes in the name", () => {
    expect(branchFromHead("ref: refs/heads/feature/auth-fix\n")).toBe("feature/auth-fix");
  });

  it("has no branch on a detached HEAD", () => {
    expect(branchFromHead("4b825dc642cb6eb9a060e54bf8d69288fbee4904\n")).toBeNull();
  });
});

/** Real git, real worktrees — the reader has to agree with the tool that wrote the layout. */
function git(cwd: string, args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

describe("repoContextOf", () => {
  let root: string;
  let repo: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "hs-repoctx-"));
    repo = join(root, "payments");
    git(root, ["init", "-q", "-b", "main", repo]);
    git(repo, ["config", "user.email", "test@example.invalid"]);
    git(repo, ["config", "user.name", "Test"]);
    git(repo, ["commit", "--allow-empty", "-qm", "seed"]);
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("names the repository by its normalized origin, with branch and top-level module", () => {
    git(repo, ["remote", "add", "origin", "https://bot:tok@GitHub.com/acme/payments.git"]);
    const nested = join(repo, "services", "auth", "src");
    mkdirSync(nested, { recursive: true });
    expect(repoContextOf(nested)).toEqual({
      repository: "github.com/acme/payments",
      branch: "main",
      module: "services",
    });
    expect(repoContextOf(repo)).toEqual({ repository: "github.com/acme/payments", branch: "main" });
  });

  it("gives every linked worktree the same repository and its own branch", () => {
    git(repo, ["remote", "add", "origin", "git@github.com:acme/payments.git"]);
    const worktree = join(root, "payments-wt-auth"); // a sibling directory, as git makes them
    git(repo, ["worktree", "add", "-q", "-b", "fix/auth-timeout", worktree]);
    mkdirSync(join(worktree, "api"), { recursive: true });
    expect(repoContextOf(join(worktree, "api"))).toEqual({
      repository: "github.com/acme/payments",
      branch: "fix/auth-timeout",
      module: "api",
    });
  });

  it("falls back to the main checkout's folder name without a remote, from any worktree", () => {
    const worktree = join(root, "scratch-wt");
    git(repo, ["worktree", "add", "-q", "-b", "wt", worktree]);
    expect(repoContextOf(worktree).repository).toBe(basename(repo));
    expect(repoContextOf(repo).repository).toBe(basename(repo));
  });

  it("leaves the branch out on a detached HEAD", () => {
    git(repo, ["checkout", "-q", "--detach"]);
    expect(repoContextOf(repo)).toEqual({ repository: basename(repo) });
  });

  it("is empty outside any repository", () => {
    const plain = mkdtempSync(join(tmpdir(), "hs-repoctx-plain-"));
    try {
      expect(repoContextOf(plain)).toEqual({});
      expect(repoContextOf("")).toEqual({});
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });
});
