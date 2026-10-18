/**
 * Where a session is working — `repository`, `branch`, `module` — in the shape the recall
 * endpoint's `context` takes and the memory metadata keys context admission compares it with.
 *
 * The server's context admission (when a bank turns it on) uses these to tell this repository's
 * memories from another's in a shared bank, and to prefer the module and branch the task is in.
 * That only works if the SAME value is written at retain and sent at recall, from every checkout of
 * the repository, so `repository` is a stable identity rather than a path:
 *
 *   - the `origin` remote (else the first remote) from the repository's shared config, normalized:
 *     credentials, scheme, port and a trailing `.git` dropped, host lower-cased — so
 *     `git@github.com:acme/payments.git` and `https://user:tok@GitHub.com/acme/payments` are both
 *     `github.com/acme/payments`;
 *   - with no remote at all, the main checkout's folder name (the `{gitProject}` the bank id uses).
 *
 * Read off disk like core/git-layout.ts, never by spawning git: this runs inside hook windows.
 * Every linked worktree reads the common directory's config, so all worktrees of a repository
 * share one identity, while `branch` comes from the worktree's own HEAD. Anything that cannot be
 * determined is left out rather than guessed: an absent key is "unknown" to the server, a wrong
 * one is a mismatch.
 */
import { readFileSync, realpathSync } from "node:fs";
import { basename, relative, sep } from "node:path";
import { resolveProjectRoot } from "./bank";
import { probeWorktree } from "./git-layout";

export interface RepoContext {
  repository?: string;
  branch?: string;
  module?: string;
}

function readOrNull(path: string): string | null {
  try {
    return readFileSync(path, "utf8");
  } catch {
    return null;
  }
}

/**
 * A git remote URL as a stable repository identity, or null when nothing identifying is left.
 *
 * Handles URLs with a scheme (`https://`, `ssh://`, `git://`, `file://`), scp-like
 * `user@host:path`, and plain paths. Only the host is lower-cased: paths are case-sensitive on
 * most forges.
 */
export function normalizeRemoteUrl(raw: string): string | null {
  const url = raw.trim();
  if (!url) return null;
  let host = "";
  let path: string;
  const withScheme = /^([a-z][a-z0-9+.-]*):\/\/([^/]*)(.*)$/i.exec(url);
  const scpLike = /^(?:[^@/\s]+@)?([^:/\s]+):(?!\/\/)(.*)$/.exec(url);
  if (withScheme) {
    const authority = withScheme[2];
    path = withScheme[3];
    // user:password@host:port -> host
    host = authority.slice(authority.lastIndexOf("@") + 1).replace(/:\d*$/, "");
    if (withScheme[1].toLowerCase() === "file") host = "";
  } else if (scpLike && scpLike[1].length > 1) {
    // `host:path`; a one-letter "host" is a Windows drive (`C:\repos\app`), i.e. a local path.
    host = scpLike[1];
    path = scpLike[2];
  } else {
    path = url; // a local path remote
  }
  path = path
    .replace(/[?#].*$/, "")
    .replace(/\/+$/, "")
    .replace(/\.git$/i, "")
    .replace(/\/+$/, "")
    .replace(/^\/+/, host ? "" : "/");
  host = host.toLowerCase().replace(/^\[|\]$/g, "");
  const identity = host ? (path ? `${host}/${path}` : host) : path;
  return identity && identity !== "/" ? identity : null;
}

/** The remotes in a git config, in file order, as `[name, url]`. */
function remoteUrls(config: string): [string, string][] {
  const out: [string, string][] = [];
  let remote: string | null = null;
  for (const line of config.split(/\r?\n/)) {
    const section = /^\s*\[\s*([^\]\s"]+)(?:\s+"([^"]*)")?\s*\]/.exec(line);
    if (section) {
      remote =
        section[1].toLowerCase() === "remote" && section[2] !== undefined ? section[2] : null;
      continue;
    }
    if (remote === null) continue;
    const entry = /^\s*url\s*=\s*(.*?)\s*$/i.exec(line);
    if (entry) out.push([remote, entry[1].replace(/^"(.*)"$/, "$1")]);
  }
  return out;
}

/** The repository identity from a common directory's config: `origin`, else the first remote. */
export function repositoryFromConfig(config: string): string | null {
  const remotes = remoteUrls(config);
  const preferred = remotes.find(([name]) => name === "origin") ?? remotes[0];
  return preferred ? normalizeRemoteUrl(preferred[1]) : null;
}

/** The branch a HEAD file names, or null for a detached HEAD. */
export function branchFromHead(head: string): string | null {
  const ref = /^ref:\s*refs\/heads\/(.+?)\s*$/m.exec(head);
  return ref ? ref[1] : null;
}

/** The top-level folder of `directory` inside `worktree`, or null at the root or outside it. */
export function moduleOf(worktree: string, directory: string): string | null {
  const rel = relative(worktree, directory);
  if (!rel || rel.startsWith("..") || rel.startsWith(sep)) return null;
  const top = rel.split(sep)[0];
  return top && top !== "." ? top : null;
}

/** The layout's paths are canonical (`/tmp` is `/private/tmp` on macOS); compare like with like. */
function canonical(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return path;
  }
}

/** Repository, branch and module for `directory`; empty outside a repository or when the
 *  layout could not be read. */
export function repoContextOf(directory: string): RepoContext {
  if (!directory) return {};
  const layout = probeWorktree(directory);
  if (layout.status !== "resolved") return {};
  const context: RepoContext = {};
  const config = readOrNull(`${layout.commonDir}${sep}config`);
  let repository = config ? repositoryFromConfig(config) : null;
  if (!repository) {
    const root = resolveProjectRoot(directory);
    repository = root.status === "resolved" ? basename(root.root) : null;
  }
  if (repository) context.repository = repository;
  const head = readOrNull(`${layout.gitDir}${sep}HEAD`);
  const branch = head ? branchFromHead(head) : null;
  if (branch) context.branch = branch;
  const module = layout.worktree ? moduleOf(layout.worktree, canonical(directory)) : null;
  if (module) context.module = module;
  return context;
}
