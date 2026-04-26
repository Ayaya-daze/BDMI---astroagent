#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_SWITCHER = path.join(__dirname, "codex_provider_switch.py");
const PROFILES_PATH = path.join(__dirname, "codex_provider_profiles.json");
const SNAPSHOT_DIR = path.join(os.homedir(), ".codex", "provider-switch-snapshots");
const STANDBY_SNAPSHOT_NAME = "current-thirdparty-standby";
const STANDBY_SNAPSHOT_PATH = path.join(
  SNAPSHOT_DIR,
  `${STANDBY_SNAPSHOT_NAME}.json`
);
const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";
const MENU_PROFILE_ORDER = [
  "official-chatgpt-login",
  "current-thirdparty-standby",
];
const COLOR = {
  reset: "\u001b[0m",
  bold: "\u001b[1m",
  dim: "\u001b[2m",
  red: "\u001b[31m",
  green: "\u001b[32m",
  yellow: "\u001b[33m",
  blue: "\u001b[34m",
  cyan: "\u001b[36m",
  gray: "\u001b[90m",
};
const BUILTIN_PROVIDER_IDS = new Set(["openai", "ollama", "lmstudio"]);
let useColor = process.stdout.isTTY;

const PROFILE_UI = {
  "official-chatgpt-login": {
    title: "官方登录",
    subtitle: "只走 openai + ChatGPT 登录，并清掉会干扰登录的 API key 配置",
    tone: "cyan",
  },
  "current-thirdparty-standby": {
    title: "第三方回退",
    subtitle: "恢复离开第三方前自动保存的 config/auth 原样快照，不再重新问 key",
    tone: "red",
  },
};

async function main() {
  const rawArgs = process.argv.slice(2);
  const parsed = parseArgs(rawArgs);
  useColor = process.stdout.isTTY && !parsed.flags.noColor;

  const profiles = loadProfiles();
  const profileNames = Object.keys(profiles);

  if (parsed.command === "help") {
    printHelp(profileNames);
    return 0;
  }

  if (parsed.command === "switch" && !parsed.profile) {
    console.error("缺少 profile 名称。");
    printHelp(profileNames);
    return 1;
  }

  if (parsed.command === "switch" && !profiles[parsed.profile]) {
    console.error(`未知 profile: ${parsed.profile}`);
    printList(profiles, null);
    return 1;
  }

  if (parsed.command !== "list") {
    ensureToolExists(PYTHON_SWITCHER, "缺少 Python 切换器脚本");
  }

  if (parsed.command === "status") {
    const snapshot = loadSnapshot();
    renderStatusCard(snapshot);
    return 0;
  }

  if (parsed.command === "list") {
    const snapshot = safeLoadSnapshot();
    printList(profiles, snapshot);
    return 0;
  }

  if (parsed.command === "switch") {
    const rl = createPromptInterface();
    try {
      return await runSwitchFlow({
        profileName: parsed.profile,
        profiles,
        flags: parsed.flags,
        interactive: true,
        rl,
      });
    } finally {
      rl.close();
    }
  }

  return await runInteractive(profiles, parsed.flags);
}

function parseArgs(argv) {
  const flags = {
    yes: false,
    dryRun: false,
    noColor: false,
  };
  const positionals = [];

  for (const arg of argv) {
    if (arg === "--yes" || arg === "-y") {
      flags.yes = true;
      continue;
    }
    if (arg === "--dry-run") {
      flags.dryRun = true;
      continue;
    }
    if (arg === "--no-color") {
      flags.noColor = true;
      continue;
    }
    positionals.push(arg);
  }

  if (positionals.length === 0) {
    return { command: "interactive", profile: null, flags };
  }

  const [first, second] = positionals;
  if (first === "help" || first === "--help" || first === "-h") {
    return { command: "help", profile: null, flags };
  }
  if (first === "status" || first === "list") {
    return { command: first, profile: null, flags };
  }
  if (first === "switch") {
    return { command: "switch", profile: second ?? null, flags };
  }
  return { command: "switch", profile: first, flags };
}

async function runInteractive(profiles, flags) {
  const snapshot = loadSnapshot();
  const menuProfiles = buildMenuProfiles(profiles);
  const defaultProfileName = guessCurrentProfile(snapshot.status) ?? menuProfiles[0];
  const defaultIndex = Math.max(menuProfiles.indexOf(defaultProfileName), 0);
  const menuMap = new Map();
  const rl = createPromptInterface();

  try {
    renderStatusCard(snapshot);
    console.log("");
    console.log(stylize("常用切换项", "bold"));

    menuProfiles.forEach((profileName, index) => {
      menuMap.set(String(index + 1), profileName);
      renderProfileLine(profiles, snapshot.status, profileName, index + 1);
    });

    console.log("");
    console.log(
      stylize(
        "提示: 真正导致 sidebar 看起来不共享会话的，是本地线程上的 provider 标签。这个入口现在会在切换时自动同步。",
        "dim"
      )
    );
    console.log(
      stylize(
        "这版只保留两个模式，不再提供额外切换分支。",
        "dim"
      )
    );

    const answer = (
      await prompt(
        rl,
        `\n输入编号或 profile 名称，直接回车默认 ${defaultIndex + 1}: `
      )
    ).trim();
    const profileName = answer
      ? menuMap.get(answer) ?? answer
      : menuProfiles[defaultIndex];

    if (!profiles[profileName]) {
      console.error(`未知选择: ${answer || "<empty>"}`);
      return 1;
    }

    return await runSwitchFlow({
      profileName,
      profiles,
      flags,
      interactive: true,
      rl,
      snapshot,
    });
  } finally {
    rl.close();
  }
}

async function runSwitchFlow({
  profileName,
  profiles,
  flags,
  interactive,
  rl = null,
  snapshot = null,
}) {
  const profile = profiles[profileName];
  const liveSnapshot = snapshot ?? loadSnapshot();
  const status = liveSnapshot.status;
  const targetProvider = getProfileProviderId(profile);
  const relabelPlan = buildRelabelPlan(status, targetProvider);
  const needsStandbySnapshot = shouldCaptureStandbySnapshot(status, profileName);

  if (profile.snapshot_only && !liveSnapshot.standbySnapshotExists) {
    console.error("缺少第三方待机快照，当前不能安全回退。");
    console.error(
      "请先在能正常连接第三方时运行一次官方入口，让它自动保存待机快照。"
    );
    return 1;
  }

  renderSwitchPlan({
    profileName,
    profile,
    status,
    relabelPlan,
    dryRun: flags.dryRun,
    standbySnapshotExists: liveSnapshot.standbySnapshotExists,
    standbySnapshotInfo: liveSnapshot.standbySnapshotInfo,
    needsStandbySnapshot,
  });

  if (!flags.yes) {
    const confirm = await askForConfirmation(rl, interactive);
    if (!confirm) {
      console.log("已取消。");
      return 0;
    }
  }

  if (needsStandbySnapshot) {
    if (flags.dryRun) {
      console.log("");
      console.log(
        stylize("dry-run: 将在真实执行前自动保存当前第三方 config/auth 待机快照", "dim")
      );
    } else {
      const snapshotResult = runCommand("python3", [
        PYTHON_SWITCHER,
        "snapshot-current",
        STANDBY_SNAPSHOT_NAME,
      ]);
      if (flags.dryRun || !snapshotResult.ok) {
        relayCommandResult(snapshotResult);
      }
      if (!snapshotResult.ok) {
        return snapshotResult.code;
      }
      liveSnapshot.standbySnapshotExists = true;
    }
  }

  const switchArgs = [PYTHON_SWITCHER, "switch", profileName];
  if (flags.dryRun) {
    switchArgs.push("--dry-run");
  }

  const switchResult = runCommand("python3", switchArgs);
  if (flags.dryRun || !switchResult.ok) {
    relayCommandResult(switchResult);
  }
  if (!switchResult.ok) {
    return switchResult.code;
  }

  if (relabelPlan) {
    const relabelArgs = [
      PYTHON_SWITCHER,
      "relabel-threads",
      "--from-provider",
      relabelPlan.fromProvider,
      "--to-provider",
      relabelPlan.toProvider,
      "--source",
      "all",
    ];
    if (flags.dryRun) {
      relabelArgs.push("--dry-run");
    }
    const relabelResult = runCommand("python3", relabelArgs);
    if (flags.dryRun || !relabelResult.ok) {
      relayCommandResult(relabelResult);
    }
    if (!relabelResult.ok) {
      return relabelResult.code;
    }
  }

  if (profileName === "official-chatgpt-login") {
    if (flags.dryRun) {
      console.log("");
      console.log(stylize("dry-run: 将在真实执行时额外运行 codex logout", "dim"));
    } else {
      const logoutResult = runCommand("codex", ["logout"]);
      if (flags.dryRun || !logoutResult.ok) {
        relayCommandResult(logoutResult);
      }
      if (!logoutResult.ok) {
        console.log(
          stylize(
            "警告: codex logout 没有成功，请手动执行一次，避免旧 keychain 状态影响登录。",
            "yellow"
          )
        );
      }
    }
  }

  console.log("");
  if (flags.dryRun) {
    console.log(stylize("预览完成", "bold", "yellow"));
    console.log(`目标模式     : ${getProfileTitle(profileName)} [${profileName}]`);
    return 0;
  }

  console.log(stylize("已切换", "bold", "green"));
  console.log(`当前模式     : ${getProfileTitle(profileName)} [${profileName}]`);
  if (needsStandbySnapshot) {
    console.log("第三方待机快照 : 已刷新");
  }
  if (relabelPlan) {
    console.log(
      `线程标签     : 已同步 ${relabelPlan.fromProvider} -> ${relabelPlan.toProvider}`
    );
  }
  if (profileName === "official-chatgpt-login") {
    console.log("下一步       : 完全退出并重开 Codex，然后按提示用 ChatGPT 登录");
  } else {
    console.log("下一步       : 完全退出并重开 Codex，让配置和 sidebar 一次性刷新");
  }
  return 0;
}

async function askForConfirmation(rl, interactive) {
  if (!interactive || !rl) {
    return false;
  }
  const answer = (await prompt(rl, "\n确认执行？[y/N]: ")).trim().toLowerCase();
  return answer === "y" || answer === "yes";
}

async function prompt(rl, text) {
  try {
    return await rl.question(text);
  } catch {
    return "";
  }
}

function loadProfiles() {
  const raw = readJsonFile(PROFILES_PATH, null);
  if (!raw) {
    throw new Error(`缺少 profile 文件: ${PROFILES_PATH}`);
  }
  return raw.profiles ?? raw;
}

function buildMenuProfiles(profiles) {
  return MENU_PROFILE_ORDER.filter((name) => profiles[name]);
}

function loadSnapshot() {
  return {
    status: loadStatus(),
    login: loadLoginStatus(),
    standbySnapshotExists: fs.existsSync(STANDBY_SNAPSHOT_PATH),
    standbySnapshotInfo: loadStandbySnapshotInfo(),
  };
}

function safeLoadSnapshot() {
  try {
    return loadSnapshot();
  } catch {
    return null;
  }
}

function loadStatus() {
  const result = runCommand("python3", [PYTHON_SWITCHER, "status", "--json"]);
  if (!result.ok) {
    throw new Error(result.stderr || result.stdout || "读取状态失败");
  }
  return JSON.parse(result.stdout);
}

function loadStandbySnapshotInfo() {
  const snapshotDoc = readJsonFile(STANDBY_SNAPSHOT_PATH, null);
  const config = snapshotDoc?.config;
  if (!config || typeof config !== "object") {
    return null;
  }
  return {
    modelProvider: config.model_provider || null,
    baseUrl: getBaseUrlFromConfig(config),
    model: config.model || null,
    reasoningEffort: config.model_reasoning_effort || null,
  };
}

function loadLoginStatus() {
  const result = runCommand("codex", ["login", "status"]);
  const raw = [result.stdout, result.stderr].filter(Boolean).join("\n").trim();

  if (!raw) {
    return {
      kind: "unknown",
      label: "未知",
      detail: "无法读取 codex login status",
    };
  }
  if (raw.includes("Not logged in")) {
    return {
      kind: "logged-out",
      label: "未登录",
      detail: "当前没有官方 ChatGPT 登录态",
    };
  }
  if (raw.includes("Logged in using an API key")) {
    return {
      kind: "api-key",
      label: "API Key",
      detail: raw.replace(/\s+/g, " ").trim(),
    };
  }
  if (raw.toLowerCase().includes("chatgpt")) {
    return {
      kind: "chatgpt",
      label: "ChatGPT",
      detail: raw.replace(/\s+/g, " ").trim(),
    };
  }
  return {
    kind: result.ok ? "other" : "error",
    label: result.ok ? "已登录" : "读取失败",
    detail: raw.replace(/\s+/g, " ").trim(),
  };
}

function buildRelabelPlan(status, targetProvider) {
  const counts = status.thread_counts_by_provider || {};
  const rolloutCounts = status.rollout_counts_by_provider || {};
  const api111Matches = Math.max(
    Number(counts.api111 || 0),
    Number(rolloutCounts.api111 || 0)
  );
  const openaiMatches = Math.max(
    Number(counts.openai || 0),
    Number(rolloutCounts.openai || 0)
  );
  if (targetProvider === "openai" && api111Matches > 0) {
    return {
      fromProvider: "api111",
      toProvider: "openai",
      count: api111Matches,
    };
  }
  if (targetProvider === "api111" && openaiMatches > 0) {
    return {
      fromProvider: "openai",
      toProvider: "api111",
      count: openaiMatches,
    };
  }
  return null;
}

function shouldCaptureStandbySnapshot(status, profileName) {
  return profileName !== STANDBY_SNAPSHOT_NAME && isThirdPartyLiveState(status);
}

function isThirdPartyLiveState(status) {
  if (!status) {
    return false;
  }
  return (
    status.model_provider === "api111" &&
    typeof status.base_url === "string" &&
    status.base_url !== "https://api.openai.com/v1"
  );
}

function guessCurrentProfile(status) {
  if (!status) {
    return null;
  }
  const provider = status.model_provider;
  const baseUrl = status.base_url;
  const forcedLoginMethod = status.forced_login_method || null;

  if (provider === "api111" && baseUrl === "https://api.xcode.best/v1") {
    return "current-thirdparty-standby";
  }
  if (provider === "openai" && forcedLoginMethod === "chatgpt") {
    return "official-chatgpt-login";
  }
  return null;
}

function renderStatusCard(snapshot) {
  const status = snapshot.status;
  const login = snapshot.login;
  const currentProfileGuess = guessCurrentProfile(status);
  const threadSummary = summarizeThreadCounts(status.thread_counts_by_provider || {});
  const rolloutSummary = summarizeThreadCounts(status.rollout_counts_by_provider || {});
  const guessedTitle = currentProfileGuess
    ? `${getProfileTitle(currentProfileGuess)} [${currentProfileGuess}]`
    : "未匹配到预设 profile";

  printRule();
  console.log(stylize("Codex Provider Switch", "bold", "blue"));
  console.log(stylize("更顺手的 Codex provider 切换入口", "dim"));
  printRule();
  console.log(`当前 provider : ${badge(status.model_provider || "<missing>", "cyan")}`);
  console.log(`当前 endpoint : ${status.base_url || "<missing>"}`);
  console.log(
    `当前模型     : ${status.model || "<missing>"} / ${
      status.model_reasoning_effort || "<missing>"
    }`
  );
  console.log(`当前登录态   : ${badge(login.label, loginTone(login.kind))} ${login.detail}`);
  console.log(
    `当前线程标签 : ${threadSummary || stylize("<none>", "dim")}`
  );
  console.log(
    `rollout 标签 : ${rolloutSummary || stylize("<none>", "dim")}`
  );
  if (
    Number(status.db_rollout_provider_mismatches || 0) > 0 ||
    Number(status.rollout_read_errors || 0) > 0
  ) {
    console.log(
      `标签检查     : mismatch=${status.db_rollout_provider_mismatches || 0} read_error=${status.rollout_read_errors || 0}`
    );
  }
  console.log(
    `第三方待机快照 : ${
      snapshot.standbySnapshotExists ? badge("ready", "green") : badge("missing", "red")
    }`
  );
  console.log(`当前配置像是 : ${guessedTitle}`);
  printRule();
}

function renderProfileLine(profiles, status, profileName, number) {
  const profile = profiles[profileName];
  const ui = PROFILE_UI[profileName] || {};
  const title = ui.title || profileName;
  const subtitle = ui.subtitle || profile.description || "";
  const targetProvider = getProfileProviderId(profile);
  const targetBaseUrl = getProfileBaseUrl(profile);
  const tone = ui.tone || "blue";
  const isCurrent = guessCurrentProfile(status) === profileName;

  const label = typeof number === "number" ? `${number}.` : String(number);

  console.log(
    `${stylize(label, tone)} ${stylize(title, "bold")} ${stylize(`[${profileName}]`, "dim")}${isCurrent ? ` ${badge("current", tone)}` : ""}`
  );
  console.log(`   ${subtitle}`);
  if (profile.snapshot_only) {
    console.log("   target: restore saved config/auth bundle");
  } else {
    console.log(`   target: ${targetProvider} -> ${targetBaseUrl}`);
  }
}

function renderSwitchPlan({
  profileName,
  profile,
  status,
  relabelPlan,
  dryRun,
  standbySnapshotExists,
  standbySnapshotInfo,
  needsStandbySnapshot,
}) {
  const targetProvider = getProfileProviderId(profile);
  const targetBaseUrl = getProfileBaseUrl(profile);
  const loginMode =
    profile.forced_login_method === "chatgpt" ? "ChatGPT" : "API Key";
  const snapshotModel = standbySnapshotInfo?.model || "<missing>";
  const snapshotReasoningEffort =
    standbySnapshotInfo?.reasoningEffort || "<missing>";
  const snapshotProvider = standbySnapshotInfo?.modelProvider || "saved-bundle";
  const snapshotBaseUrl = standbySnapshotInfo?.baseUrl || "saved-bundle";

  console.log("");
  printRule();
  console.log(stylize("切换计划", "bold"));
  printRule();
  console.log(
    `目标 profile : ${getProfileTitle(profileName)} ${stylize(`[${profileName}]`, "dim")}`
  );
  console.log(`当前 provider: ${status.model_provider || "<missing>"}`);
  console.log(
    `目标 provider: ${profile.snapshot_only ? snapshotProvider : targetProvider}`
  );
  console.log(
    `目标 endpoint: ${profile.snapshot_only ? snapshotBaseUrl : targetBaseUrl}`
  );
  console.log(
    `模型配置     : ${
      profile.snapshot_only
        ? snapshotModel
        : profile.model || "Codex 官方默认"
    } / ${
      profile.snapshot_only
        ? snapshotReasoningEffort
        : profile.reasoning_effort || "Codex 官方默认"
    }`
  );
  console.log(`登录方式     : ${loginMode}`);
  if (profile.snapshot_only) {
    console.log(
      `配置来源     : 已保存的第三方待机快照 (${standbySnapshotExists ? "ready" : "missing"})`
    );
  } else {
    console.log("API Key      : 不使用，强制走 ChatGPT 登录");
  }

  if (needsStandbySnapshot) {
    console.log("待机快照     : 将在切换前自动更新");
  }

  if (relabelPlan) {
    console.log(
      `线程同步     : ${relabelPlan.fromProvider} -> ${relabelPlan.toProvider} (${relabelPlan.count} 条)`
    );
  } else {
    console.log("线程同步     : 无需改动");
  }

  if (profileName === "official-chatgpt-login") {
    console.log("后续动作     : switch 完成后自动执行 codex logout");
  }
  if (dryRun) {
    console.log(`执行模式     : ${badge("dry-run", "yellow")}`);
  }
  printRule();
}

function printList(profiles, snapshot) {
  if (snapshot) {
    renderStatusCard(snapshot);
    console.log("");
  }
  console.log(stylize("全部 profile", "bold"));
  for (const profileName of Object.keys(profiles)) {
    renderProfileLine(profiles, snapshot?.status ?? null, profileName, "-");
  }
}

function printHelp(profileNames) {
  console.log(stylize("Codex Provider Switch", "bold"));
  console.log("");
  console.log("用法:");
  console.log("  codex-provider-switch");
  console.log("  codex-provider-switch status");
  console.log("  codex-provider-switch list");
  console.log("  codex-provider-switch switch <profile> [--yes] [--dry-run]");
  console.log("  codex-provider-switch <profile> [--yes] [--dry-run]");
  console.log("");
  console.log("日常 profile:");
  for (const name of MENU_PROFILE_ORDER) {
    if (profileNames.includes(name)) {
      console.log(`  - ${name}`);
    }
  }
  console.log("");
  console.log("说明:");
  console.log("  - 只保留两个模式: 官方登录 / 第三方回退。");
  console.log("  - 切到官方前会先自动保存当前第三方 config/auth 待机快照。");
  console.log("  - 第三方回退直接恢复保存好的配置包，不再重新问 key。");
  console.log("  - 切换 provider 时会自动 relabel 本地线程标签，但不会再输出线程内容。");
}

function createPromptInterface() {
  return createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: Boolean(process.stdin.isTTY && process.stdout.isTTY),
  });
}

function getProfileTitle(profileName) {
  return PROFILE_UI[profileName]?.title || profileName;
}

function getProfileProviderId(profile) {
  if (
    profile.provider_mode === "builtin" ||
    BUILTIN_PROVIDER_IDS.has(profile.provider_id)
  ) {
    return profile.provider_id;
  }
  return profile.provider.id;
}

function getProfileBaseUrl(profile) {
  if (
    profile.provider_mode === "builtin" ||
    BUILTIN_PROVIDER_IDS.has(profile.provider_id)
  ) {
    if (profile.provider_id === "openai") {
      if (profile.forced_login_method === "chatgpt") {
        return DEFAULT_CHATGPT_CODEX_BASE_URL;
      }
      return profile.openai_base_url || DEFAULT_OPENAI_BASE_URL;
    }
    return "<builtin>";
  }
  return profile.provider.base_url;
}

function getBaseUrlFromConfig(config) {
  if (config.model_provider === "openai") {
    if (config.openai_base_url) {
      return config.openai_base_url;
    }
    if (config.forced_login_method === "chatgpt") {
      return DEFAULT_CHATGPT_CODEX_BASE_URL;
    }
    return DEFAULT_OPENAI_BASE_URL;
  }
  const providers = config.model_providers;
  if (
    providers &&
    typeof providers === "object" &&
    providers[config.model_provider] &&
    typeof providers[config.model_provider] === "object"
  ) {
    return providers[config.model_provider].base_url || "<missing>";
  }
  return "<missing>";
}

function summarizeThreadCounts(counts) {
  const entries = Object.entries(counts || {});
  if (entries.length === 0) {
    return "";
  }
  return entries.map(([provider, count]) => `${provider}=${count}`).join("  ");
}

function loginTone(kind) {
  switch (kind) {
    case "api-key":
      return "yellow";
    case "chatgpt":
      return "green";
    case "logged-out":
      return "red";
    default:
      return "blue";
  }
}

function printRule() {
  console.log(stylize("=".repeat(72), "gray"));
}

function badge(text, tone) {
  return stylize(` ${text} `, "bold", tone);
}

function stylize(text, ...tones) {
  if (!useColor) {
    return text;
  }
  const parts = tones.map((tone) => COLOR[tone]).filter(Boolean).join("");
  return `${parts}${text}${COLOR.reset}`;
}

function relayCommandResult(result) {
  if (result.stdout) {
    process.stdout.write(result.stdout.endsWith("\n") ? result.stdout : `${result.stdout}\n`);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr.endsWith("\n") ? result.stderr : `${result.stderr}\n`);
  }
}

function runCommand(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    env: options.env ?? process.env,
  });

  if (result.error) {
    return {
      ok: false,
      code: 1,
      stdout: result.stdout || "",
      stderr: result.error.message,
    };
  }

  return {
    ok: result.status === 0,
    code: result.status ?? 1,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
  };
}

function ensureToolExists(filePath, message) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${message}: ${filePath}`);
  }
}

function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) {
      return fallback;
    }
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

main()
  .then((code) => {
    if (typeof code === "number") {
      process.exitCode = code;
    }
  })
  .catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
