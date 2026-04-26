# Codex Provider Switch

This toolset switches Codex between:

- your current working third-party config/auth bundle
- official OpenAI ChatGPT login mode

without hand-editing `~/.codex/config.toml`.

## Files

- `codex-provider-switch`
  - executable entrypoint
  - recommended way to launch the tool
- `codex_provider_switch.mjs`
  - prettier Node wrapper
  - shows current provider, login state, thread buckets, rollout buckets, and standby snapshot status
  - daily menu is intentionally reduced to two modes:
    - `official-chatgpt-login`
    - `current-thirdparty-standby`
  - automatically relabels local thread provider tags when switching between
    `api111` and `openai`
- `codex_provider_switch.py`
  - backend CLI for status, profile switching, backups, and thread relabeling
- `codex_provider_profiles.json`
  - profile definitions

## Why sessions looked "not shared"

Codex Desktop stores local thread visibility under the thread's
`model_provider` label in `~/.codex/state_5.sqlite`.

That means:

- old third-party threads may sit under `api111`
- official built-in provider threads may sit under `openai`
- if those labels differ, the sidebar can look split even though the local data
  is still on disk

The Node wrapper now fixes that automatically:

- switching to an `openai` profile relabels local `api111` threads to `openai`
- switching back to an `api111` profile relabels local `openai` threads to
  `api111`
- if an older run already changed the DB row but missed the rollout metadata,
  the next relabel also scans rollout `session_meta` and repairs those leftovers

Backups are still created before the DB is changed, and console output no
longer prints thread titles or prompt text during relabeling.

## The intended switching model

This tool now follows a simpler and safer model:

- `official-chatgpt-login`
  - switches Codex to built-in `openai`
  - forces ChatGPT login mode
  - intentionally does not pin `openai_base_url`; Codex should use the ChatGPT backend default in this mode
  - clears stale API-key auth that can interfere with official login
  - runs `codex logout` after switching so you can sign in cleanly
- `current-thirdparty-standby`
  - restores a saved working third-party `config.toml + auth.json` bundle
  - does not ask for the key again
  - does not try to rebuild the third-party config from loose prompt inputs

Before leaving a working third-party setup, the wrapper automatically saves a
standby bundle snapshot to:

```bash
~/.codex/provider-switch-snapshots/current-thirdparty-standby.json
```

That makes the rollback path deterministic.

## Recommended usage

Open the interactive selector:

```bash
./tools/codex/codex-provider-switch
```

Show the current status card:

```bash
./tools/codex/codex-provider-switch status
```

List the two supported modes:

```bash
./tools/codex/codex-provider-switch list
```

Dry-run a switch:

```bash
./tools/codex/codex-provider-switch switch official-chatgpt-login --dry-run --yes
```

You can also use the Python backend directly:

```bash
python3 ./tools/codex/codex_provider_switch.py status
```

## Supported modes

- `official-chatgpt-login`
  - recommended official mode
  - use this when you want Codex to show the ChatGPT sign-in flow
- `current-thirdparty-standby`
  - recommended rollback mode
  - restores the saved working third-party bundle exactly

## Third-party rollback behavior

The important change is that rollback is now bundle-based, not prompt-based.

- before leaving a working third-party setup, the wrapper saves the current
  config/auth bundle
- rolling back restores that saved bundle
- this avoids re-entering keys and avoids guessing how a particular third-party
  provider wanted its config written

## ChatGPT login note

By default, Codex stores CLI auth in `~/.codex/auth.json`.
Only if you explicitly set `cli_auth_credentials_store = "keyring"` or
`"auto"` does it prefer OS-level keychain/keyring storage.

For the common default setup, the stable official path is
`official-chatgpt-login`:

- switch config to built-in `openai`
- do not write `openai_base_url = "https://api.openai.com/v1"` in ChatGPT login mode
- remove API-key auth that can conflict
- run `codex logout`
- reopen Codex and sign in again

If you hit bootstrap errors such as `account/read failed during TUI bootstrap`,
run:

```bash
codex logout
```

Then reopen Codex and log in again.

## Manual backend commands

Relabel threads manually if you ever need it:

```bash
python3 ./tools/codex/codex_provider_switch.py relabel-threads \
  --from-provider openai \
  --to-provider api111 \
  --source all
```

Preview relabeling first:

```bash
python3 ./tools/codex/codex_provider_switch.py relabel-threads \
  --from-provider api111 \
  --to-provider openai \
  --source all \
  --dry-run
```

## Safety notes

- switching backs up `config.toml` and `auth.json`
- relabeling backs up `state_5.sqlite`
- built-in providers such as `openai` are handled through the proper built-in
  config path
- after a real switch or relabel, fully restart Codex Desktop so config, auth,
  and sidebar state reload together
