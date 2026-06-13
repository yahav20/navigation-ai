# Bug Report — navigation-ai

Read-only code review of the backend (`src/`) and the `atlas-web` frontend. No fixes applied — findings only.

## Backend (`navigation-ai/src`)

| # | Severity | Location | Issue | Impact |
|---|----------|----------|-------|--------|
| 1 | Critical | `src/agent/itinerary/planner.py:430` | `AIMessage` used but never imported | `NameError` crash on `MAX_REPLANS` hard-stop — the error-recovery path itself crashes |
| 2 | Critical | `src/agent/itinerary/replanner.py:481` | `travel_plan` referenced but never defined, error swallowed | LLM quality-review step never runs — every itinerary silently uses the fallback markdown |
| 3 | High | `src/agent/itinerary/critic.py:261` / `src/agent/itinerary/planner.py:420` | `switch_travel_triggered` set but never reset | Repeatedly re-injects `switch_travel_options` on every future replan |
| 4 | High | `src/agent/travel/adjustments.py:110-114` | `return_flight_options` not cleared on travel reset (unlike `metadata.py`) | Stale inbound flights can pair with a fresh outbound search |
| 5 | Medium | `src/agent/advisor/replanner.py` | Off-by-one in `MAX_REPLAN_STEPS` check | Allows 6 replan cycles instead of the intended 5 |
| 6 | Medium | `src/agent/itinerary/replanner.py:565-587` | Dead `_evaluate_step_result` calls undefined `_strip_json_fences` | Latent `NameError` if ever wired up |
| 7 | Medium | `src/agent/itinerary/critic.py:41` | `MAX_CRITIC_ATTEMPTS` defined but never enforced | Documented safety ceiling doesn't actually exist |
| 8 | Medium | `src/agent/core/edge.py:67` + `graph.py` + `general_chat.py` | `GeneralChatNode` imported, never wired in; `"general_chat"` not a valid intent literal | Unreachable branch, half-finished feature |
| 9 | Medium | `src/agent/itinerary/formatter.py:1` | Stale header comment pointing to old path | Misleads contributors tracing imports |
| 10 | Medium | `src/agent/itinerary/step_handlers.py` (~197-355) | Multiple silent/unlogged broad `except Exception` blocks | Masks real tool failures behind hardcoded fallback values |
| 11 | Low | `src/agent/nodes/` | Orphaned `.pyc`-only directory, no source | Confuses anyone searching for "itinerary nodes" |
| 12 | Low | `src/agent/itinerary/planner.py` | Unused `silent` import | Dead import |
| 13 | Low | `src/agent/advisor/replanner.py:169` | Redundant re-import of `PlannedToolCall` | Code smell from refactor |
| 14 | Low | `src/agent/itinerary/activity_selector.py:595-615` | `coffee_place`/`breakfast_place` collapsed into one value | Inconsistent with `itinerary_tools.py`'s 4-slot model; LLM's distinct picks get discarded |
| 15 | Low | `src/agent/travel/alternatives.py:48-49` | Dead no-op `for _c in usable: pass` | Leftover debug code |
| 16 | Low | `src/security.py:106` | `validate_input` rejects all non-ASCII chars | Blocks legit input like accented names, €/₪, en-dashes |

## Frontend (`atlas-web/src`)

| # | Severity | Location | Issue | Impact |
|---|----------|----------|-------|--------|
| 1 | Critical/High | `src/components/thread/ItineraryViewer.tsx:133-135` | Client-side fetch to Nominatim, no User-Agent, no persistent cache, no timeout | Loading state can hang forever; risk of IP ban |
| 2 | High | `src/components/thread/ItineraryViewer.tsx:91-108` | Leaflet loaded from `unpkg.com` CDN, no `onerror`, no SRI | Hangs forever if CDN blocked/offline; supply-chain risk |
| 3 | High | `src/components/thread/ItineraryViewer.tsx:333,343-347` | `bindPopup` injects unsanitized LLM-generated strings as HTML | XSS via itinerary content (hotel name, description, etc.) |
| 4 | Medium-High | `src/providers/Stream.tsx:174-198` | `firstMessage` effect re-runs on every streamed token | Wasteful `.map()` over all threads on hot path |
| 5 | Medium-High | `src/components/thread/messages/ai.tsx:217-224` | `isLastMessage` can match >1 message with same `id` | Duplicate `<Interrupt>` (HITL prompt) rendering |
| 6 | Medium | `src/hooks/use-file-upload.tsx:83-188` | Drag/drop listeners torn down/re-attached on every `contentBlocks` change | `dragCounter` can desync, `dragOver` stuck |
| 7 | Medium | `src/components/thread/messages/human.tsx:55` | Editing a human message drops images/PDF attachments | Resubmits text-only, attachments silently lost |
| 8 | Medium | `src/components/thread/messages/human.tsx:49` | Edit `value` not resynced if `contentString` changes mid-edit | Stale content shown after branch switch while editing |
| 9 | Medium | `src/providers/Stream.tsx:62-85` | `mergeFetchedThreads` can overwrite fresher local state | Sidebar thread preview can lose latest exchange |
| 10 | Medium | `src/components/thread/index.tsx:311-391` | Fragile `prevMessageLength` bookkeeping for `firstTokenReceived` | Loading indicator can mis-trigger on regenerate/branch switch |
| 11 | Medium | `src/components/thread/ItineraryViewer.tsx:421` | `current` day index reset only on `destination` change, not `days.length` | `days[current]` undefined → crash during replans |
| 12 | Medium | `src/components/thread/ItineraryViewer.tsx:260-366` | `useEffect` deps `[day.day, destination]` w/ exhaustive-deps disabled | Map doesn't refresh when `day.slots`/`hotels` update |
| 13 | Medium | `src/components/thread/ItineraryViewer.tsx:269-363` | Concurrent `init()` calls race on shared `mapRef` | Possible "Leaflet map container already initialized" error |
| 14 | Medium | `src/components/thread/agent-selector.tsx:179` | Hardcoded "Build a 3-day itinerary..." contradicts `${days}` | Sends contradictory prompt to LLM regardless of user input |
| 15 | Low | `src/components/thread/messages/ai.tsx:118-133` | `useAgentStatus` is dead code, diverges from live logic | Maintenance hazard if revived |
| 16 | Low | `src/components/thread/index.tsx:543` | Fallback key `${message.type}-${index}` for ID-less messages | Possible animation/state misattribution during streaming |
| 17 | Low | `src/components/thread/messages/ai.tsx:294` | Meaningless `key` on `Fragment` outside list context | Leftover from refactor |
| 18 | Low | `src/lib/ensure-tool-responses.ts:10` | `tool_calls?.length === 0` doesn't catch `undefined` | Masked downstream by `?? []`, but fragile |
| 19 | Low | `src/providers/Stream.tsx:294` | `(stream.error as any).message` unsafe cast | Non-Error-shaped errors silently swallowed |
| 20 | Low | `src/components/thread/messages/ai.tsx:235-237` | `parsePartialJson` errors silently swallowed | No error UI for malformed tool-call args |
| 21 | Low | `src/components/thread/messages/ai.tsx:122,329` | `(values as any).progress_log` assumed `string[]`, unchecked | Non-string entries crash `AssistantMessageLoading` |
| 22 | Config | `atlas-web/tsconfig.json:30-31` | `include` references nonexistent paths | Leftover from upstream template |
| 23 | Config | `src/app/api/[..._path]/route.ts:8-9` | Missing env vars fall back to `"remove-me"` | Confusing failures instead of fail-fast config error |
| 24 | Config | `atlas-web/eslint.config.js:22` | `no-explicit-any` globally disabled | Hides backend↔frontend shape mismatches (#21) |

## Top priorities

- **Backend #2** — silent quality regression affecting every generated itinerary.
- **Frontend #1-3** — crash/XSS risk in the itinerary map view.
