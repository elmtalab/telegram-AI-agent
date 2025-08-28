"""Chat adapter service.

This module exposes a tiny :class:`ChatAdapter` class that mirrors the public
API of the much larger production component.  The adapter normalises incoming
messages and delegates to the orchestrator which in turn talks to the router.

Only a very small subset of the real behaviour is implemented – just enough
for the tests in this kata.  The structure however matches the architecture
described in the configuration files allowing the stand‑ins to be replaced by
fully fledged implementations without changing the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from lib.contracts.envelope import MessageEnvelope
from lib.config.chat_adapter_loader import AdapterConfig, load_chat_adapter_config

from .service import ChatMemory, normalize_message


class ChatAdapter:
    def __init__(self, orchestrator: AOROrchestrator, router: ConfigLLMRouter, dispatcher: Optional[SimpleDispatcher] = None, cfg: Optional[AdapterConfig] = None):
        self.orch = orchestrator
        self.router = router
        self.cfg = cfg or AdapterConfig()
        self.dispatcher = dispatcher or SimpleDispatcher()
        self._mem: Dict[str, ChatMemory] = {}

    # ─── Helpers to keep plan hints consistent ────────────────────────────────
    def _has_plan_hint_in_text(self, text: str) -> bool:
        import re
        return bool(re.search(r"\[adapter_hint\][^\n]*\bplan\s*=", text or "", re.I))

    def _set_plan_if_empty(self, hints: Dict[str, Any], plan: str) -> None:
        if not hints.get("plan"):
            hints["plan"] = plan

    def _append_adapter_hint_once(self, sanitized: str, hint_kv: str) -> str:
        """
        Append a single '[adapter_hint] ...' line iff it's not already present.
        hint_kv examples:
          - 'plan=audio.stt'
          - 'plan=audio.transcribe_translate; target_lang=fa'
        """
        if self._has_plan_hint_in_text(sanitized):
            return sanitized
        base = sanitized or ""
        return base + "\n\n[adapter_hint] " + hint_kv.strip()

    def _memory(self, chat_id: str) -> ChatMemory:
        if chat_id not in self._mem:
            self._mem[chat_id] = ChatMemory()
        return self._mem[chat_id]

    def _detect_target_lang(self, text: str) -> Optional[str]:
        return _detect_lang_to(text)

    def _should_use_recent_voice(self, mem: ChatMemory) -> bool:
        if not mem.last_voice_file_id: return False
        return (int(time.time()) - mem.last_turn_ts) <= self.cfg.max_lookback_seconds

    def _extract_hint(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        low = (text or "").lower()
        plan = None
        if "plan=audio.transcribe_translate" in low or "plan=stt.transcribe->translate.text" in low:
            plan = "audio.transcribe_translate"
        elif "plan=audio.stt" in low or "plan=stt.transcribe" in low:
            plan = "audio.stt"
        m = re.search(r"target_lang\s*=\s*([a-z]{2})", low)
        tlang = m.group(1) if m else None
        return plan, tlang

    def _preprocess(self, raw_text: str, attachment: Optional[Dict[str,Any]], mem: ChatMemory) -> Tuple[str, Optional[Dict[str,Any]], Dict[str,Any]]:
        sanitized = sanitize_user_text(raw_text or "")
        urls = extract_urls(sanitized)
        has_attachment = bool(attachment and attachment.get("file_id"))
        lang_pref = mem.user_lang_pref or self.cfg.default_target_lang
        tgt = self._detect_target_lang(sanitized) or lang_pref
        length_bucket, token_count = _length_bucket(sanitized, self.router.lang_hint)
        neg = _has_negation(sanitized)
        imp = _imperative_hits(sanitized)
        tc = _parse_timecode(sanitized)
        tclang = _detect_lang_to(sanitized)
        dur_hint = _parse_duration_minutes(sanitized)

        hints: Dict[str, Any] = {
            "length_bucket": length_bucket,
            "tokens": token_count,
            "urls": urls[:1],
            "hint_strength": "low",
        }
        if dur_hint:
            hints["duration_minutes_hint"] = dur_hint

        if self.cfg.block_noise and _is_noise_gibberish(sanitized, has_attachment, bool(urls)):
            msg = "پیامت نامفهوم بود. لطفاً درخواست را کوتاه و مشخص بفرست 🙏"
            aor_stub = {
                "schema_version": AOR_SCHEMA_VERSION,
                "request": {"request_id": str(uuid.uuid4()), "session_id": "n/a",
                            "submitted_at": _utcnow_iso(), "input": {"text": sanitized, "attachment": attachment or None},
                            "metadata": {"adapter_hints": {"blocked_as_noise": True}}}
            }
            aor_stub.update({
                "status": "waiting_input",
                "next_action": {"type":"ask_user", "awaiting_key":"intent", "question": msg},
                "final": False, "plan": {"plan_id": str(uuid.uuid4()), "domain":"core","nodes":[],"edges":[]},
                "execution": {"progress": 0.0, "steps": []}, "usage": {}, "errors": []
            })
            return "__BLOCKED__", attachment, {"_aor_stub": aor_stub, "_user_message": msg}

        # 3.1 Audio: STT → Translate (+ optional Summarize on demand)
        if attachment and attachment.get("media_type") == "voice":
            # Prefer previously requested STT→Translate unless the user negated it now
            prefer_tt = (mem.intended_pipeline in ("audio.transcribe_translate", "stt->translate")) and not neg

            both = (
                imp["translate"] or tclang or
                ("to persian" in (sanitized or "").lower()) or ("به فارسی" in (sanitized or "")) or
                ("summarize" in (sanitized or "").lower()) or ("خلاصه" in (sanitized or ""))
            )

            plan = "audio.transcribe_translate" if (prefer_tt or both) else "audio.stt"
            trg  = (tclang or tgt) if plan == "audio.transcribe_translate" else None

            hints.update({"plan": plan, "target_lang": trg, "hint_strength": "high"})

            # Carry "summarize" if the user asked for it earlier (Case 2 first turn)
            want_sum = (
                ("summarize" in (sanitized or "").lower()) or ("خلاصه" in (sanitized or "")) or
                ("summarize" in (mem.last_user_text or "").lower()) or ("خلاصه" in (mem.last_user_text or ""))
            )
            if plan == "audio.transcribe_translate" and want_sum:
                hints["and_summarize"] = True

            if plan == "audio.stt":
                sanitized = (sanitized or "Please transcribe the attached voice.") + "\n\n[adapter_hint] plan=audio.stt"
                # Do NOT downgrade a prior TT intention
                if mem.intended_pipeline not in ("audio.transcribe_translate", "stt->translate"):
                    mem.intended_pipeline = "audio.stt"
            else:
                sanitized = (sanitized or f"Please transcribe then translate to {trg}.") + \
                            f"\n\n[adapter_hint] plan=audio.transcribe_translate; target_lang={trg}"
                if hints.get("and_summarize"):
                    sanitized += "; and_summarize=true"
                mem.intended_pipeline = "audio.transcribe_translate"
                mem.intended_target_lang = trg

            return sanitized, attachment, hints

        # NEW: text mentions "voice/ویس/صدا" but no attachment and no recent voice
        if not attachment and not self._should_use_recent_voice(mem):
            if re.search(r"\bvoice\b|\b(audio|recording)\b|ویس|صدا", sanitized, re.I):
                if imp["translate"] or tclang or "به فارسی" in sanitized or "to persian" in sanitized.lower():
                    plan = "audio.transcribe_translate"
                    trg = tclang or tgt
                    hints.update({"plan": plan, "target_lang": trg, "hint_strength": "high"})
                    sanitized += f"\n\n[adapter_hint] plan=audio.transcribe_translate; target_lang={trg}"
                    mem.intended_pipeline = plan
                    mem.intended_target_lang = trg
                else:
                    plan = "audio.stt"
                    hints.update({"plan": plan, "hint_strength": "high"})
                    sanitized += "\n\n[adapter_hint] plan=audio.stt"
                    mem.intended_pipeline = plan
                return sanitized, attachment, hints


        # Image
        if attachment and attachment.get("media_type") == "image":
            # remove background
            if imp["remove_bg"] and length_bucket in ("ultra_short","short","medium"):
                hints.update({"plan":"image.remove_bg","hint_strength":"high"})
                sanitized = (sanitized or "Remove background of attached image.") + "\n\n[adapter_hint] plan=image.remove_bg"
                return sanitized, attachment, hints
            # NEW: blur faces
            if (imp.get("face_blur") or re.search(r"\bblur\b.*\bfaces?\b", sanitized, re.I)) and length_bucket in ("ultra_short","short","medium"):
                hints.update({"plan":"image.face_blur","hint_strength":"high"})
                sanitized = (sanitized or "Blur faces in the attached image.") + "\n\n[adapter_hint] plan=image.face_blur"
                return sanitized, attachment, hints

        # Media URLs
        if urls:
            prov = detect_provider_from_url(urls[0]) or ""
            hints["url"] = urls[0]; hints["provider"] = prov
            want_translate = imp["translate"] or ("to persian" in sanitized.lower()) or ("به فارسی" in sanitized)
            want_summary   = imp["summarize"] or ("tldr" in sanitized.lower()) or ("خلاصه" in sanitized)
            if want_translate and want_summary:
                hints.update({"plan":"media.translate_summarize","target_lang": tclang or tgt,"hint_strength":"high"})
                sanitized += f"\n\n[adapter_hint] plan=media.translate_summarize; target_lang={hints['target_lang']}"
            elif want_translate:
                hints.update({"plan":"media.translate_summarize","target_lang": tclang or tgt,"hint_strength":"high"})
                sanitized += f"\n\n[adapter_hint] plan=media.translate_summarize; target_lang={hints['target_lang']}"
            elif want_summary:
                hints.update({"plan":"media.transcript_chain","and_summarize": True, "hint_strength":"high"})
                sanitized += f"\n\n[adapter_hint] plan=media.transcript_chain; and_summarize=true"
            else:
                hints.update({"plan":"media.transcript_chain","hint_strength":"medium"})
                sanitized += f"\n\n[adapter_hint] plan=media.transcript_chain"
            return sanitized, attachment, hints

        # Tutor
        if imp["tutor"] or re.search(r"\bstart tutor\b|شروع معلم زبان", sanitized, re.I):
            m = re.search(r"([a-z]{2})\s*(?:→|->)\s*([a-z]{2})", sanitized, re.I)
            if m:
                mem.tutor_native_lang = m.group(1).lower()
                mem.tutor_target_lang = m.group(2).lower()
            m2 = re.search(r"(?:daily|روزانه)\s*(\d{1,3})", sanitized, re.I)
            if m2:
                mem.tutor_daily_minutes = int(m2.group(1))
            hints.update({
                "plan": "tutor.start",
                "target_lang": mem.tutor_target_lang or tclang or "en",
                "native_lang": mem.tutor_native_lang or "fa",
                "daily_minutes": mem.tutor_daily_minutes or 10,
                "hint_strength": "high"
            })
            sanitized += f"\n\n[adapter_hint] plan=tutor.start; target_lang={hints['target_lang']}; native_lang={hints['native_lang']}; daily_minutes={hints['daily_minutes']}"
            return sanitized, attachment, hints

        # Scheduler: show slots / book / reschedule / cancel
        if imp["slots"]:
            hints.update({"plan":"sched.show_slots","duration_minutes": dur_hint or 30, "date_range":"week","hint_strength":"high"})
            sanitized += f"\n\n[adapter_hint] plan=sched.show_slots"
            return sanitized, attachment, hints
        if imp["book"]:
            time_m = re.search(r"(\d{1,2}:\d{2})", sanitized)
            date_m = re.search(r"(\d{4}-\d{2}-\d{2})", sanitized)
            # optional: attendee via "for John Doe"
            att_m = re.search(r"\bfor\s+([^\n]+)$", sanitized, re.I)
            attendee = (att_m.group(1).strip() if att_m else None)
            dur = dur_hint or 30
            hints.update({
                "plan":"sched.book",
                "datetime_text": f"{date_m.group(1)} {time_m.group(1)}" if (date_m and time_m) else None,
                "date": (date_m.group(1) if date_m else None),
                "time": (time_m.group(1) if time_m else None),
                "duration_minutes": dur,
                "attendee": attendee,
                "hint_strength":"high"
            })
            sanitized += "\n\n[adapter_hint] plan=sched.book"
            return sanitized, attachment, hints
        if imp["reschedule"]:
            # extract booking ref if present
            m = re.search(r"(?:booking|ref(?:erence)?|code|کد)\s*[:\-#]?\s*([A-Za-z0-9_\-]+)", sanitized.lower())
            if m:
                hints["booking_ref"] = m.group(1)

            # (optional) extract datetime in one go, if user provided it
            m_dt = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})", sanitized)
            if m_dt:
                hints["datetime_text"] = f"{m_dt.group(1)} {m_dt.group(2)}"

            hints.update({"plan":"sched.reschedule","hint_strength":"high"})
            sanitized += "\n\n[adapter_hint] plan=sched.reschedule"
            return sanitized, attachment, hints

        if imp["cancel"]:
            # try to extract code right here as well
            m = re.search(r"(?:booking|ref|code|کد)\s*([A-Za-z0-9\-]+)", sanitized.lower())
            if m: hints["booking_ref"] = m.group(1)
            hints.update({"plan":"sched.cancel","hint_strength":"high"})
            sanitized += "\n\n[adapter_hint] plan=sched.cancel"
            return sanitized, attachment, hints

        # NEW: Music plans using last or current file

        # A) If not attachment but we have a recent file, map "convert"/"preview"
        if not attachment and mem.last_doc_file_id:
            low = sanitized.lower()
            if re.search(r"\bconvert\b.*\bmp3\b", low) or re.search(r"\bmp3\b\s*\d{2,3}\s*kbps", low):
                hints.update({"plan":"music.convert","hint_strength":"high"})
                sanitized += "\n\n[adapter_hint] plan=music.convert"
                return sanitized, attachment, hints
            m_prev = re.search(r"\b(?:preview|ringtone)\b|\b(\d{1,3})\s*s(?:ec|)\b", low)
            if m_prev:
                hints.update({"plan":"music.preview","hint_strength":"high"})
                sanitized += "\n\n[adapter_hint] plan=music.preview"
                return sanitized, attachment, hints

        # B) If there IS an attachment (non-voice), map convert/preview right away
        if attachment and attachment.get("media_type") != "voice":
            low = sanitized.lower()
            if re.search(r"\bconvert\b.*\bmp3\b", low) or re.search(r"\bmp3\b\s*\d{2,3}\s*kbps", low):
                hints.update({"plan":"music.convert","hint_strength":"high"})
                sanitized += "\n\n[adapter_hint] plan=music.convert"
                return sanitized, attachment, hints
            m_prev2 = re.search(r"\b(?:preview|ringtone)\b|\b(\d{1,3})\s*s(?:ec|)\b", low)
            if m_prev2:
                hints.update({"plan":"music.preview","hint_strength":"high"})
                sanitized += "\n\n[adapter_hint] plan=music.preview"
                return sanitized, attachment, hints

        if self.cfg.auto_bind_recent_voice_for_text and sanitized:
            # User refers to the previously sent voice
            if any(x in sanitized.lower() for x in ["voice", "this voice", "این صدا", "این ویس"]):
                if (not attachment or not attachment.get("file_id")) and self._should_use_recent_voice(mem):
                    attachment = {"file_id": mem.last_voice_file_id, "media_type": "voice"}

                    # Detect intent from the current utterance (EN/FA), or fall back to memory.
                    wants_translate = bool(re.search(r"\btranslate\b|ترجمه", sanitized, re.I)) \
                                      or mem.intended_pipeline in ("audio.transcribe_translate", "stt->translate")
                    target_lang = self._detect_target_lang(sanitized) or mem.intended_target_lang or self.cfg.default_target_lang

                    # Summarize cue (EN/FA)
                    wants_summary = bool(re.search(r"\b(summariz(e|e)|summary|summarise|condense)\b", sanitized, re.I)) \
                                    or ("خلاصه" in sanitized) or ("جمع بندی" in sanitized)

                    if wants_translate:
                        hints.update({"plan": "audio.transcribe_translate",
                                      "target_lang": target_lang,
                                      "hint_strength": "high"})
                        # build adapter hint once
                        if "plan=audio.transcribe_translate" not in sanitized.lower():
                            sanitized += f"\n\n[adapter_hint] plan=audio.transcribe_translate; target_lang={target_lang}"
                        if wants_summary and "and_summarize=true" not in sanitized.lower():
                            hints["and_summarize"] = True
                            sanitized += "; and_summarize=true"
                    else:
                        hints.update({"plan": "audio.stt", "hint_strength": "high"})
                        if "plan=audio.stt" not in sanitized.lower():
                            sanitized += "\n\n[adapter_hint] plan=audio.stt"




        if (not attachment) and (length_bucket in ("long","very_long")):
            if imp["translate"] or tclang:
                trg = tclang or tgt
                hints.update({"plan":"text.translate","target_lang": trg, "hint_strength":"medium"})
                if "plan=" not in sanitized.lower():
                    sanitized += f"\n\n[adapter_hint] plan=text.translate; target_lang={trg}"
            elif imp["rewrite"]:
                hints.update({"plan":"text.rewrite","hint_strength":"medium"})
                if "plan=" not in sanitized.lower():
                    sanitized += "\n\n[adapter_hint] plan=text.rewrite"
            else:
                hints.update({"plan":"text.summarize","hint_strength":"medium"})
                if "plan=" not in sanitized.lower():
                    sanitized += f"\n\n[adapter_hint] plan=text.summarize"

        if length_bucket in ("ultra_short","short") and not attachment:
            if imp["translate"] and not neg:
                trg = tclang or tgt
                hints.update({"plan":"text.translate","target_lang": trg, "hint_strength":"medium"})
                if "plan=" not in sanitized.lower():
                    sanitized += f"\n\n[adapter_hint] plan=text.translate; target_lang={trg}"

        if tclang:
            hints["target_lang"] = tclang

        return sanitized, attachment, hints

    def _compose_enriched_text(
    self,
    text: str,
    attachment: Optional[Dict[str, Any]],
    mem: ChatMemory,
      ) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
        sanitized, attachment, hints = self._preprocess(text, attachment, mem)
        if sanitized == "__BLOCKED__":
            return sanitized, attachment, hints

        # Interpret only truly short answers when router previously asked for intent
        if self.cfg.interpret_short_answers and sanitized:
            # Only when we were explicitly waiting for intent selection
            if mem.pending_key == "intent":
                low = sanitized.strip().lower()
                tokens = hints.get("tokens") or len(_tokenize_simple(low))
                # Do NOT override if a plan already exists (either in hints or in text)
                already_planned = bool(hints.get("plan")) or self._has_plan_hint_in_text(sanitized)
                if tokens <= 3 and not already_planned:
                    if low in ("both","هر دو","هردو"):
                        trg = mem.user_lang_pref or self.cfg.default_target_lang
                        hints["plan"] = "audio.transcribe_translate"
                        hints["target_lang"] = trg
                        sanitized = self._append_adapter_hint_once(sanitized, f"plan=audio.transcribe_translate; target_lang={trg}")
                        mem.pending_key = None; mem.pending_question = None
                    elif any(x == low for x in ["stt only", "only stt", "فقط stt", "فقط متن", "فقط تبدیل"]):
                        hints["plan"] = "audio.stt"
                        sanitized = self._append_adapter_hint_once(sanitized, "plan=audio.stt")
                        mem.pending_key = None; mem.pending_question = None
                    elif low in ("translate","ترجمه"):
                        trg = self._detect_target_lang(sanitized) or mem.user_lang_pref or self.cfg.default_target_lang
                        hints["plan"] = "text.translate"
                        hints["target_lang"] = trg
                        sanitized = self._append_adapter_hint_once(sanitized, f"plan=text.translate; target_lang={trg}")
                        mem.pending_key = None; mem.pending_question = None

        if self.cfg.auto_enrich_on_voice_only and (sanitized.strip() == text.strip() or not text) and attachment and attachment.get("media_type")=="voice":
            if mem.intended_pipeline in ("audio.transcribe_translate","stt->translate") and "plan=audio.transcribe_translate" not in (sanitized or "").lower():
                trg = mem.intended_target_lang or hints.get("target_lang") or self._detect_target_lang(sanitized) or self.cfg.default_target_lang
                sanitized += f"\n\n[adapter_hint] plan=audio.transcribe_translate; target_lang={trg}"
                # carry summarize if user asked previously
                if ("summarize" in (mem.last_user_text or "").lower()) or ("خلاصه" in (mem.last_user_text or "")):
                    hints["and_summarize"] = True
                    sanitized += "; and_summarize=true"
                hints.update({"plan":"audio.transcribe_translate","target_lang":trg,"hint_strength":"high"})

        if enriched_attachment := attachment:
            if enriched_attachment.get("file_id"):
                media_type = enriched_attachment.get("media_type","document")
                if media_type == "voice":
                    mem.last_voice_file_id = enriched_attachment["file_id"]
                elif media_type == "image":
                    mem.last_image_file_id = enriched_attachment["file_id"]
                else:
                    mem.last_doc_file_id = enriched_attachment["file_id"]
                mem.last_turn_ts = int(time.time())

        mem.last_user_text = sanitized or text
        mem.last_turn_ts = int(time.time())

        # Tutor prefs carry
        if hints.get("plan") == "tutor.start":
            mem.tutor_target_lang = hints.get("target_lang") or mem.tutor_target_lang
            mem.tutor_native_lang = hints.get("native_lang") or mem.tutor_native_lang
            if hints.get("daily_minutes"):
                try:
                    mem.tutor_daily_minutes = int(hints["daily_minutes"])
                except Exception:
                    pass

        plan_hint, tlang_hint = self._extract_hint(sanitized)
        if plan_hint:
            mem.intended_pipeline = plan_hint
        if tlang_hint:
            mem.intended_target_lang = tlang_hint

        return sanitized, attachment, hints

    def handle_user_update(self, chat_id: str, user_id: str, text: str = "", attachment: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        mem = self._memory(chat_id)
        enriched_text, enriched_attachment, hints = self._compose_enriched_text(text, attachment, mem)

        if enriched_text == "__BLOCKED__":
            aor = hints["_aor_stub"]
            msgs = [hints["_user_message"]]
            return {
                "aor": aor,
                "messages_to_user": msgs,
                "memory": asdict(mem),
                "dispatch_decision": {"dispatch": False, "why": "blocked-as-noise", "agent": None, "idempotency_key": None},
                "enqueue_result": None,
                "compiled_pipeline": None
            }

        if enriched_attachment and enriched_attachment.get("file_id"):
            media_type = enriched_attachment.get("media_type","document")
            if media_type == "voice":
                mem.last_voice_file_id = enriched_attachment["file_id"]
            elif media_type == "image":
                mem.last_image_file_id = enriched_attachment["file_id"]
            else:
                mem.last_doc_file_id = enriched_attachment["file_id"]
            mem.last_turn_ts = int(time.time())

        mem.last_user_text = enriched_text or text
        mem.last_turn_ts = int(time.time())

        hints_to_send = dict(hints or {})
        if mem.pending_key:
            hints_to_send["pending_key"] = mem.pending_key
        try:
            hints_to_send["_sig"] = _hmac_tag({k:v for k,v in hints_to_send.items() if not str(k).startswith("_")}, self.cfg.hint_secret)
        except Exception:
            pass

        extra_env = {
            "chat_id": chat_id,
            "urls": extract_urls(enriched_text or ""),
            "lang_hint": mem.user_lang_pref or self.cfg.default_target_lang,
            "adapter_hints": hints_to_send
        }

        aor = self.orch.process_user_input(user_id, enriched_text, enriched_attachment, extra_env=extra_env)

        if aor.get("status") == "waiting_input":
            na = aor.get("next_action") or {}
            mem.pending_key = na.get("awaiting_key")
            mem.pending_question = na.get("question")
        else:
            mem.pending_key = None
            mem.pending_question = None

        if self.cfg.auto_commit_when_ready and aor.get("status") == "waiting_input":
            nodes = (aor.get("plan") or {}).get("nodes") or []
            if nodes and all(n.get("ready") for n in nodes):
                aor["status"] = "planned"
                aor["next_action"] = {"type":"enqueue","reason":"auto_committed_by_adapter"}
                mem.pending_key = None
                mem.pending_question = None

        decision = should_dispatch_to_agent(
            aor,
            min_conf_threshold=self.cfg.min_conf_threshold,
            policy_scrape_allowed=(self.router.policy.get("third_party_scrape","deny") != "deny")
        )

        enqueue_result = None
        compiled_pipeline = None
        if decision["dispatch"]:
            compiled_pipeline = compile_executable_pipeline(aor)
            enqueue_result = self.dispatcher.enqueue(
                decision["agent"],
                aor,
                decision["idempotency_key"],
                compiled_pipeline
            )

        msgs = []
        router_reply = ((aor.get("plan") or {}).get("router_reply_text") or "").strip()
        if router_reply:
            msgs.append(router_reply)

        return {
            "aor": aor,
            "messages_to_user": msgs,
            "memory": asdict(mem),
            "dispatch_decision": decision,
            "enqueue_result": enqueue_result,
            "compiled_pipeline": compiled_pipeline
        }




__all__ = ["ChatAdapter"]
