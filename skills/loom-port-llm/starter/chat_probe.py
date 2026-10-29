"""Derives the control program's "chat" block from a model's Jinja chat template.

The template is rendered on probe conversations and the pieces the control
program needs are cut out of the output: role prefixes/suffix, generation
prompts with thinking on/off, the default system prompt, the tools block and
its order relative to the system message, the tool-call format and separators,
tool-response framing, how reasoning appears in history, and content trimming.
`verify()` then re-renders a set of conversations the way control.loom will
(`simulate()`) and compares them with the template, so mismatches show up
before anything runs on the GPU.

usage: python chat_probe.py TEMPLATE_SOURCE [OUT.json]
  TEMPLATE_SOURCE: a .gguf (tokenizer.chat_template), a tokenizer_config.json, or a .jinja file
  (inspect_gguf.py calls derive() + verify() for you)
"""
import datetime
import json
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
CALL1 = {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}
CALL2 = {"name": "search", "arguments": {"query": "louvre hours"}}


class ProbeError(Exception):
    pass


def environment():
    def raise_exception(message):
        raise ProbeError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    # Same as transformers' chat-template environment.
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


def load_template(source):
    if source.endswith(".gguf"):
        import gguf
        f = gguf.GGUFReader(source).fields.get("tokenizer.chat_template")
        return f.contents() if f is not None else ""
    if source.endswith(".json"):
        t = json.load(open(source, encoding="utf-8")).get("chat_template", "")
        if isinstance(t, list):  # named templates: take "default"
            t = {x["name"]: x["template"] for x in t}.get("default", "")
        return t
    return open(source, encoding="utf-8").read()


class Template:
    def __init__(self, text, bos="", eos=""):
        self.t = environment().from_string(text)
        self.bos, self.eos = bos, eos

    def render(self, messages, tools=None, gen=False, **kw):
        return self.t.render(messages=messages, tools=tools, add_generation_prompt=gen,
                             bos_token=self.bos, eos_token=self.eos, **kw)


def calls_msg(content, calls, reasoning=None):
    m = {"role": "assistant", "content": content,
         "tool_calls": [{"id": f"call_{i}", "type": "function", "function": c} for i, c in enumerate(calls)]}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def between(text, a, b, start=0):
    i = text.index(a, start) + len(a)
    return text[i:text.index(b, i)]


def derive(template_text, bos="", eos=""):
    T = Template(template_text, bos, eos)
    notes = []
    chat = {}
    # BOS: emitted by the control program at the very start when the template prints it.
    chat["bos"] = bos if bos and T.render([{"role": "user", "content": "USERPROBE1"}]).startswith(bos) else ""
    # Prefixes and suffix: an assistant turn before the last user query renders plainly.
    r = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"},
                  {"role": "assistant", "content": "ASSTPROBE1"}, {"role": "user", "content": "USERPROBE2"}])
    if r.startswith(bos) and bos:
        r = r[len(bos):]
    chat["system_prefix"] = r[:r.index("SYSPROBE")]
    single = T.render([{"role": "user", "content": "USERPROBE1"}])
    chat["suffix"] = single[single.index("USERPROBE1") + len("USERPROBE1"):]
    gap = between(r, "SYSPROBE", "USERPROBE1")
    if not gap.startswith(chat["suffix"]):
        raise ProbeError(f"system message is not closed by the user suffix {chat['suffix']!r}: {gap!r}")
    chat["user_prefix"] = gap[len(chat["suffix"]):]
    gap = between(r, "USERPROBE1", "ASSTPROBE1")
    chat["assistant_prefix"] = gap[len(chat["suffix"]):]
    head = single[:single.index("USERPROBE1")]
    if head.startswith(bos) and bos:
        head = head[len(bos):]
    default = head[:len(head) - len(chat["user_prefix"])] if head.endswith(chat["user_prefix"]) else ""
    if default:
        if not (default.startswith(chat["system_prefix"]) and default.endswith(chat["suffix"])):
            raise ProbeError(f"default system block has an unexpected shape: {default!r}")
        chat["default_system"] = default[len(chat["system_prefix"]):len(default) - len(chat["suffix"])]
        notes.append(f"template inserts a default system prompt: {chat['default_system'][:60]!r}...")
    else:
        chat["default_system"] = ""
    trimmed = T.render([{"role": "user", "content": "  USERPROBE1 \n"}])
    chat["trim_content"] = "  USERPROBE1" not in trimmed
    # Generation prompts and thinking.
    base = T.render([{"role": "user", "content": "USERPROBE1"}])
    gen = lambda **kw: T.render([{"role": "user", "content": "USERPROBE1"}], gen=True, **kw)[len(base):]
    g_default, g_on, g_off = gen(), gen(enable_thinking=True), gen(enable_thinking=False)
    # Reasoning in history (agent loop: assistant after the last user query, followed by a tool result).
    agent = lambda m: T.render([{"role": "user", "content": "USERPROBE1"}, m,
                                {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"}], tools=TOOLS)
    with_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning="REASONPROBE"))
    without_r = agent(calls_msg("ASSTPROBE1", [CALL1], reasoning=""))
    mid_r = between(with_r, "USERPROBE1", "ASSTPROBE1")
    if "REASONPROBE" in mid_r:
        pre = mid_r[:mid_r.index("REASONPROBE")]
        cut = pre.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]) if chat["assistant_prefix"] in pre else None
        if cut is None:
            raise ProbeError(f"assistant prefix {chat['assistant_prefix']!r} not found before reasoning: {mid_r!r}")
        chat["think_open"] = pre[cut:]
        chat["think_close"] = mid_r[mid_r.index("REASONPROBE") + len("REASONPROBE"):]
        mid_n = between(without_r, "USERPROBE1", "ASSTPROBE1")
        empty_block = mid_n[mid_n.index(chat["assistant_prefix"]) + len(chat["assistant_prefix"]):]
        chat["history_think"] = "always" if empty_block == chat["think_open"] + chat["think_close"] else "nonempty"
        if empty_block not in ("", chat["think_open"] + chat["think_close"]):
            notes.append(f"empty reasoning renders as {empty_block!r} (not modeled)")
    else:
        chat["think_open"], chat["think_close"], chat["history_think"] = "", "", "nonempty"
    early = T.render([{"role": "user", "content": "USERPROBE1"},
                      {"role": "assistant", "content": "ASSTPROBE1", "reasoning_content": "REASONPROBE"},
                      {"role": "user", "content": "USERPROBE2"}])
    # Qwen-style templates drop reasoning from turns before the last user query.
    chat["history_think_scope"] = "all" if "REASONPROBE" in early else "after_last_query"
    think_marker = chat["think_open"].strip() or "<think>"
    thinks = any(think_marker in g for g in (g_default, g_on)) or bool(chat["think_open"])
    plain = g_off if g_off != g_on else g_default
    think = g_on if g_on != g_off else g_default
    if thinks and chat["think_open"] and not think.endswith(chat["think_open"]):
        # The model emits <think> itself (Qwen3 hybrid); open the block in the prompt so
        # output starts in the reasoning channel. Deliberate deviation: +2 prompt tokens.
        think = think + chat["think_open"]
        notes.append("generation_prompt appends think_open (the model would emit it): +tokens vs the template")
    toggle = g_on != g_off
    chat["generation_prompt"] = think if thinks else g_default
    chat["generation_prompt_plain"] = plain if toggle else chat["generation_prompt"]
    chat["thinking"] = "toggle" if (thinks and toggle) else ("always" if thinks else "never")
    chat["thinking_default"] = bool(thinks and (not toggle or g_default == g_on))
    chat["starts_in_reasoning"] = bool(thinks)
    chat["think_end"] = chat["think_close"].strip() or "</think>"
    # Tools block.
    t_plain = T.render([{"role": "user", "content": "USERPROBE1"}], tools=TOOLS)
    t_sys = T.render([{"role": "system", "content": "SYSPROBE"}, {"role": "user", "content": "USERPROBE1"}], tools=TOOLS)
    printed = None
    for only in (False, True):
        for seps in ((", ", ": "), (",", ":")):
            cand = [json.dumps(t["function"] if only else t, ensure_ascii=False, separators=seps) for t in TOOLS]
            if all(c in t_plain for c in cand):
                printed, chat["tools_function_only"] = cand, only
                if seps != (", ", ": "):
                    notes.append("template prints tools compactly; Loom prints the request's tool JSON as sent")
                break
        if printed:
            break
    if not printed:
        raise ProbeError("tool definitions are not printed as JSON objects one per line; fill tools_* by hand")
    block = t_plain[:t_plain.index("USERPROBE1")]
    if block.startswith(bos) and bos:
        block = block[len(bos):]
    block = block[:len(block) - len(chat["user_prefix"])]
    if not (block.startswith(chat["system_prefix"]) and block.endswith(chat["suffix"])):
        raise ProbeError(f"tools are not rendered as a system block: {block[:200]!r}")
    body = block[len(chat["system_prefix"]):len(block) - len(chat["suffix"])]
    i0, i1 = body.index(printed[0]), body.index(printed[1])
    # Each tool is preceded by tool_line_prefix (Qwen: "\n"; GLM: "").
    chat["tool_line_prefix"] = line = body[i0 + len(printed[0]):i1]
    if line.strip():
        raise ProbeError(f"tools are separated by non-whitespace text {line!r}; fill tools_* by hand")
    if not body[:i0].endswith(line):
        raise ProbeError("the first tool is not preceded like the others")
    tools_prefix, tools_suffix = body[:i0 - len(line)] if line else body[:i0], body[i1 + len(printed[1]):]
    s_body = t_sys[:t_sys.index("USERPROBE1")]
    s_body = s_body[len(bos) if bos and s_body.startswith(bos) else 0:]
    s_body = s_body[len(chat["system_prefix"]):len(s_body) - len(chat["user_prefix"]) - len(chat["suffix"])]
    if s_body.startswith("SYSPROBE"):
        chat["tools_after_system"] = True
        rest = s_body[len("SYSPROBE"):]
        lead = rest[:rest.index(printed[0]) - len(line)]
        body_start = len(lead) - len(lead.lstrip())
        chat["tools_system_separator"], chat["tools_prefix"] = lead[:body_start], lead[body_start:]
    elif s_body.endswith("SYSPROBE"):
        chat["tools_after_system"] = False
        rest = s_body[:-len("SYSPROBE")]
        chat["tools_prefix"] = rest[:rest.index(printed[0]) - len(line)]
        after = rest[rest.index(printed[1]) + len(printed[1]):]
        if not after.startswith(tools_suffix):
            raise ProbeError("tools suffix differs when a system message follows")
        chat["tools_system_separator"] = after[len(tools_suffix):]
    else:
        raise ProbeError(f"system message is not merged with the tools block: {s_body[:200]!r}")
    chat["tools_suffix"] = tools_suffix
    # Tool calls.
    two = T.render([{"role": "user", "content": "USERPROBE1"}, calls_msg("ASSTPROBE1", [CALL1, CALL2]),
                    {"role": "tool", "tool_call_id": "call_0", "content": "TOOLPROBE1"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "TOOLPROBE2"},
                    {"role": "user", "content": "USERPROBE2"}], tools=TOOLS)
    turn = two[two.index("ASSTPROBE1"):]
    if '{"name": "get_weather"' in turn:
        chat["call_format"] = "hermes"
    elif "<function=get_weather>" in turn:
        chat["call_format"] = "qwen_xml"
    elif "get_weather<arg_key>" in turn:
        chat["call_format"] = "glm"
    else:
        raise ProbeError("unknown tool-call format: " + turn[:300])
    start1 = two.index("<tool_call>", two.index("ASSTPROBE1"))
    chat["call_marker"] = "<tool_call>"
    chat["content_call_separator"] = two[two.index("ASSTPROBE1") + len("ASSTPROBE1"):start1]
    end1 = two.index("</tool_call>", start1) + len("</tool_call>")
    start2 = two.index("<tool_call>", end1)
    chat["call_separator"] = two[end1:start2]
    end2 = two.index("</tool_call>", start2) + len("</tool_call>")
    after_calls = two[end2:]
    if not after_calls.startswith(chat["suffix"]):
        raise ProbeError(f"assistant turn with calls is not closed by the suffix: {after_calls[:80]!r}")
    after_calls = after_calls[len(chat["suffix"]):]
    pre = after_calls[:after_calls.index("TOOLPROBE1")]
    mid = between(after_calls, "TOOLPROBE1", "TOOLPROBE2")
    post = after_calls[after_calls.index("TOOLPROBE2") + len("TOOLPROBE2"):after_calls.index("USERPROBE2")]
    post = post[:len(post) - len(chat["user_prefix"])]
    for k in range(len(mid) + 1):
        item_suffix, item_prefix = mid[:k], mid[k:]
        if pre.endswith(item_prefix) and post.startswith(item_suffix):
            chat["tool_turn_prefix"] = pre[:len(pre) - len(item_prefix)]
            chat["tool_item_prefix"], chat["tool_item_suffix"] = item_prefix, item_suffix
            chat["tool_turn_suffix"] = post[len(item_suffix):]
            break
    else:
        raise ProbeError(f"cannot split tool-response framing: pre={pre!r} mid={mid!r} post={post!r}")
    chat["terminal"] = chat["suffix"].strip() or chat["suffix"]
    return chat, notes


def simulate(chat, messages, tools=None, gen=False, enable_thinking=None):
    """Python model of control.loom's rendering (chat_begin + chat_message + chat_end)."""
    trim = (lambda s: s.strip()) if chat["trim_content"] else (lambda s: s)
    out = chat.get("bos", "")
    first_system = messages and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
    sys_text = trim(messages[0]["content"]) if first_system else ""
    use_default = not first_system and chat["default_system"]
    body = chat["default_system"] if use_default else sys_text
    if tools:
        printed = "".join(chat["tool_line_prefix"] + json.dumps(t["function"] if chat["tools_function_only"] else t, ensure_ascii=False)
                          for t in tools)
        tb = chat["tools_prefix"] + printed + chat["tools_suffix"]
        sep = chat["tools_system_separator"] if body else ""
        merged = body + sep + tb if chat["tools_after_system"] else tb + sep + body
        out += chat["system_prefix"] + merged + chat["suffix"]
    elif use_default:
        out += chat["system_prefix"] + chat["default_system"] + chat["suffix"]
    prior = None
    last_query = max([i for i, m in enumerate(messages) if m["role"] == "user"], default=-1)
    for i, m in enumerate(messages):
        role, text = m["role"], trim(m.get("content") or "")
        if prior == "tool" and role != "tool":
            out += chat["tool_turn_suffix"]
        if role == "system" and i == 0 and tools and first_system:
            pass
        elif role == "tool":
            out += ("" if prior == "tool" else chat["tool_turn_prefix"]) + chat["tool_item_prefix"] + text + chat["tool_item_suffix"]
        elif role == "assistant":
            thought = (m.get("reasoning_content") or "").strip()
            shown = chat["history_think_scope"] == "all" or i > last_query
            think = chat["think_open"] + thought + chat["think_close"] if shown and (thought or chat["history_think"] == "always") and chat["think_open"] else ""
            calls = "".join((chat["call_separator"] if j else "") + render_call(chat, c["function"])
                            for j, c in enumerate(m.get("tool_calls") or []))
            sep = chat["content_call_separator"] if text and calls else ""
            out += chat["assistant_prefix"] + think + text + sep + calls + chat["suffix"]
        else:
            out += (chat["user_prefix"] if role == "user" else chat["system_prefix"]) + text + chat["suffix"]
        prior = role
    if gen:
        if prior == "tool":
            out += chat["tool_turn_suffix"]
        on = chat["thinking_default"] if enable_thinking is None or chat["thinking"] != "toggle" else enable_thinking
        out += chat["generation_prompt"] if on else chat["generation_prompt_plain"]
    elif prior == "tool":
        out += chat["tool_turn_suffix"]
    return out


def render_call(chat, call):
    args = call["arguments"]
    if chat["call_format"] == "hermes":
        a = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return '<tool_call>\n{"name": "' + call["name"] + '", "arguments": ' + a + "}\n</tool_call>"
    args = json.loads(args) if isinstance(args, str) else args
    val = lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    if chat["call_format"] == "qwen_xml":
        return ("<tool_call>\n<function=" + call["name"] + ">\n"
                + "".join(f"<parameter={k}>\n{val(v)}\n</parameter>\n" for k, v in args.items())
                + "</function>\n</tool_call>")
    return ("<tool_call>" + call["name"]
            + "".join(f"<arg_key>{k}</arg_key><arg_value>{val(v)}</arg_value>" for k, v in args.items()) + "</tool_call>")


CONVERSATIONS = [
    ("chat", [{"role": "user", "content": "Hi there"}], None),
    ("system + multi-turn", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "One"},
                             {"role": "assistant", "content": "Two"}, {"role": "user", "content": "Three"}], None),
    ("tools", [{"role": "user", "content": "Weather in Paris?"}], TOOLS),
    ("system + tools", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}], TOOLS),
    ("agent loop", [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Weather in Paris?"},
                    calls_msg("", [{"name": "get_weather", "arguments": {"city": "Paris"}}], reasoning="Need the tool."),
                    {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 18}"}], TOOLS),
    ("parallel calls", [{"role": "user", "content": "Weather and search"},
                        calls_msg("Checking.", [CALL1, CALL2], reasoning="Two tools."),
                        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
                        {"role": "tool", "tool_call_id": "call_1", "content": "9-18"}], TOOLS),
]


def verify(template_text, chat, bos="", eos=""):
    """Returns [(name, ok, detail)]: template render vs simulate() for each conversation."""
    T = Template(template_text, bos, eos)
    results = []
    for name, msgs, tools in CONVERSATIONS:
        for kw in ({},) if chat["thinking"] != "toggle" else ({"enable_thinking": True}, {"enable_thinking": False}):
            label = name + (f" (enable_thinking={kw['enable_thinking']})" if kw else "")
            try:
                ref = T.render(msgs, tools=tools, gen=True, **kw)
            except ProbeError as e:
                results.append((label, None, f"template refuses: {e}"))
                continue
            got = simulate(chat, msgs, tools, gen=True, enable_thinking=kw.get("enable_thinking"))
            # The thinking prompt may deliberately append think_open.
            if got != ref and chat["thinking"] != "never" and got.endswith(chat["think_open"]) and got[:-len(chat["think_open"])] == ref:
                results.append((label, True, "exact except the deliberate think_open in the generation prompt"))
                continue
            if got == ref:
                results.append((label, True, "exact"))
            else:
                k = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
                results.append((label, False, f"differs at byte {k}: loom {got[max(0, k - 40):k + 60]!r} vs template {ref[max(0, k - 40):k + 60]!r}"))
    return results


def main():
    text = load_template(sys.argv[1])
    bos = ""
    if sys.argv[1].endswith(".gguf"):
        import gguf
        r = gguf.GGUFReader(sys.argv[1])
        f, tokens = r.fields.get("tokenizer.ggml.bos_token_id"), r.fields.get("tokenizer.ggml.tokens")
        if f is not None and tokens is not None:
            bos = tokens.contents()[f.contents()]
    chat, notes = derive(text, bos)
    for n in notes:
        print("note:", n)
    print(json.dumps(chat, indent=1, ensure_ascii=False))
    for name, ok, detail in verify(text, chat, bos):
        print(("OK  " if ok else "SKIP" if ok is None else "FAIL"), name, "-", detail)
    if len(sys.argv) > 2:
        json.dump(chat, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
