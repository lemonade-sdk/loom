# Shared Glimmer prompt rendering (must match glimmer-loom control.loom exactly).
from tokenizers import Tokenizer
import os
TOK = Tokenizer.from_file(os.environ["GLIMMER_TOKENIZER"])  # the model's HF tokenizer.json
SYSTEM = ('<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\n'
          'Reasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|>')
def render(user):
    return SYSTEM + "<|start|>user<|message|>" + user + "<|eot|><|start|>assistant"
def ids(text):
    return TOK.encode(text, add_special_tokens=False).ids
def raw_from_chat(reasoning, content, finished):
    # Reconstruct the model's raw output text from the parsed channels.
    t = ""
    if reasoning is not None:
        t += " to=self<|message|>" + reasoning
        if content is not None or finished:
            t += "<|eom|><|start|>assistant"
    if content is not None:
        t += " to=user<|message|>" + content
    return t
