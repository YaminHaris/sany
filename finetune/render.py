"""Renders a conversation exactly the way Ollama 0.32's qwen2.5 template does."""

import json

PROP_ORDER = ['anyOf', 'type', 'items', 'description', 'enum']


def go_json(obj, sort_keys=True):
    s = json.dumps(obj, separators=(',', ':'), ensure_ascii=False, sort_keys=sort_keys)
    return s.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _prop(p):
    return {k: p[k] for k in PROP_ORDER if k in p and p[k] not in (None, '', [])}


def tool_function_json(fn, prop_sort=True):
    params = fn['parameters']
    ordered = {'type': params.get('type', 'object')}
    if params.get('required'):
        ordered['required'] = params['required']
    props = params.get('properties', {})
    keys = sorted(props) if prop_sort else list(props)
    ordered['properties'] = {k: _prop(props[k]) for k in keys}
    out = {'name': fn['name']}
    if fn.get('description'):
        out['description'] = fn['description']
    out['parameters'] = ordered
    return go_json(out, sort_keys=False)


def render(messages, tools, add_generation_prompt=False, args_sort=True, prop_sort=True):
    """Returns (text, spans) where spans are (start, end) char ranges of assistant outputs to train on."""
    system = next((m['content'] for m in messages if m['role'] == 'system'), '')
    msgs = []
    for m in messages:
        if m['role'] == 'system':
            continue
        # Ollama merges consecutive user turns into one (session context + request).
        if msgs and m['role'] == 'user' and msgs[-1]['role'] == 'user':
            msgs[-1] = {'role': 'user', 'content': msgs[-1]['content'] + '\n\n' + m['content']}
        else:
            msgs.append(m)
    text = ''
    if system or tools:
        text += '<|im_start|>system'
        if system:
            text += '\n' + system
        if tools:
            text += ('\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
                     'You are provided with function signatures within <tools></tools> XML tags:\n<tools>')
            for t in tools:
                text += '\n{"type": "function", "function": ' + tool_function_json(t['function'], prop_sort) + '}'
            text += ('\n</tools>\n\nFor each function call, return a json object with function name and arguments '
                     'within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')
        text += '<|im_end|>\n'
    spans = []
    for i, m in enumerate(msgs):
        last = i == len(msgs) - 1
        if m['role'] == 'user':
            text += '<|im_start|>user\n' + m['content'] + '<|im_end|>\n'
        elif m['role'] == 'assistant':
            text += '<|im_start|>assistant\n'
            start = len(text)
            if m.get('content'):
                text += m['content']
            elif m.get('tool_calls'):
                text += '<tool_call>\n'
                for c in m['tool_calls']:
                    f = c['function']
                    text += '{"name": "' + f['name'] + '", "arguments": ' + go_json(f['arguments'], sort_keys=args_sort) + '}\n'
                text += '</tool_call>'
            if not last:
                text += '<|im_end|>\n'
                spans.append((start, len(text) - 1))  # include <|im_end|>, exclude trailing newline
            else:
                text += '<|im_end|>'
                spans.append((start, len(text)))
        elif m['role'] == 'tool':
            text += '<|im_start|>user\n<tool_response>\n' + m['content'] + '\n</tool_response><|im_end|>\n'
        if m['role'] != 'assistant' and last and add_generation_prompt:
            text += '<|im_start|>assistant\n'
    return text, spans
