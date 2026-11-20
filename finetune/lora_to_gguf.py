"""Minimal PEFT LoRA (Qwen2) -> llama.cpp/Ollama GGUF adapter converter.

Usage: python lora_to_gguf.py out/lora-v1 out/lora-v1/qwen-agent-lora.gguf
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'llama.cpp' / 'gguf-py'))
import gguf  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

PROJ = {
    'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k', 'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output', 'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up',
    'mlp.down_proj': 'ffn_down',
}
KEY = re.compile(r'layers\.(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.lora_([AB])\.weight$')


def main(src, dst):
    cfg = json.load(open(Path(src) / 'adapter_config.json'))
    assert not cfg.get('use_dora'), 'DoRA is not supported'
    tensors = load_file(str(Path(src) / 'adapter_model.safetensors'))

    alpha = float(cfg['lora_alpha'])
    if cfg.get('use_rslora'):
        # rsLoRA scales by alpha / sqrt(r), whereas GGUF/llama.cpp scales by LORA_ALPHA / r.
        alpha = alpha * math.sqrt(float(cfg['r']))

    w = gguf.GGUFWriter(dst, 'qwen2')
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, 'lora')
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    n = 0
    for k in sorted(tensors):
        m = KEY.search(k)
        if not m:
            raise SystemExit(f'unexpected tensor {k}')
        layer, proj, ab = m.groups()
        name = f'blk.{layer}.{PROJ[proj]}.weight.lora_{ab.lower()}'
        w.add_tensor(name, tensors[k].astype(np.float16))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f'wrote {n} tensors to {dst}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
