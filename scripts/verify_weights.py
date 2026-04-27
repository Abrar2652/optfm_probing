#!/usr/bin/env python3
"""
Deep verification: is the experiment really using pre-trained OPTFM weights,
or silently falling back to random initialization?

Checks:
  1. Checkpoint file exists and is a valid state_dict.
  2. Enumerate ALL checkpoint keys vs ALL model keys.
  3. load_state_dict(strict=True) — must succeed or explicitly list deltas.
  4. Parameter-by-parameter: before vs after loading — which tensors changed?
  5. Compare a freshly-instantiated "random" model with the "pretrained" model:
     for every shared key, assert tensors are NOT equal (otherwise nothing loaded).
  6. Run the canonical-pair forward and also compare embeddings between the two
     models on a non-trivial input — if they are bit-identical, loading failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from scripts._tee_log import start_logging
from models.sgformer_mip import SGFormer_MIP, create_model
from data.milp_pairs import construct_canonical_pair, construct_control_pair, milp_to_tensors


CKPT = Path("D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth")


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def main():
    start_logging("verify_weights")
    banner("1. CHECKPOINT FILE")
    print(f"Path:    {CKPT}")
    print(f"Exists:  {CKPT.exists()}")
    print(f"Size:    {CKPT.stat().st_size:,} bytes")

    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)
    print(f"Type:    {type(state_dict).__name__}")
    print(f"# keys:  {len(state_dict)}")

    banner("2. CHECKPOINT KEYS vs MODEL KEYS")
    fresh = SGFormer_MIP()
    model_keys = set(fresh.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    print(f"Model has {len(model_keys)} keys, checkpoint has {len(ckpt_keys)} keys")

    only_in_model = sorted(model_keys - ckpt_keys)
    only_in_ckpt = sorted(ckpt_keys - model_keys)
    shared = sorted(model_keys & ckpt_keys)

    print(f"\nShared keys       : {len(shared)}")
    print(f"Missing from ckpt : {len(only_in_model)}")
    for k in only_in_model:
        print(f"   - {k}")
    print(f"Unexpected in ckpt: {len(only_in_ckpt)}")
    for k in only_in_ckpt:
        print(f"   + {k}")

    banner("3. SHAPE COMPATIBILITY FOR SHARED KEYS")
    shape_mismatches = []
    for k in shared:
        ms = tuple(fresh.state_dict()[k].shape)
        cs = tuple(state_dict[k].shape)
        if ms != cs:
            shape_mismatches.append((k, ms, cs))
    if shape_mismatches:
        print(f"!! {len(shape_mismatches)} SHAPE MISMATCHES:")
        for k, ms, cs in shape_mismatches:
            print(f"   {k}: model={ms}  ckpt={cs}")
    else:
        print(f"OK: all {len(shared)} shared keys have matching shapes")

    banner("4. STRICT LOAD — MUST SUCCEED")
    strict_model = SGFormer_MIP()
    try:
        strict_model.load_state_dict(state_dict, strict=True)
        print("strict=True SUCCEEDED — every model parameter was loaded from ckpt")
        strict_ok = True
    except RuntimeError as e:
        print("strict=True FAILED:")
        print(str(e))
        strict_ok = False

    banner("5. PARAMETER-BY-PARAMETER: did values actually change?")
    # Baseline: same seed random init
    torch.manual_seed(0)
    baseline = SGFormer_MIP()
    baseline_sd = {k: v.clone() for k, v in baseline.state_dict().items()}

    # Now load weights into the same-seed baseline
    torch.manual_seed(0)
    loaded = SGFormer_MIP()
    missing, unexpected = loaded.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    changed = 0
    unchanged = 0
    unchanged_names = []
    for k in shared:
        before = baseline_sd[k]
        after = loaded.state_dict()[k]
        if torch.allclose(before, after, atol=0, rtol=0):
            unchanged += 1
            unchanged_names.append(k)
        else:
            changed += 1

    print(f"Changed     : {changed} / {len(shared)}")
    print(f"Unchanged   : {unchanged} / {len(shared)}")
    if unchanged_names:
        print("Unchanged keys (SUSPICIOUS if any are weights):")
        for k in unchanged_names[:10]:
            print(f"   - {k}")

    banner("6. TENSOR NORM SANITY CHECK ON SELECTED LAYERS")
    pick = [
        "var_embedding.weight",
        "cons_embedding.weight",
        "trans_conv.convs.0.Wq.weight",
        "trans_conv.convs.0.Wk.weight",
        "trans_conv.convs.0.Wv.weight",
        "GCN.conv_v_to_c.feature_module_left.0.weight",
        "GCN.conv_c_to_v.feature_module_left.0.weight",
        "fc.weight",
    ]
    print(f"{'key':<55} {'‖ckpt‖':>10} {'‖baseline‖':>12}  equal?")
    for k in pick:
        if k in state_dict:
            ck = state_dict[k].norm().item()
            bs = baseline_sd[k].norm().item()
            eq = torch.allclose(state_dict[k], baseline_sd[k])
            print(f"{k:<55} {ck:>10.4f} {bs:>12.4f}  {eq}")
        else:
            print(f"{k:<55}   (not in ckpt)")

    banner("7. END-TO-END: do pretrained vs random models produce DIFFERENT outputs?")
    # Use the factory exactly as the experiment does
    m_pre = create_model("optfm", pretrained_path=str(CKPT))
    m_rnd = create_model("random")

    # Check a few parameters differ
    pre_sd = m_pre.state_dict()
    rnd_sd = m_rnd.state_dict()

    diff_params = 0
    same_params = 0
    for k in pre_sd.keys():
        if torch.allclose(pre_sd[k], rnd_sd[k]):
            same_params += 1
        else:
            diff_params += 1
    print(f"create_model('optfm', ckpt) vs create_model('random')")
    print(f"   differing params: {diff_params}")
    print(f"   identical params: {same_params}")

    # Forward pass on the control pair (chain vs star) — non-trivial input
    pair = construct_control_pair()
    cons_x, ei, ea, var_x = milp_to_tensors(pair.milp_a)

    m_pre.eval(); m_rnd.eval()
    with torch.no_grad():
        e_pre = m_pre.get_graph_embedding(cons_x, ei, ea, var_x)
        e_rnd = m_rnd.get_graph_embedding(cons_x, ei, ea, var_x)

    print(f"\ncontrol-A embedding (pretrained): norm={e_pre.norm():.6f}")
    print(f"control-A embedding (random)    : norm={e_rnd.norm():.6f}")
    print(f"cosine(pre, rnd)                : {torch.nn.functional.cosine_similarity(e_pre.unsqueeze(0), e_rnd.unsqueeze(0)).item():.6f}")
    print(f"bit-identical                   : {torch.equal(e_pre, e_rnd)}")

    # Canonical pair — 1-WL indistinguishable; both embeddings of A and B should match
    # *within each model*, but pretrained and random should still differ *between* models.
    canon = construct_canonical_pair()
    ca, ia, aa, va = milp_to_tensors(canon.milp_a)
    cb, ib, ab, vb = milp_to_tensors(canon.milp_b)
    with torch.no_grad():
        pa = m_pre.get_graph_embedding(ca, ia, aa, va)
        pb = m_pre.get_graph_embedding(cb, ib, ab, vb)
        ra = m_rnd.get_graph_embedding(ca, ia, aa, va)
        rb = m_rnd.get_graph_embedding(cb, ib, ab, vb)
    print("\nCanonical pair (should be identical within each model):")
    print(f"   pretrained:  cos(A,B) = {torch.nn.functional.cosine_similarity(pa.unsqueeze(0), pb.unsqueeze(0)).item():.6f}")
    print(f"   random    :  cos(A,B) = {torch.nn.functional.cosine_similarity(ra.unsqueeze(0), rb.unsqueeze(0)).item():.6f}")
    print("Between-model canonical-A comparison (should DIFFER if weights loaded):")
    print(f"   cos(pretrained_A, random_A) = {torch.nn.functional.cosine_similarity(pa.unsqueeze(0), ra.unsqueeze(0)).item():.6f}")
    print(f"   bit-identical               = {torch.equal(pa, ra)}")

    banner("VERDICT")
    ok_strict   = strict_ok
    ok_changed  = changed > 0
    ok_diff     = diff_params > 0
    ok_outputs  = not torch.equal(e_pre, e_rnd)

    print(f"[{'PASS' if ok_strict  else 'FAIL'}] strict=True load succeeded")
    print(f"[{'PASS' if ok_changed else 'FAIL'}] loaded params differ from fresh init ({changed}/{len(shared)} changed)")
    print(f"[{'PASS' if ok_diff    else 'FAIL'}] create_model('optfm', ckpt) differs from create_model('random') ({diff_params} params)")
    print(f"[{'PASS' if ok_outputs else 'FAIL'}] pretrained and random models produce different embeddings on control input")

    if ok_strict and ok_changed and ok_diff and ok_outputs:
        print("\n>>> VERIFIED: the experiment is using real pre-trained OPTFM weights.")
        return 0
    else:
        print("\n>>> PROBLEM: weight loading is NOT what it claims to be.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
