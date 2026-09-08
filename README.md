# seecode
"""Explicit measurement/capture contracts; no model or CUDA initialization here."""

from time import perf_counter


MODES = ("performance", "capture")
TIMING_CONTRACT = {
    "version": 1,
    "primary_metric": "decode_seconds",
    "generation_start": "before_request_reset_and_prompt_H2D",
    "decode_start": "prefill_complete_first_target_root_available",
    "generation_end": "stop_commit_cleanup_and_required_device_work_complete",
    "includes": "proposal_catchup_verify_accept_commit_required_transfers_and_sync",
    "excludes": "model_load_tokenization_warmup_serialization_file_IO",
    "decode_token_denominator": "generated_tokens_minus_prefill_root",
    "stage_timings": "diagnostic_only_not_summed_or_used_as_natural_kernel_cost",
}


class GenerationTimer:
    """Three device-complete boundaries; no per-round timing synchronization."""

    def __init__(self, synchronize, *, clock=None):
        self.synchronize = synchronize
        self.clock = clock or perf_counter

    def start(self):
        self.synchronize()
        self.started = self.clock()

    def prefill_done(self):
        self.synchronize()
        self.decode_started = self.clock()

    def finish(self, tokens, *, mode):
        self.synchronize()
        ended = self.clock()
        decode = ended - self.decode_started
        return {
            "generation_seconds": ended - self.started,
            "prefill_seconds": self.decode_started - self.started,
            "decode_seconds": decode,
            "generated_tokens": tokens,
            "decode_tokens": max(0, tokens - 1),
            "decode_tokens_per_second": (tokens - 1) / decode
            if tokens > 1 and decode > 0 else None,
            "device_complete_boundaries": True,
            "measurement_kind": "natural_reference" if mode == "performance"
            else "capture_instrumented",
        }


def parse_tensor_states(values):
    """Explicit qid:turn:round selections, bounded across the selected arms."""
    states = []
    for value in values:
        try:
            state = tuple(int(x) for x in value.split(":"))
        except ValueError as error:
            raise ValueError("tensor state must be qid:turn:round") from error
        if len(state) != 3 or any(x < 0 for x in state):
            raise ValueError("tensor state must be nonnegative qid:turn:round")
        if state in states:
            raise ValueError("duplicate tensor state")
        states.append(state)
    return states


def cpu_snapshot(value):
    """Own detached CPU storage; never retain a mutable view of live GPU KV."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_snapshot(x) for x in value)
    if value is None or type(value) in (int, float, bool, str):
        return value
    raise TypeError(f"unsupported snapshot type: {type(value)}")


def capture_state(decoder, *, ids, logits, hidden, features, consumed, root, plan=None):
    length = int(decoder.lengths[0])  # Native length metadata is CPU-resident.
    return cpu_snapshot({
        "boundary": "after_commit_before_next_round",
        "input_ids": ids,
        "pending_root": root,
        "target_kv_length": length,
        "target_kv": [data[..., :length, :] for data in decoder.backing],
        "drafter_kv": decoder.model.ea_layer.stable_kv,
        "pending_features": features,
        "drafter_consumed": consumed,
        "logits": logits,
        "hidden": hidden,
        "logits_scope": "path_expanded_verification" if plan is not None
        else "target_forward_that_produced_committed_AR_token",
        "plan": None if plan is None else {
            "tokens": plan.tokens, "retrieve": plan.retrieve,
            "positions": plan.positions, "mask": plan.mask,
        },
    })
