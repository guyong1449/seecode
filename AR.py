"""Paired raw-token Spec-Bench checks for the Qwen3 common-target ports."""

from __future__ import annotations

import argparse
import json
import math
import copy
import subprocess
import os
import sys
import time
from pathlib import Path

from hsds.datasets.spec_bench import SpecBenchCaseId, load_spec_bench
from hsds.validation.exactness import compare_token_traces, parse_token_trace
from hsds.validation.divergence import record_failure
from methods import ar_corpus as corpus
from methods import common_target_port as port
from methods.common import sha256_file, write_json_atomic

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1] / "configs/common_target_specbench.json"
)
DATASET_PATH = Path(
    "/dkucc/home/rw335/heterogeneous-sd-scheduling/upstreams/Spec-Bench-fd2c1cd7/data/spec_bench/question.jsonl"
)


class ARTrajectoryArchive:
    """Durable per-turn AR evidence; completion means coverage, not SD exactness."""

    top_k = 10

    def __init__(self, directory, metadata):
        self.directory = directory
        self.expected = {tuple(x) for x in metadata["expected_case_ids"]}
        if len(self.expected) != len(metadata["expected_case_ids"]):
            raise port.CommonTargetPortError("duplicate archive case identity")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "turns").mkdir()
        self.context = {}
        self.limited = []
        self.entries = []
        self.seen = set()
        self.metadata = dict(
            metadata,
            state="collecting",
            top_k=self.top_k,
            purpose="offline AR candidate analysis; no SD or performance claim",
            logits_scope="generated AR positions only; no rejected tree nodes or hidden states",
            order_semantics="serialized result order; not a recorded serving arrival trace",
        )
        self.checkpoint()

    def checkpoint(self):
        write_json_atomic(
            self.directory / "manifest.json",
            dict(
                self.metadata,
                completed_cases=len(self.entries),
                turns=self.entries,
                context_limited_cases=self.limited,
            ),
        )

    def __call__(self, trace, choice, tokenizer, batch_index):
        identity = tuple(
            trace[k]
            for k in ("dataset_sha256", "question_id", "turn_index", "choice_index")
        )
        if identity not in self.expected or identity in self.seen:
            raise port.CommonTargetPortError("unexpected or duplicate archived trace")
        parse_token_trace(trace)
        tokens = trace["raw_generated_token_ids"]
        if choice.logprobs is None or len(choice.logprobs) != len(tokens):
            raise port.CommonTargetPortError("AR logprob/token coverage differs")
        steps = []
        for index, (token, probabilities) in enumerate(
            zip(tokens, choice.logprobs, strict=True)
        ):
            if (
                probabilities is None
                or token not in probabilities
                or len(probabilities) < self.top_k
            ):
                raise port.CommonTargetPortError("AR sampled token or top-k is missing")
            values = []
            for candidate, value in probabilities.items():
                if (
                    type(candidate) is not int
                    or candidate < 0
                    or not math.isfinite(value.logprob)
                    or type(value.rank) is not int
                    or value.rank < 1
                ):
                    raise port.CommonTargetPortError("invalid AR top-k entry")
                values.append(
                    dict(
                        token_id=candidate,
                        logprob=float(value.logprob),
                        rank=value.rank,
                    )
                )
            values.sort(key=lambda x: (x["rank"], x["token_id"]))
            steps.append(
                dict(
                    generation_index=index,
                    prefix_length=len(trace["prompt_token_ids"]) + index,
                    sampled_token_id=token,
                    candidates=values,
                )
            )
        filename = f"turns/{trace['question_id']}-{trace['turn_index']}-{trace['choice_index']}.json"
        record = dict(
            trace=trace,
            batch_index=batch_index,
            sequence_index=len(self.entries),
            generated_steps=steps,
            **self.context,
            prompt_text=tokenizer.decode(
                trace["prompt_token_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            output_text=tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
        )
        if self.context:
            record.update(
                finish_reason=choice.finish_reason,
                answer_complete=trace["stop_event"]["kind"] == "eos_token",
                length_capped=trace["stop_event"]["kind"] == "max_output_boundary",
                context_limited=False,
                input_tokens=len(trace["prompt_token_ids"]),
                generated_tokens=len(tokens),
                history_output_text=tokenizer.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
            )
        write_json_atomic(self.directory / filename, record)
        self.entries.append(
            dict(
                case_id=list(identity),
                path=filename,
                sha256=sha256_file(self.directory / filename),
                generated_tokens=len(tokens),
                stop_event=trace["stop_event"],
            )
        )
        self.seen.add(identity)
        self.checkpoint()
        print(
            f"ar_trajectory_progress={len(self.entries)}/{len(self.expected)}",
            flush=True,
        )

    def finish(self, payload):
        if self.seen | {
            tuple(c["case_id"]) for c in self.limited
        } != self.expected or len(payload["traces"]) != len(self.seen):
            raise port.CommonTargetPortError("AR archive coverage is incomplete")
        by_id = {
            tuple(
                r[k]
                for k in ("dataset_sha256", "question_id", "turn_index", "choice_index")
            ): r
            for r in payload["traces"]
        }
        for entry in self.entries:
            path = self.directory / entry["path"]
            if (
                sha256_file(path) != entry["sha256"]
                or json.loads(path.read_text())["trace"]
                != by_id[tuple(entry["case_id"])]
            ):
                raise port.CommonTargetPortError("AR archive bytes or trace differ")
        self.metadata.update(
            state="complete" if not self.limited else "context_limited",
            runtime=payload["runtime"],
            runtime_controls=payload["runtime_controls"],
            generated_tokens=sum(e["generated_tokens"] for e in self.entries),
            eos_turns=sum(e["stop_event"]["kind"] == "eos_token" for e in self.entries),
            length_capped_turns=sum(
                e["stop_event"]["kind"] == "max_output_boundary" for e in self.entries
            ),
        )
        if "plan" in self.metadata:
            self.metadata.update(
                execution_complete=True, coverage_complete=not self.limited
            )
        self.checkpoint()
        if "plan" in self.metadata:
            corpus.validate(self.directory, require_complete=False)
        manifest_hash = sha256_file(self.directory / "manifest.json") + "\n"
        if not self.limited:
            (self.directory / "COMPLETE").write_text(manifest_hash)
        if "plan" in self.metadata:
            (self.directory / "EXECUTION_COMPLETE").write_text(manifest_hash)
            try:
                corpus.validate(self.directory)
            except Exception:
                for name in ("COMPLETE", "EXECUTION_COMPLETE"):
                    (self.directory / name).unlink(missing_ok=True)
                raise


def stop_event(
    tokens: list[int], max_tokens: int, eos_ids: list[int], finish_reason: str
) -> dict:
    if not tokens or len(tokens) > max_tokens:
        raise port.CommonTargetPortError("empty or over-length output")
    positions = [i for i, token in enumerate(tokens) if token in eos_ids]
    if positions:
        if positions != [len(tokens) - 1] or finish_reason != "stop":
            raise port.CommonTargetPortError(
                "output extends past EOS or finish reason differs"
            )
        return {
            "kind": "eos_token",
            "position": len(tokens) - 1,
            "matched_token_ids": tokens[-1:],
        }
    if len(tokens) != max_tokens or finish_reason != "length":
        raise port.CommonTargetPortError("output has no legal terminal boundary")
    return {
        "kind": "max_output_boundary",
        "position": len(tokens),
        "matched_token_ids": [],
    }


def select_questions(dataset, question_ids):
    if len(set(question_ids)) != len(question_ids) or not question_ids:
        raise port.CommonTargetPortError("question IDs must be nonempty and unique")
    selected = [q for q in dataset.questions if q.question_id in question_ids]
    if len(selected) != len(question_ids):
        raise port.CommonTargetPortError("question IDs are absent from frozen dataset")
    return selected


def generate_questions(
    vllm,
    engine,
    tokenizer,
    config,
    questions,
    protocol,
    synchronize,
    trace_sink=None,
    request_groups=None,
):
    traces, batches = [], []
    groups = request_groups or [
        dict(
            group_id=start // config.max_num_seqs,
            question_ids=[
                q.question_id for q in questions[start : start + config.max_num_seqs]
            ],
        )
        for start in range(0, len(questions), config.max_num_seqs)
    ]
    for group_spec in groups:
        group = [q for q in questions if q.question_id in group_spec["question_ids"]]
        histories = {q.question_id: [] for q in group}
        history_capped = {q.question_id: False for q in group}
        for turn in range(max(len(q.turns) for q in group)):
            active = [q for q in group if turn < len(q.turns)]
            prompts = []
            overflows = []
            for q in active:
                messages = histories[q.question_id]
                messages.append({"role": "user", "content": q.turns[turn]})
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if type(ids) is not list or any(
                    type(token) is not int for token in ids
                ):
                    raise port.CommonTargetPortError(
                        "chat template did not return integer token IDs"
                    )
                if not ids or len(ids) + config.max_tokens > config.max_model_len:
                    if trace_sink is None or "plan" not in trace_sink.metadata:
                        raise port.CommonTargetPortError(
                            f"prompt exceeds context without truncation: {q.question_id}/{turn}"
                        )
                    overflows.append(
                        dict(question_id=q.question_id, input_tokens=len(ids))
                    )
                prompts.append({"prompt_token_ids": ids})
            if overflows:
                # Keep the original call membership; no partial group is submitted.
                for q in group:
                    for blocked_turn in range(turn, len(q.turns)):
                        trace_sink.limited.append(
                            dict(
                                case_id=[
                                    protocol["dataset_sha256"],
                                    q.question_id,
                                    blocked_turn,
                                    0,
                                ],
                                request_group_id=group_spec["group_id"],
                                max_output_tokens=config.max_tokens,
                                context_limited=True,
                                generated_tokens=0,
                                answer_complete=False,
                                reason="context_capacity"
                                if blocked_turn == turn
                                and q.question_id
                                in {x["question_id"] for x in overflows}
                                else "original_group_call_blocked"
                                if blocked_turn == turn
                                else "upstream_turn_unavailable",
                                overflow_members=overflows,
                                messages=copy.deepcopy(histories[q.question_id])
                                if blocked_turn == turn
                                else None,
                                prompt_token_ids=next(
                                    (
                                        p["prompt_token_ids"]
                                        for aq, p in zip(active, prompts, strict=True)
                                        if aq.question_id == q.question_id
                                    ),
                                    None,
                                )
                                if blocked_turn == turn
                                else None,
                            )
                        )
                trace_sink.checkpoint()
                break
            sampling = vllm.SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=config.max_tokens,
                ignore_eos=False,
                seed=0,
                stop_token_ids=protocol["eos_token_ids"],
                detokenize=False,
                **({"logprobs": trace_sink.top_k} if trace_sink is not None else {}),
            )
            synchronize()
            begin = time.perf_counter()
            outputs = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
            synchronize()
            elapsed = time.perf_counter() - begin
            if len(outputs) != len(active):
                raise port.CommonTargetPortError("wrong output count")
            batches.append(
                {
                    "question_ids": [q.question_id for q in active],
                    "turn_index": turn,
                    "elapsed_seconds_diagnostic": elapsed,
                }
            )
            for q, prompt, output in zip(active, prompts, outputs, strict=True):
                if (
                    list(output.prompt_token_ids) != prompt["prompt_token_ids"]
                    or len(output.outputs) != 1
                ):
                    raise port.CommonTargetPortError("output/prompt identity differs")
                choice = output.outputs[0]
                ids = list(choice.token_ids)
                record = {
                    "schema_version": 1,
                    "dataset_sha256": protocol["dataset_sha256"],
                    "question_id": q.question_id,
                    "turn_index": turn,
                    "choice_index": 0,
                    "status": "complete",
                    "prompt_token_ids": prompt["prompt_token_ids"],
                    "raw_generated_token_ids": ids,
                    "stop_event": stop_event(
                        ids,
                        config.max_tokens,
                        protocol["eos_token_ids"],
                        choice.finish_reason,
                    ),
                }
                parse_token_trace(record)
                traces.append(record)
                if trace_sink is not None:
                    trace_sink.context = dict(
                        task=q.task_id,
                        source_category=q.source_category,
                        source_user_turns=list(q.turns),
                        messages=copy.deepcopy(histories[q.question_id]),
                        template_kwargs=corpus.TEMPLATE,
                        request_group_id=group_spec["group_id"],
                        request_group_question_ids=group_spec["question_ids"],
                        generation_call_index=len(batches) - 1,
                        generation_call_question_ids=[aq.question_id for aq in active],
                        max_output_tokens=config.max_tokens,
                        eos_token_ids=protocol["eos_token_ids"],
                        history_contains_length_capped=history_capped[q.question_id],
                        trajectory_version=trace_sink.metadata.get("plan", {}).get(
                            "version", "legacy_smoke"
                        ),
                        decode_kwargs=dict(
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        history_decode_kwargs=dict(
                            skip_special_tokens=True, clean_up_tokenization_spaces=False
                        ),
                        probability_semantics="P(next token | prompt_token_ids + raw_generated_token_ids[:generation_index]); top-10, not full vocabulary",
                    )
                    trace_sink(record, choice, tokenizer, len(batches) - 1)
                history_capped[q.question_id] |= (
                    record["stop_event"]["kind"] == "max_output_boundary"
                )
                histories[q.question_id].append(
                    {
                        "role": "assistant",
                        "content": tokenizer.decode(
                            ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                )
    return traces, batches


def compare_artifacts(oracle, candidate):
    if oracle["mode"] != "ar":
        raise port.CommonTargetPortError("oracle must be an independent AR run")
    for field in (
        "protocol_sha256",
        "question_ids",
        "max_tokens",
        "batch_size",
        "max_model_len",
        "target_revision",
        "source_file_hashes",
        "runtime_controls",
    ):
        if oracle[field] != candidate[field]:
            raise port.CommonTargetPortError(f"comparison contract differs: {field}")
    expected = [
        SpecBenchCaseId(*identity) for identity in candidate["expected_case_ids"]
    ]
    if oracle["expected_case_ids"] != candidate["expected_case_ids"]:
        raise port.CommonTargetPortError("expected case identities differ")
    matched = compare_token_traces(expected, oracle["traces"], candidate["traces"])
    return {
        "matched_cases": len(matched),
        "exact_on_selected_cases": True,
        "full_specbench": len(matched) == 560,
        "speed_claim": False,
        "accounting_validated": False,
    }


def run(
    config,
    source_root,
    question_ids,
    output,
    oracle_path=None,
    trajectory_directory=None,
    archive_plan=None,
    group_ids=None,
):
    if trajectory_directory is not None and config.mode != "ar":
        raise port.CommonTargetPortError(
            "trajectory collection requires independent AR"
        )
    protocol = json.loads(PROTOCOL_PATH.read_text())
    dataset = load_spec_bench(DATASET_PATH)
    if (
        dataset.sha256 != protocol["dataset_sha256"]
        or port.TARGET_REVISION != protocol["target_revision"]
        or config.seed != protocol["seed"]
    ):
        raise port.CommonTargetPortError("runtime differs from the paired protocol")
    if oracle_path is not None and oracle_path.resolve() == output.resolve():
        raise port.CommonTargetPortError("output must not overwrite the AR oracle")
    request_groups = None
    plan = None
    if archive_plan is not None:
        if trajectory_directory is None or config.mode != "ar":
            raise port.CommonTargetPortError(
                "archive plan requires AR trajectory directory"
            )
        plan = json.loads(archive_plan.read_text())
        corpus.check_plan(plan, config, source_root)
        request_groups = plan["groups"]
        if group_ids is not None:
            corpus.require(
                len(group_ids) == len(set(group_ids))
                and set(group_ids).issubset({g["group_id"] for g in request_groups}),
                "invalid retry groups",
            )
            request_groups = [g for g in request_groups if g["group_id"] in group_ids]
        question_ids = [q for g in request_groups for q in g["question_ids"]]
    questions = select_questions(dataset, question_ids)
    if (
        sha256_file(config.target_model_path / "tokenizer_config.json")
        != protocol["tokenizer_config_sha256"]
    ):
        raise port.CommonTargetPortError("Qwen3 tokenizer template differs")
    expected = [
        list(SpecBenchCaseId(dataset.sha256, q.question_id, turn, 0).as_tuple())
        for q in questions
        for turn in range(len(q.turns))
    ]
    if output.exists():
        raise port.CommonTargetPortError("output must not overwrite earlier evidence")
    engine = None
    payload = None
    archive = None
    complete = False
    if trajectory_directory is not None:
        archive = ARTrajectoryArchive(
            trajectory_directory,
            {
                "expected_case_ids": expected,
                "protocol": protocol,
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "target_revision": port.TARGET_REVISION,
                "target_model_path": str(config.target_model_path),
                "engine_kwargs": config.engine_kwargs(),
                "interpreter": sys.executable,
                "source_file_hashes": {
                    name: sha256_file(source_root / name) for name in port.SOURCE_FILES
                },
                "collector_sha256": sha256_file(Path(__file__)),
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "question_ids": [q.question_id for q in questions],
                "case_metadata": [
                    {
                        "question_id": q.question_id,
                        "task": q.task_id,
                        "source_category": q.source_category,
                    }
                    for q in questions
                ],
                "max_output_tokens": config.max_tokens,
            },
        )
    if plan is not None:
        archive.metadata.update(
            schema="hsds.ar_corpus.attempt.v1",
            plan=plan,
            plan_sha256=corpus.digest(plan),
            coverage_complete=False,
            execution_complete=False,
            project_commit=subprocess.check_output(
                ["git", "-C", str(corpus.ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            collector_files={
                str(p.relative_to(corpus.ROOT)): sha256_file(p)
                for p in [
                    Path(__file__).resolve(),
                    corpus.ROOT / "methods/ar_corpus.py",
                    corpus.ROOT / "methods/common_target_port.py",
                    corpus.ROOT / "run_method.sh",
                ]
            },
            startup_argv=sys.argv,
            log_paths=subprocess.check_output(
                ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"], text=True
            ).strip(),
            selected_group_ids=[g["group_id"] for g in request_groups],
        )
        archive.checkpoint()
    try:
        vllm, engine, policy, runtime = port._load_runtime(config, source_root)
        port._validate_observed_config(engine, config)
        policy.reset_tetris_receipts()
        policy.reset_turbospec_dsd()
        tokenizer = engine.get_tokenizer()
        torch = port.importlib.import_module("torch")
        if plan is not None:
            archive.metadata["runtime_identity"] = corpus.runtime_metadata(
                torch, tokenizer, source_root
            )
            archive.checkpoint()
        before = port.read_counters(engine) if config.mode != "ar" else None
        traces, batches = generate_questions(
            vllm,
            engine,
            tokenizer,
            config,
            questions,
            protocol,
            torch.cuda.synchronize,
            **({"trace_sink": archive} if archive is not None else {}),
            **(
                {"request_groups": request_groups} if request_groups is not None else {}
            ),
        )
        counters = (
            port._counter_delta(port.read_counters(engine), before)
            if before is not None
            else None
        )
        payload = {
            "artifact_kind": "qwen3_common_target_specbench_token_check",
            "mode": config.mode,
            "dsd_feedback": list(getattr(policy, "TURBOSPEC_DSD_FEEDBACK", ()))
            if config.mode == "turbospec_dsd"
            else [],
            "protocol": protocol,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "question_ids": [q.question_id for q in questions],
            "expected_case_ids": expected,
            "max_tokens": config.max_tokens,
            "batch_size": config.max_num_seqs,
            "max_model_len": config.max_model_len,
            "target_revision": port.TARGET_REVISION,
            "source_file_hashes": {
                name: sha256_file(source_root / name) for name in port.SOURCE_FILES
            },
            "engine_kwargs": config.engine_kwargs(),
            "runtime": runtime,
            "runtime_controls": {
                "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
            },
            "traces": traces,
            "batches": batches,
            "counter_delta": counters,
            "policy_receipts": list(
                policy.TETRIS_RECEIPTS
                if config.mode == "tetris"
                else policy.TURBOSPEC_DSD_RECEIPTS
                if config.mode == "turbospec_dsd"
                else []
            ),
            "self_draft_diagnostic": config.mode in {"tetris", "draft_fixed"}
            and config.target_model_path == config.draft_model_path,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "claims": {
                "gpu_execution": True,
                "paper_fidelity": False,
                "canonical_correctness": False,
                "accounting_validated": False,
                "speed_claim": False,
            },
            "comparison": None,
        }
        write_json_atomic(output, payload)
        try:
            validate_live_policy(config, policy, counters)
            if oracle_path is not None:
                payload["comparison"] = compare_artifacts(
                    json.loads(oracle_path.read_text()), payload
                )
                payload["oracle_sha256"] = sha256_file(oracle_path)
        except Exception as error:
            payload["comparison"] = {
                "exact_on_selected_cases": False,
                "error": str(error),
            }
            write_json_atomic(output, payload)
            if oracle_path is not None:
                record_failure(oracle_path, output, error)
            raise
        write_json_atomic(output, payload)
        complete = True
        return payload
    finally:
        if engine is not None:
            port._shutdown(engine)
        if complete and archive is not None:
            archive.finish(payload)


def validate_live_policy(config, policy, counters):
    """Token equality alone must not admit an inactive or wrong policy arm."""
    if config.mode == "tetris":
        if not any(row.get("active") is True for row in policy.TETRIS_RECEIPTS):
            raise port.CommonTargetPortError("TETRIS selector did not activate")
    elif policy.TETRIS_RECEIPTS:
        raise port.CommonTargetPortError("non-TETRIS arm invoked the selector")
    if config.mode == "turbospec_dsd":
        if not policy.TURBOSPEC_DSD_RECEIPTS or not policy.TURBOSPEC_DSD_FEEDBACK:
            raise port.CommonTargetPortError("DSD policy/feedback did not activate")
    elif policy.TURBOSPEC_DSD_RECEIPTS:
        raise port.CommonTargetPortError("non-DSD arm invoked the DSD policy")
    if config.mode in {"tetris", "draft_fixed", "turbospec_fixed"}:
        if counters is None or counters["draft_tokens"] < 1:
            raise port.CommonTargetPortError("fixed/TETRIS arm produced no drafts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("ar", "turbospec_fixed", "turbospec_dsd", "draft_fixed", "tetris"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--trajectory-directory", type=Path)
    parser.add_argument("--archive-plan", type=Path)
    parser.add_argument("--group-ids", type=int, nargs="+")
    parser.add_argument("--fixed-k", type=int, default=3)
    parser.add_argument("--draft-model-path", type=Path)
    parser.add_argument("--dsd-profile-path", type=Path)
    parser.add_argument("--dsd-profile-sha256")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    config = port.CommonTargetConfig(
        mode=args.mode,
        max_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        fixed_k=args.fixed_k,
        draft_model_path=args.draft_model_path,
        dsd_profile_path=args.dsd_profile_path,
        dsd_profile_sha256=args.dsd_profile_sha256,
    )
    ids = list(range(81, 561)) if args.all_questions else protocol["smoke_question_ids"]
    run(
        config,
        args.source_root,
        ids,
        args.output,
        args.oracle,
        args.trajectory_directory,
        args.archive_plan,
        args.group_ids,
    )


if __name__ == "__main__":
    main()
