"""Charter behavioral MCQ — swap-debiased first-token logprob (no judge, no sampling).

Presents each item at all 4 option rotations, reads the first-token logprob over
the displayed letters A-D, and sums each ORIGINAL option's logprob across the four
display slots it occupies. The position prior (~1 nat on these SmolLM SFT
checkpoints — big enough to swamp content in any single arrangement) cancels by
construction; argmax = position-free choice. See README.md for why this is the
only valid options-in-context protocol at this model scale, and for the audited
caveats on interpretation.

Usage (run from charter_mcq/, normally via slurm/eval_charter_mcq.sh):
    python run_eval.py model.name=<alias> model.pretrained=<path-or-repo>
    python run_eval.py ... testing=true testing_limit=16

The slurm wrapper installs the per-checkpoint TRAINING chat template (same
mechanism as every other component); this script just calls
tokenizer.apply_chat_template. Deterministic forward passes only — no decoding
hyperparameters, no judge, so results are exactly reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mreval.results import result_filename, save_results  # noqa: E402

from charter_mcq.scoring import (  # noqa: E402
    SCORER_ID,
    aggregate,
    build_result_rows,
    gold_index,
    letter_prompt,
    letter_token_ids,
    predict,
    swap_scores,
    validate_items,
)

JUDGE_META = {
    "id": SCORER_ID,
    "provider": "rule-based",
    "model": "first-token-logprob",
    "kind": "rule",
}
DECODING = {"strategy": "greedy"}  # deterministic forward passes; no sampling at all


def load_items(cfg: DictConfig) -> list[dict]:
    ds = load_dataset(cfg.dataset.repo, split=cfg.dataset.split,
                      revision=cfg.dataset.get("revision"))
    items = [dict(r) for r in ds]
    validate_items(items)
    if cfg.testing:
        items = items[: int(cfg.testing_limit)]
        logger.warning(f"testing=true -> truncated to {len(items)} items")
    return items


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.model.pretrained:
        raise ValueError("model.pretrained is required (registry alias resolution "
                         "happens in slurm/eval_charter_mcq.sh)")
    model_name = cfg.model.name or Path(str(cfg.model.pretrained)).name

    items = load_items(cfg)
    logger.info(f"{len(items)} items from {cfg.dataset.repo}@{cfg.dataset.get('revision')}")

    tok = AutoTokenizer.from_pretrained(cfg.model.pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained, dtype=getattr(torch, cfg.model.dtype)).to("cuda").eval()
    letter_ids = letter_token_ids(tok)

    def render(item: dict, rot: int) -> str:
        # The training chat template is installed process-wide by the slurm
        # wrapper (model_registry.sh -> .pth hook); no per-call override here.
        return tok.apply_chat_template(
            [{"role": "user", "content": letter_prompt(item, rot)}],
            add_generation_prompt=True, tokenize=False)

    logger.info(f"prompt tail (rot0, item0): {render(items[0], 0)[-120:]!r}")

    per_item: dict[str, dict] = {}
    pos_lp_sum = [0.0] * 4  # raw per-display-slot mean logprob (primacy diagnostic)
    with torch.no_grad():
        for k, item in enumerate(items):
            rows = []
            for rot in range(4):
                ids = tok(render(item, rot), return_tensors="pt",
                          add_special_tokens=False).input_ids.to("cuda")
                lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
                row = [torch.logsumexp(lp[v], dim=0).item() for v in letter_ids]
                rows.append(row)
                for j in range(4):
                    pos_lp_sum[j] += row[j]
            scores = swap_scores(rows)
            order = sorted(range(4), key=lambda o: -scores[o])
            per_item[item["id"]] = {
                "pred": predict(scores),
                "gold": gold_index(item),
                "scores": [round(s, 4) for s in scores],
                "margin": round(scores[order[0]] - scores[order[1]], 4),
            }
            if (k + 1) % 100 == 0:
                hits = sum(v["pred"] == v["gold"] for v in per_item.values())
                logger.info(f"{k + 1}/{len(items)}  acc so far {hits / (k + 1):.3f}")

    band_of = {it["id"]: it["e4b_blind_band"] for it in items}
    metrics = aggregate(per_item, band_of)
    metrics["per_position_mean_logprob"] = [round(x / (4 * len(items)), 3) for x in pos_lp_sum]
    logger.info(f"acc {metrics['correct']} = {metrics['acc']:.1%}  bands "
                + "  ".join(f"{b}:{a:.0%}" for b, a in metrics["band_acc"].items()))
    logger.info(f"per-display-slot mean logprob (primacy tell): "
                f"{metrics['per_position_mean_logprob']}")

    source = f"{cfg.dataset.repo}@{(cfg.dataset.get('revision') or 'main')[:12]}"
    out_dir = Path(cfg.output_dir)
    fname = result_filename(model_name, "charter_mcq", JUDGE_META["id"], "greedy")
    path = save_results(
        out_dir / fname,
        model=model_name,
        benchmark="charter_mcq",
        results=build_result_rows(per_item, {it["id"]: it for it in items}, source=source),
        decoding=DECODING,
        judge_meta=JUDGE_META,
        extra={"dataset": {"repo": cfg.dataset.repo, "split": cfg.dataset.split,
                           "revision": cfg.dataset.get("revision")}},
        metrics=metrics,
    )
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
