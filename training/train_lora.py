"""Train a Qwen3 ClaimCourt LoRA adapter on a local ROCm GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_length: int) -> None:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.items: list[dict[str, list[int]]] = []
        for row in rows:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(text, truncation=True, max_length=max_length)
            self.items.append({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, masks, labels = [], [], []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        masks.append(item["attention_mask"] + [0] * padding)
        labels.append(item["input_ids"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dataset = ChatDataset(args.data, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)

    started = time.perf_counter()
    losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()) * args.gradient_accumulation)
        print(f"epoch={epoch + 1} loss={sum(losses[-len(loader):]) / len(loader):.4f}", flush=True)

    torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metrics = {
        "base_model": str(args.model),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "examples": len(dataset),
        "epochs": args.epochs,
        "lora_rank": 16,
        "final_loss": losses[-1],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "gpu": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
    }
    (args.output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
