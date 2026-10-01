"""Fine-tune Qwen3-TTS-12Hz Base on accented multi-speaker data, on Apple MPS.

Adapted from QwenLM/Qwen3-TTS finetuning/sft_12hz.py (Apache-2.0):
- runs on MPS with SDPA attention (no CUDA / flash-attn)
- multi-speaker: each utterance is conditioned on its own speaker's averaged
  x-vector (precomputed), so the model learns the *accent* across speakers
- the saved custom voice (id 3000) is a blend of the speaker embeddings, a
  new voice that isn't any one of the real speakers

usage: train_wiseguy.py --data train_with_codes.jsonl --spk spk_emb.pt --out ckpt ...
"""
import argparse, json, math, os, random, shutil, sys, time

import torch
import torch.nn.functional as F
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

sys.path.insert(0, str(paths.QWEN_SRC / "finetuning"))
from dataset import TTSDataset  # noqa: E402
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402


class SpeakerDataset(TTSDataset):
    """TTSDataset without per-item reference mels: we pass the speaker index."""

    def __init__(self, data, processor, config, spk_index):
        super().__init__(data, processor, config)
        self.spk_index = spk_index

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_ids = self._tokenize_texts(self._build_assistant_text(item["text"]))
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": torch.tensor(item["audio_codes"], dtype=torch.long),
            "ref_mel": torch.zeros(1, 1, 128),  # unused
            "spk": self.spk_index[item["speaker"]],
        }

    def collate_fn(self, batch):
        out = super().collate_fn(batch)
        out.pop("ref_mels")
        out["spk"] = torch.tensor([b["spk"] for b in batch], dtype=torch.long)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(paths.CHECKPOINT), help="start from the shipped voice (default) or a Base model")
    ap.add_argument("--data", required=True, nargs="+", help="one or more *_codes.jsonl (e.g. new data + models/finetune/replay_codes.jsonl)")
    ap.add_argument("--spk", required=True, help="torch file: {'names': [...], 'emb': Tensor[n, d], 'voice': Tensor[d]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speaker_name", default="wiseguy")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=180, help="skip utterances longer than this many codec frames")
    ap.add_argument("--freeze_text", action="store_true", help="freeze the text embedding")
    ap.add_argument("--save_every_epoch", action="store_true")
    args = ap.parse_args()

    dev = torch.device("mps")
    torch.manual_seed(0)
    random.seed(0)

    qwen = Qwen3TTSModel.from_pretrained(args.init, dtype=torch.float32, attn_implementation="sdpa")
    model = qwen.model.to(dev)
    config = AutoConfig.from_pretrained(args.init)

    spk = torch.load(args.spk)
    spk_emb = spk["emb"].to(dev, torch.float32)
    spk_index = {n: i for i, n in enumerate(spk["names"])}

    data = [json.loads(l) for f in args.data for l in open(f)]
    data = [d for d in data if d["speaker"] in spk_index and len(d["audio_codes"]) <= args.max_frames]
    random.shuffle(data)
    print(f"{len(data)} utterances, speakers: {spk['names']}", flush=True)
    ds = SpeakerDataset(data, qwen.processor, config, spk_index)
    # batches of similar length: fewer distinct tensor shapes for the MPS allocator to cache
    order = sorted(range(len(data)), key=lambda i: len(data[i]["audio_codes"]))
    batches = [order[i : i + args.batch] for i in range(0, len(order), args.batch)]

    class Buckets(torch.utils.data.Sampler):
        def __iter__(self):
            random.shuffle(batches)
            return iter(batches)

        def __len__(self):
            return len(batches)

    dl = torch.utils.data.DataLoader(ds, batch_sampler=Buckets(), collate_fn=ds.collate_fn, num_workers=0)

    if model.speaker_encoder is not None:  # absent when warm-starting from a custom-voice checkpoint
        for p in model.speaker_encoder.parameters():
            p.requires_grad_(False)
    if args.freeze_text:
        model.talker.model.text_embedding.weight.requires_grad_(False)
    try:
        model.talker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing on", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"no gradient checkpointing: {e}", flush=True)
    model.talker.model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(dl) / args.accum) * args.epochs
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        run = 0.0
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            input_ids, codec_ids = b["input_ids"], b["codec_ids"]
            codec_mask = b["codec_mask"]
            text_emb = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * b["text_embedding_mask"]
            codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) * b["codec_embedding_mask"]
            codec_emb[:, 6, :] = spk_emb[b["spk"]]
            emb = text_emb + codec_emb
            for k in range(1, 16):
                ek = model.talker.code_predictor.get_input_embeddings()[k - 1](codec_ids[:, :, k])
                emb = emb + ek * codec_mask.unsqueeze(-1)
            out = model.talker(
                inputs_embeds=emb[:, :-1, :],
                attention_mask=b["attention_mask"][:, :-1],
                output_hidden_states=True,
            )
            # The upstream script passes pre-shifted labels into a loss that shifts
            # again (off by one frame); compute both losses aligned by hand.
            logits = out.logits
            main_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                        b["codec_0_labels"][:, 1:].reshape(-1), ignore_index=-100)
            # The code predictor is conditioned on the talker state that *predicted*
            # codebook 0, i.e. the position before each code (upstream uses the code's
            # own position, one step late, which garbles generation after fine-tuning).
            hidden = out.hidden_states[0][-1][codec_mask[:, 1:]]
            sub_logits, _ = model.talker.forward_sub_talker_finetune(codec_ids[codec_mask], hidden)
            sub_loss = F.cross_entropy(sub_logits.reshape(-1, sub_logits.shape[-1]).float(),
                                       codec_ids[codec_mask][:, 1:].reshape(-1))
            loss = main_loss + 0.3 * sub_loss
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            (loss / args.accum).backward()
            run += loss.item()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                torch.mps.empty_cache()
                step += 1
                if step % 10 == 0:
                    el = time.time() - t0
                    print(f"epoch {epoch} step {step}/{total_steps} loss {run / (args.accum * 10):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el / step:.1f}s/step "
                          f"mem {torch.mps.driver_allocated_memory() / 1e9:.1f}GB", flush=True)
                    run = 0.0
        if args.save_every_epoch or epoch == args.epochs - 1:
            save(model, args, spk["voice"], f"epoch{epoch}")


def save(model, args, voice, tag):
    out = os.path.join(args.out, tag)
    shutil.copytree(args.init, out, dirs_exist_ok=True)
    cfg = json.load(open(os.path.join(args.init, "config.json")))
    cfg["tts_model_type"] = "custom_voice"
    cfg["talker_config"]["spk_id"] = {args.speaker_name: 3000}
    cfg["talker_config"]["spk_is_dialect"] = {args.speaker_name: False}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items() if not k.startswith("speaker_encoder")}
    sd["talker.model.codec_embedding.weight"][3000] = voice.to(torch.bfloat16)
    from safetensors.torch import save_file

    save_file(sd, os.path.join(out, "model.safetensors"))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
