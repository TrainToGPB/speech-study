"""STEP 3 — whisper-base를 Zeroth-Korean 소량에 full fine-tune (M5·MPS).

평가는 학습 루프 밖(02/04)에서 수동으로 — MPS에서 Trainer 내부 generate 리스크를
피하고 루프를 단순·견고하게 유지한다. 저장은 끝에서 1회.
"""
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from common import (
    CKPT_DIR,
    DataCollatorSpeechSeq2SeqWithPadding,
    load_model,
    load_processor,
    load_splits,
    make_prepare,
    rule,
    seed_everything,
)

N_TRAIN = 1000
MAX_STEPS = 200
BATCH = 8


def main() -> None:
    seed_everything()
    processor = load_processor()
    model, device = load_model()
    model.config.use_cache = False  # 학습 시 캐시 off

    ds_train, _ = load_splits(n_train=N_TRAIN, n_eval=1)
    prepare = make_prepare(processor)
    ds_train = ds_train.map(prepare, remove_columns=ds_train.column_names, num_proc=1)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    rule(f"STEP 3 · train (n={len(ds_train)}, steps={MAX_STEPS}, batch={BATCH}, device={device})")
    args = Seq2SeqTrainingArguments(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=20,
        max_steps=MAX_STEPS,
        fp16=False,
        bf16=False,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=ds_train,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    trainer.train()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(CKPT_DIR))
    processor.save_pretrained(str(CKPT_DIR))
    print(f"\n저장 완료: {CKPT_DIR}")


if __name__ == "__main__":
    main()
