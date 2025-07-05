import yaml
import os
import gc
import logging
import psutil
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
import numpy as np
from datasets import Dataset, Audio, concatenate_datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open("whisper_finetune.yaml") as f:
    config = yaml.safe_load(f)

# Check GPU availability
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Device count: {torch.cuda.device_count()}")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

# 1. Load processor and model
logger.info("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(
    config["model_path"],
    language="hy",
    task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(config["model_path"])
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()


# Don't enable gradient checkpointing manually - we'll do it in training arguments

# 2. Create dataset loading function
def load_common_voice_from_files(data_path):
    logger.info(f"Loading dataset from {data_path}...")
    # Read metadata files
    train_df = pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
    dev_df = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(data_path, "test.tsv"), sep="\t")

    # Add full path to audio files
    clips_dir = os.path.join(data_path, "clips")
    train_df["audio_path"] = train_df["path"].apply(lambda x: os.path.join(clips_dir, x))
    dev_df["audio_path"] = dev_df["path"].apply(lambda x: os.path.join(clips_dir, x))
    test_df["audio_path"] = test_df["path"].apply(lambda x: os.path.join(clips_dir, x))

    # Filter out missing files
    train_df = train_df[train_df["audio_path"].apply(os.path.exists)]
    dev_df = dev_df[dev_df["audio_path"].apply(os.path.exists)]
    test_df = test_df[test_df["audio_path"].apply(os.path.exists)]

    # Print dataset sizes
    logger.info(f"Loaded {len(train_df)} training samples")
    logger.info(f"Loaded {len(dev_df)} validation samples")
    logger.info(f"Loaded {len(test_df)} test samples")

    # Create Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df[["audio_path", "sentence"]])
    dev_ds = Dataset.from_pandas(dev_df[["audio_path", "sentence"]])
    test_ds = Dataset.from_pandas(test_df[["audio_path", "sentence"]])

    # Cast to audio format
    train_ds = train_ds.cast_column("audio_path", Audio(sampling_rate=16000))
    dev_ds = dev_ds.cast_column("audio_path", Audio(sampling_rate=16000))
    test_ds = test_ds.cast_column("audio_path", Audio(sampling_rate=16000))

    return {
        "train": train_ds,
        "validation": dev_ds,
        "test": test_ds
    }


# 3. Load dataset
dataset = load_common_voice_from_files(config["dataset_path"])


# 4. Preprocessing function (idempotent version)
def prepare_dataset(batch):
    # Process audio features if not already done
    if "input_features" not in batch:
        audio = batch["audio_path"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features[0]

    # Process labels if not already done
    if "labels" not in batch:
        batch["labels"] = processor.tokenizer(
            batch["sentence"],
            padding="max_length",
            max_length=225,
            truncation=True
        ).input_ids
    return batch


def get_safe_core_count():
    total_cores = os.cpu_count()
    available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB

    # Conservative estimate: 0.5GB per core needed
    mem_constrained_cores = min(total_cores, int(available_mem * 2))

    # Use 75% of cores but respect memory limits
    return max(1, min(mem_constrained_cores, int(total_cores * 0.75)))


num_cores = get_safe_core_count()
# 5. Apply preprocessing to each split individually with caching
cache_dir = os.path.join(config["dataset_path"], "preprocessed_cache")
os.makedirs(cache_dir, exist_ok=True)

for split in ["train", "validation", "test"]:
    # Use the exact directory name that exists in your cache
    cache_path = os.path.join(cache_dir, f"{split}_preprocessed.arrow")

    # Check if valid cache exists
    if os.path.exists(cache_path):
        try:
            logger.info(f"Loading preprocessed {split} split from cache...")
            cached_ds = Dataset.load_from_disk(cache_path)

            # Verify cache contains required columns
            if "input_features" in cached_ds.column_names and "labels" in cached_ds.column_names:
                logger.info(f"Using valid cached preprocessed data for {split} split")
                dataset[split] = cached_ds
                continue
            else:
                logger.info(f"Cached data for {split} split is incomplete. Reprocessing...")
        except Exception as e:
            logger.error(f"Error loading cached dataset: {e}. Reprocessing...")

    logger.info(f"Preprocessing {split} split...")

    num_samples = len(dataset[split])
    logger.info(f"Processing {num_samples} samples...")

    # Use adaptive batch size based on dataset size
    batch_size = max(10, min(500, int(num_samples / 100)))
    logger.info(f"Using batch size: {batch_size}")

    processed = []

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch_ds = dataset[split].select(batch_indices)

        try:
            processed_batch = batch_ds.map(
                prepare_dataset,
                remove_columns=["audio_path", "sentence"],
                num_proc=num_cores
            )
            processed.append(processed_batch)
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size}: {str(e)}")
            continue

        del batch_ds
        gc.collect()

    if processed:
        # Concatenate all processed batches
        dataset[split] = concatenate_datasets(processed)

        # Set dataset format for PyTorch
        dataset[split] = dataset[split].with_format("pt")

        logger.info(f"Saving preprocessed {split} split to cache...")
        dataset[split].save_to_disk(cache_path)
        logger.info(f"Saved preprocessed {split} split to {cache_path}")
    else:
        logger.warning(f"Warning: No batches processed for {split} split!")

    del processed
    gc.collect()
    logger.info(f"{split} split preprocessing completed")


# 6. Create custom data collator
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# 7. Load WER metric
wer_metric = evaluate.load("wer")


# 8. Define compute_metrics function
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

    # Decode predictions
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Decode labels
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# 9. Training arguments
test_mode = False

# Convert all numeric parameters to appropriate types
lr = float(config["training"]["learning_rate"])
batch_size = int(config["training"]["per_device_train_batch_size"])
grad_accum = int(config["training"]["gradient_accumulation_steps"])
warmup = int(config["training"]["warmup_steps"])
max_steps = int(config["training"]["max_steps"])
logging_steps = int(config["training"]["logging_steps"])
eval_steps = int(config["training"]["eval_steps"])
save_steps = int(config["training"]["save_steps"])
fp16 = bool(config["training"]["fp16"])

# Reduced batch sizes for stability
per_device_train_batch_size = max(1, batch_size // 4) if not test_mode else 2
per_device_eval_batch_size = max(1, 8 // 4) if not test_mode else 2

training_args = Seq2SeqTrainingArguments(
    save_total_limit=3,  # Keep only last 3 checkpoints
    logging_dir=os.path.join(config["training"]["output_dir"], "logs"),
    output_dir=config["training"]["output_dir"],
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=1 if test_mode else grad_accum,
    learning_rate=lr,
    warmup_steps=100 if test_mode else warmup,
    max_steps=2 if test_mode else max_steps,
    logging_steps=1 if test_mode else logging_steps,
    eval_strategy="steps" if test_mode else config["training"]["evaluation_strategy"],
    eval_steps=1 if test_mode else eval_steps,
    save_steps=1 if test_mode else save_steps,
    fp16=fp16 and torch.cuda.is_available(),  # Only enable if GPU available
    predict_with_generate=True,
    generation_max_length=225,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,  # Enable automatic gradient checkpointing
    max_grad_norm=1.0,  # Gradient clipping
)

# 10. Create Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 11. Start training
logger.info("Starting training...")
try:
    # Free up memory before training
    torch.cuda.empty_cache()
    gc.collect()

    trainer.train()

    # 13. Save final model
    output_dir = config["training"]["output_dir"]
    trainer.save_model(os.path.join(output_dir, "final"))
    processor.save_pretrained(os.path.join(output_dir, "final"))
    logger.info(f"Training complete! Model saved to {output_dir}/final")

except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    # Save current state for debugging
    try:
        trainer.save_model(os.path.join(config["training"]["output_dir"], "interrupted"))
        logger.info("Model saved in interrupted state for debugging")
    except Exception as save_error:
        logger.error(f"Failed to save interrupted model: {str(save_error)}")