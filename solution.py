from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = "/content/drive/MyDrive"
TRAIN_PATH  = f"{PROJECT_DIR}/train.csv"
TEST_PATH   = f"{PROJECT_DIR}/test_public.csv"
OUT_DIR     = f"{PROJECT_DIR}/outputs"

import os
os.makedirs(OUT_DIR, exist_ok=True)
!pip -q install transformers accelerate bitsandbytes sentencepiece datasets peft trl --upgrade
import os, re, math, random, json, torch
import pandas as pd
from collections import Counter
from typing import List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

DEVICE_MAP = "auto"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

BASE_MODEL_ID = "deepseek-ai/deepseek-math-7b-instruct"
LORA_DIR = f"{OUT_DIR}/deepseek_math_qlora"

MAX_TOKENS = 256
TOP_P = 0.95
TEMPERATURE_GRID = [0.7]
N_SAMPLES = 2
FEWSHOTS = [
    ("Вычислите 12·(5-2).", "[36]"),
    ("Найдите предел: lim_{x->0} (sin x)/x.", "[1]"),
    ("Решите уравнение: 2x-7=5.", "[6]"),
]

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
def load_base_model(model_id=BASE_MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True
    )
    return tok, mdl

tokenizer, base_model = load_base_model()
print(BASE_MODEL_ID)
def build_prompt(task_text: str, fewshots=FEWSHOTS) -> str:
    header = (
        "Ты — математический ассистент. Решай задачу пошагово.\n"
        "В конце выведи только одну строку строго в формате [число].\n\n"
        "Примеры:\n"
    )
    for q, a in fewshots:
        header += f"Задача: {q}\nОтвет: {a}\n\n"
    header += f"Задача: {task_text}\nОтвет:"
    return header

num_pat = re.compile(r"\[([^\[\]\n]+)\]")

def normalize_answer(txt: str):
    candidates = num_pat.findall(txt)
    if not candidates:
        fallback = re.findall(r"-?\d+(?:[.,]\d+)?", txt.replace(",", "."))
        if not fallback: return None
        val = fallback[-1]
    else:
        val = candidates[-1].strip()

    val = val.replace(",", ".")
    try:
        if "/" in val and all(p.strip("-").isdigit() for p in val.split("/", 1)):
            num, den = val.split("/", 1); den = int(den); num = int(num)
            if den == 0: return None
            if num % den == 0: val = str(num // den)
            else: val = str(num / den)
        else:
            if re.fullmatch(r"-?\d+", val): val = str(int(val))
            else: val = str(float(val))
    except Exception:
        return None

    try:
        f = float(val)
        if abs(f) < 1e-12: val = "0"
        if math.isfinite(f) and abs(f - round(f)) < 1e-12: val = str(int(round(f)))
    except: pass
    return f"[{val}]"
  def generate_once(model, task_text: str, temperature: float, seed=None):
    if seed is not None: torch.manual_seed(seed)
    prompt = build_prompt(task_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=512, do_sample=True,
        temperature=temperature, top_p=0.95,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

from collections import Counter
import math, numpy as np

def solve_with_grid_and_consistency(model, task_text: str, temps=TEMPERATURE_GRID, n_samples=N_SAMPLES,
                                    return_votes=False, early_stop_k=2):
    votes = []
    cnt = Counter()
    for t in temps:
        for i in range(n_samples):
            raw = generate_once(model, task_text, temperature=t, seed=1000 + i + int(t*10))
            ans = normalize_answer(raw)
            if ans:
                votes.append(ans)
                cnt[ans] += 1
                if cnt[ans] >= early_stop_k:
                    best = ans
                    return (best, votes) if return_votes else best

    if not votes:
        return ("[0]", []) if return_votes else "[0]"

    c = Counter(votes)
    best, bestc = c.most_common(1)[0]
    ties = [k for k, v in c.items() if v == bestc]
    if len(ties) > 1:
        def numify(x):
            try: return float(x.strip("[]"))
            except: return float("nan")
        nums = [numify(k) for k in ties if k]
        nums = [x for x in nums if math.isfinite(x)]
        if nums:
            med = float(np.median(nums))
            if abs(med - round(med)) < 1e-12: med = int(round(med))
            best = f"[{med}]"
    return (best, votes) if return_votes else best
    import random, math as m

def make_arithmetic(n=100):
    tasks, answers = [], []
    ops = ['+', '-', '*']
    for _ in range(n):
        a, b = random.randint(-50, 50), random.randint(-50, 50)
        op = random.choice(ops)
        expr = f"{a} {op} {b}"
        val = eval(expr)
        tasks.append(f"Вычислите значение выражения: {expr}.")
        answers.append(f"[{val}]")
    return tasks, answers

def make_linear(n=80):
    tasks, answers = [], []
    for _ in range(n):
        a = random.randint(1, 12)
        x = random.randint(-20, 20)
        b = random.randint(-30, 30)
        c = a*x + b
        tasks.append(f"Решите уравнение: {a}x + {b} = {c}. Введите значение x.")
        answers.append(f"[{x}]")
    return tasks, answers

def make_quadratic_integer_roots(n=60):
    tasks, answers = [], []
    for _ in range(n):
        r1 = random.randint(-12, 12)
        r2 = random.randint(-12, 12)
        a = random.choice([1, 1, 2])
        b = -a*(r1 + r2)
        c = a*(r1*r2)
        tasks.append(f"Найдите корень многочлена: {a}x^2 + {b}x + {c} = 0. Введите любой один корень.")
        answers.append(f"[{r1}]")
    return tasks, answers

def make_gcd(n=60):
    tasks, answers = [], []
    for _ in range(n):
        a = random.randint(1, 200)
        b = random.randint(1, 200)
        g = m.gcd(a, b)
        tasks.append(f"Найдите наибольший общий делитель чисел {a} и {b}.")
        answers.append(f"[{g}]")
    return tasks, answers

def paraphrase_tasks(model, tasks: list, answers: list, num_paraphrases=1, temperature=0.7):
    out_tasks, out_answers = [], []
    for i, (t, a) in enumerate(zip(tasks, answers)):
        for k in range(num_paraphrases):
            prompt = (
                "Переформулируй математическую задачу и сохрани её смысл так, чтобы ответ не менялся.\n"
                "Выведи только новый вариант условия без пояснений.\n\n"
                f"Задача: {t}\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=temperature, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            new_task = text[len(prompt):].strip().strip('"').strip()
            if new_task:
                out_tasks.append(new_task)
                out_answers.append(a)
        if (i+1) % 20 == 0:
            print(f"Парафразировано {i+1}/{len(tasks)}")
    return out_tasks, out_answers
import pandas as pd, os
assert os.path.exists(TRAIN_PATH),

df_train = pd.read_csv(TRAIN_PATH)
task_col = df_train.columns[0]
ans_col  = df_train.columns[1] if len(df_train.columns) > 1 else None

def _norm_brackets(x: str) -> str:
    s = str(x).strip().strip("[]")
    return f"[{s}]"

base = pd.DataFrame({
    "task": df_train[task_col].astype(str),
    "answer": (df_train[ans_col].astype(str) if ans_col else "[0]").apply(_norm_brackets),
    "source": "orig"
})

para_tasks, para_answers = paraphrase_tasks(base_model, base["task"].tolist(), base["answer"].tolist(), 1, 0.7)
paraphr = pd.DataFrame({"task": para_tasks, "answer": para_answers, "source": "paraphrase"})

syn_tasks, syn_answers = [], []
for maker in [make_arithmetic, make_linear, make_quadratic_integer_roots, make_gcd]:
    t, a = maker()
    syn_tasks += t; syn_answers += a
synth = pd.DataFrame({"task": syn_tasks, "answer": syn_answers, "source": "synthetic"})

expanded = pd.concat([base, paraphr, synth], ignore_index=True)


orig = expanded[expanded["source"]=="orig"]
par  = expanded[expanded["source"]=="paraphrase"]
syn  = expanded[expanded["source"]=="synthetic"]
balanced = pd.concat([orig, orig, par, syn], ignore_index=True).sample(frac=1.0, random_state=42)

expanded_path = f"{OUT_DIR}/expanded_train.csv"
balanced_path = f"{OUT_DIR}/expanded_train_balanced.csv"
base_path     = f"{OUT_DIR}/train_orig_only.csv"

expanded.to_csv(expanded_path, index=False)
balanced.to_csv(balanced_path, index=False)
base.to_csv(base_path, index=False)
import os, pandas as pd
from datasets import Dataset
from transformers import (AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def format_example(task, answer):
    return (
        "Ты — математический ассистент. Решай задачу кратко и по шагам.\n"
        "В конце строго выведи ответ в формате [число].\n\n"
        "### Задача:\n"
        f"{task}\n\n"
        "### Правильный ответ:\n"
        f"{answer}"
    )

train_file = f"{OUT_DIR}/expanded_train_balanced.csv"
assert os.path.exists(train_file),

df = pd.read_csv(train_file)
ds = Dataset.from_pandas(df).shuffle(seed=42)
ds = ds.map(lambda s: {"text": format_example(s["task"], s["answer"])})

base_model_4bit = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    torch_dtype=DTYPE,
    device_map=DEVICE_MAP,
)
base_model_4bit = prepare_model_for_kbit_training(base_model_4bit)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
lora_model = get_peft_model(base_model_4bit, peft_config)


tokenizer.pad_token = tokenizer.eos_token
def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN
    )
tok_ds = ds.map(tok, batched=True, remove_columns=ds.column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=LORA_DIR,
    num_train_epochs=FINETUNE_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    save_strategy="epoch",
    bf16=(DTYPE == torch.bfloat16),
    fp16=False if DTYPE == torch.bfloat16 else True,
    optim="paged_adamw_32bit",
    report_to="none",
)


trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=collator,
)

trainer.train()
lora_model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(LORA_DIR)
def load_for_inference():
    if os.path.exists(LORA_DIR):
        print(LORA_DIR)
        tok = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            LORA_DIR,
            torch_dtype=DTYPE,
            device_map=DEVICE_MAP,
            low_cpu_mem_usage=True
        )
        return tok, mdl
    else:
        print(BASE_MODEL_ID)
        return tokenizer, base_model

inf_tokenizer, inf_model = load_for_inference()
import os
import pandas as pd

assert os.path.exists(TEST_PATH), "Не найден TEST_PATH"
df_test = pd.read_csv(TEST_PATH)
task_col = df_test.columns[0]
tasks = df_test[task_col].astype(str).tolist()

sub_path   = f"{OUT_DIR}/submission.csv"
subw_path  = f"{OUT_DIR}/submission_with_tasks.csv"
votes_path = f"{OUT_DIR}/submission_votes_log.csv"


done = 0
if os.path.exists(sub_path):
    try:
        done = len(pd.read_csv(sub_path))
        print(f"Резюмируем: уже готово {done}/{len(tasks)}")
    except Exception:
        done = 0

answers = []
logs = []

if done > 0:
    answers = pd.read_csv(sub_path)['answer'].astype(str).tolist()
    prev_logs = pd.read_csv(votes_path) if os.path.exists(votes_path) else pd.DataFrame(columns=["idx","task","pred","votes"])
else:
    prev_logs = pd.DataFrame(columns=["idx","task","pred","votes"])


for i, t in enumerate(tasks[done:], start=done+1):
    pred, votes = solve_with_grid_and_consistency(inf_model, t, temps=TEMPERATURE_GRID,
                                                  n_samples=N_SAMPLES, return_votes=True)
    answers.append(pred)
    logs.append({"idx": i, "task": t, "pred": pred, "votes": "|".join(votes)})

    if (i % 5 == 0) or (i == len(tasks)):
        pd.DataFrame({"answer": answers}).to_csv(sub_path, index=False)
        pd.DataFrame({task_col: tasks[:len(answers)], "answer": answers}).to_csv(subw_path, index=False)
        out_logs = pd.concat([prev_logs, pd.DataFrame(logs)], ignore_index=True)
        out_logs.to_csv(votes_path, index=False)
        print(f"[TEST] {i}/{len(tasks)}")
        logs = []
    
