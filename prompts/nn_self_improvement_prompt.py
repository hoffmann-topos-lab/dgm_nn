"""
prompts/nn_self_improvement_prompt.py — Prompt de diagnóstico LLM para o DGM-NN.

Análogo ao self_improvement_prompt.py do DGM original, mas adaptado para
diagnosticar logs de treinamento de redes neurais em vez de logs de patching
de código de repositórios GitHub.

Fluxo:
  1. `get_diagnose_prompt_nn()` monta o system message (código atual do agente)
     e o user prompt (logs de treinamento + métricas).
  2. O LLM responde com JSON contendo diagnóstico e proposta de melhoria.
  3. `get_problem_description_nn()` formata o JSON em um problem_statement
     que será passado para `nn_agent.py --self_improve`.
"""

import os

from utils.common_utils import read_file
from utils.nn_eval_utils import diagnose_training, load_results_from_dir


# ---------------------------------------------------------------------------
# Sumário do agente — injetado no problem_statement final
# ---------------------------------------------------------------------------

nn_agent_summary = """# NN Agent Summary

- **Main File**: `nn_agent.py`
  - **Training mode**: `python nn_agent.py --task_id <task_id> --output_file <path>`
    - Defines and trains a neural network on the given task, saves result.json.
  - **Self-improve mode**: `python nn_agent.py --self_improve --problem_statement "..." ...`
    - Uses an LLM with bash/edit tools to modify its own source code.

- **Key helper functions (primary targets for evolution)**:
  - `build_model(input_shape, num_classes)` — Model architecture (default: 2-layer MLP). Change this to add CNN, ResNet blocks, BatchNorm, Dropout, etc.
  - `get_transforms(task)` — Data augmentation pipeline. Returns `(train_transform, val_transform)`. Change this to add RandomCrop, HorizontalFlip, ColorJitter, Normalize tweaks, etc.
  - `get_dataloaders(task, train_transform, val_transform)` — Dataset loading. Change this to adjust batch size, num_workers, samplers.
  - `get_optimizer(model, task)` — Optimizer and LR. Change this to use SGD+momentum, weight decay, CosineAnnealingLR, ReduceLROnPlateau, etc.
  - `run_training(task_id, output_file)` — Main training loop (inside this function). Change this to add gradient clipping, mixed precision, early stopping, etc.

- **Supported tasks**: mnist, fashion_mnist, cifar10, cifar100, svhn, stl10
  - Each task has its own `input_shape`, `num_classes`, `threshold`, `max_epochs`, `max_train_time_seconds`.
  - All changes MUST remain compatible with ALL six tasks.

- **Output format** (`result.json`):
  ```json
  {
    "task_id": "mnist",
    "accuracy": 0.97,
    "loss": 0.09,
    "epochs_trained": 10,
    "train_time_seconds": 45.2,
    "history": {"train_loss": [...], "val_loss": [...], "val_accuracy": [...]},
    "error": null
  }
  ```

- **Tools available**: `bash` (run shell commands), `edit` (modify files).
- **Dependencies**: PyTorch + torchvision already installed. Do NOT install additional packages without updating `requirements.txt`.
- **Hard constraints**:
  - Do NOT modify `run_self_improve()` or the argument parsing block.
  - Do NOT use `while True` loops.
  - Changes must keep ALL six tasks working.

"""

# ---------------------------------------------------------------------------
# System message — contém o código atual do agente
# ---------------------------------------------------------------------------

_system_message_template = """\
You are an expert ML engineer reviewing the performance of a neural network \
training agent that evolves its own code to improve accuracy on ML benchmarks.

# NN Agent Code
The current implementation of the agent (including applied patches).
----- NN Agent Code Start -----
{code}
----- NN Agent Code End -----

Your task is to identify ONE concrete, high-impact improvement that would \
increase the agent's accuracy on ML benchmarks. The improvement must be \
general enough to benefit ALL supported tasks \
(mnist, fashion_mnist, cifar10, cifar100, svhn, stl10).
"""

# ---------------------------------------------------------------------------
# User prompt — diagnóstico de uma tarefa falha/não-resolvida
# ---------------------------------------------------------------------------

_diagnose_prompt_task = """\
# Training Result for Task '{task_id}'

## Summary
- **Accuracy** (validation): {accuracy:.4f}
- **Loss** (validation):     {loss:.4f}
- **Epochs trained**:        {epochs_trained}
- **Train time**:            {train_time:.1f}s
- **Threshold** (resolved):  {threshold:.2%}
- **Gap to threshold**:      {gap:+.4f}

## Training History
{history_text}

## Automated Diagnosis
{diagnosis}

---

# Instructions

Analyze the training result above and propose ONE concrete improvement to \
`nn_agent.py` that would close the gap to the accuracy threshold.

Focus on the agent's **general** training capabilities. Good candidates:
- Architecture improvements in `build_model()` (CNN, residual blocks, BatchNorm, Dropout)
- Learning rate scheduling in `get_optimizer()` (CosineAnnealingLR, ReduceLROnPlateau)
- Data augmentation in `get_transforms()` (RandomCrop, HorizontalFlip, ColorJitter)
- Training loop improvements (gradient clipping, mixed precision, early stopping)

Respond **precisely** in the following JSON format (including the markers):

```json
<JSON>
```

In <JSON>, provide:
- "log_summarization": What the training curves reveal (overfitting? underfitting? early plateau? learning rate too high/low?).
- "potential_improvements": List of 2-3 specific, actionable improvements to `nn_agent.py`.
- "improvement_proposal": ONE improvement chosen from the list, described in detail (what to change, why it should help).
- "implementation_suggestion": Concrete code-level guidance — which function to change, what code to add/replace, referencing the agent's structure.
- "problem_description": The improvement phrased as a GitHub issue description that a software engineer can implement by reading `nn_agent.py`.

Your response will be automatically parsed. Do NOT include the literal `<JSON>` tag.
"""

# ---------------------------------------------------------------------------
# User prompt — agente não produziu mudanças (empty patch)
# ---------------------------------------------------------------------------

_diagnose_prompt_empty = """\
# Empty Patch — Agent Produced No Changes

The neural network training agent ran in self-improve mode but produced an \
empty patch (no code changes were made to `nn_agent.py`).

# Agent Self-Improve Log
----- Log Start -----
{self_evo_log}
----- Log End -----

---

# Instructions

Analyze the log above and propose improvements to `nn_agent.py` that make \
the agent more reliable at generating code modifications during self-improvement.

Respond **precisely** in the following JSON format (including the markers):

```json
<JSON>
```

In <JSON>, provide:
- "log_summarization": What happened during self-improvement (why no changes were made).
- "potential_improvements": List of 2-3 improvements to increase reliability of code generation.
- "improvement_proposal": ONE improvement chosen from the list, described in detail.
- "implementation_suggestion": Concrete code-level guidance referencing the agent's structure.
- "problem_description": The improvement phrased as a GitHub issue description.

Your response will be automatically parsed. Do NOT include the literal `<JSON>` tag.
"""

# ---------------------------------------------------------------------------
# Template do problem_statement final → passado ao nn_agent.py --self_improve
# ---------------------------------------------------------------------------

_problem_description_template = (
    "{nn_agent_summary}"
    "# To Implement\n\n"
    "{implementation_suggestion}\n\n"
    "{problem_description}"
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_current_code(root_dir: str, patch_files: list) -> str:
    """
    Lê o código atual do agente (nn_agent.py + helpers relevantes),
    listando os patches aplicados ao final para contexto do LLM.
    """
    code_parts = []
    target_files = [
        "nn_agent.py",
        "nn_bench/tasks.py",
        "utils/nn_eval_utils.py",
    ]
    for rel_path in target_files:
        full_path = os.path.join(root_dir, rel_path)
        if os.path.exists(full_path):
            code_parts.append(f"# {rel_path}")
            code_parts.append(read_file(full_path))

    for i, patch_file in enumerate(patch_files):
        rel = os.path.relpath(patch_file, root_dir)
        code_parts.append(f"# Patch {i + 1}: {rel}")
        code_parts.append(read_file(patch_file))

    return "\n\n".join(code_parts)


def _format_history(history: dict, max_epochs: int = 20) -> str:
    """
    Formata o histórico de treinamento como tabela Markdown.
    Limita ao último `max_epochs` épocas para não estourar o contexto.
    """
    train_losses = history.get("train_loss", [])
    val_losses   = history.get("val_loss", [])
    val_accs     = history.get("val_accuracy", [])

    if not any([train_losses, val_losses, val_accs]):
        return "(sem histórico disponível)"

    n = max(len(train_losses), len(val_losses), len(val_accs))
    # Trunca ao início + últimas épocas se muito longo
    indices = list(range(n))
    if n > max_epochs:
        keep = 5
        indices = indices[:keep] + ["..."] + indices[-(max_epochs - keep):]

    lines = ["| Época | train_loss | val_loss | val_accuracy |",
             "|-------|------------|----------|--------------|"]
    for idx in indices:
        if idx == "...":
            lines.append("| ...   | ...        | ...      | ...          |")
            continue
        tl = f"{train_losses[idx]:.4f}" if idx < len(train_losses) else "—"
        vl = f"{val_losses[idx]:.4f}"   if idx < len(val_losses)   else "—"
        va = f"{val_accs[idx]:.4f}"     if idx < len(val_accs)     else "—"
        lines.append(f"| {idx + 1:5d} | {tl:10s} | {vl:8s} | {va:12s} |")

    return "\n".join(lines)


def _find_best_result(task_id: str, out_dir: str, commit: str) -> dict | None:
    """Encontra o melhor result.json para a tarefa no commit indicado."""
    search_dirs = [
        os.path.join(out_dir, commit, "predictions"),
        os.path.join(out_dir, "initial", "predictions"),
    ]
    best = None
    for pred_dir in search_dirs:
        if not os.path.isdir(pred_dir):
            continue
        results = load_results_from_dir(pred_dir, [task_id])
        r = results.get(task_id)
        if r and not r.get("error") and (
            best is None or r.get("accuracy", 0.0) > best.get("accuracy", 0.0)
        ):
            best = r
    return best


def _read_self_evo_log(out_dir: str, run_id: str, max_chars: int = 30_000) -> str:
    """Lê o log de self-improvement de um run, truncando se necessário."""
    log_path = os.path.join(out_dir, run_id, "self_evo.md")
    if not os.path.exists(log_path):
        return "(log não encontrado)"
    content = read_file(log_path)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n<log truncado>"
    return content


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_diagnose_prompt_nn(
    task_id: str,
    commit: str,
    root_dir: str,
    out_dir: str,
    patch_files: list = [],
    run_id: str = None,
) -> tuple[str, str]:
    """
    Monta o system message e o user prompt para o diagnóstico LLM.

    Args:
        task_id:     ID da tarefa (ex: "mnist"), ou "solve_empty_patches".
        commit:      Run ID do pai (ou "initial").
        root_dir:    Diretório raiz do projeto.
        out_dir:     Diretório de output da run corrente.
        patch_files: Lista de patches já aplicados (para contexto do LLM).
        run_id:      Run ID do self-improvement recém-executado (para empty patch).

    Returns:
        (system_message, user_prompt)
    """
    code = _get_current_code(root_dir, patch_files)
    system_message = _system_message_template.format(code=code)

    if task_id == "solve_empty_patches" and run_id:
        self_evo_log = _read_self_evo_log(out_dir, run_id)
        user_prompt = _diagnose_prompt_empty.format(self_evo_log=self_evo_log)
        return system_message, user_prompt

    # Tarefa normal: diagnostica logs de treinamento
    from nn_bench.tasks import get_task
    try:
        task = get_task(task_id)
    except ValueError:
        # Tarefa desconhecida — fallback genérico
        task = {"threshold": 0.9, "description": task_id}

    result = _find_best_result(task_id, out_dir, commit)
    threshold  = task.get("threshold", 0.9)
    accuracy   = (result or {}).get("accuracy", 0.0)
    loss       = (result or {}).get("loss", float("inf"))
    epochs     = (result or {}).get("epochs_trained", 0)
    train_time = (result or {}).get("train_time_seconds", 0.0)
    history    = (result or {}).get("history", {})
    gap        = threshold - accuracy

    history_text = _format_history(history)
    diagnosis    = diagnose_training(result, task_id)

    user_prompt = _diagnose_prompt_task.format(
        task_id=task_id,
        accuracy=accuracy,
        loss=loss,
        epochs_trained=epochs,
        train_time=train_time,
        threshold=threshold,
        gap=gap,
        history_text=history_text,
        diagnosis=diagnosis,
    )
    return system_message, user_prompt


def get_problem_description_nn(response_json: dict) -> str:
    """
    Formata o JSON de resposta do LLM no problem_statement final
    que será passado para `nn_agent.py --self_improve`.

    Args:
        response_json: Dicionário com chaves do diagnóstico LLM
                       (implementation_suggestion, problem_description, ...).

    Returns:
        String de problem_statement pronta para uso.
    """
    return _problem_description_template.format(
        nn_agent_summary=nn_agent_summary,
        implementation_suggestion=response_json.get("implementation_suggestion", ""),
        problem_description=response_json.get("problem_description", ""),
    )
