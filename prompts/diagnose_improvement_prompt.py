"""
prompts/diagnose_improvement_prompt.py — Diagnóstico pós-melhoria para o DGM-NN.

Compara a performance do agente antes e depois de aplicar um model_patch,
analisando mudanças em accuracy, loss e curvas de treinamento.
"""

from prompts.self_improvement_prompt import get_current_code, read_mdlog_file
from utils.common_utils import read_file
from utils.nn_eval_utils import load_results_from_dir

import os


diagnose_improvement_system_message = """You are an expert ML engineer analyzing the impact of a code patch on a neural network training agent.

# NN Agent Code
The current code of the NN training agent with the model patch applied.
----- NN Agent Code Start -----
{code}
----- NN Agent Code End -----

# Model Patch
The code changes from the latest improvement iteration.
----- Model Patch Start -----
{model_patch_text}
----- Model Patch End -----

# Your Task
Analyze whether the model patch has improved the agent's training capabilities:
1. Compare accuracy/loss before and after the patch.
2. Identify any regressions (tasks that got worse).
3. Evaluate whether the changes are general or task-specific.
"""

diagnose_improvement_prompt = """Here are the training results for the NN agent, before and after applying the model patch.

# Results Before Patch
{before_results}

# Results After Patch
{after_results}

# Instructions

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "impact": Detailed analysis of the patch's impact on training performance. Compare accuracy and loss for each task before vs after. This should be thorough.
- "improvements": List of tasks and metrics that improved after the patch.
- "regressions": List of tasks and metrics that got worse after the patch (empty list if none).
- "score": Overall score from -2 to 2 (-2 = major regression, 0 = no change, 2 = major improvement).

Your response will be automatically parsed. Do NOT include the `<JSON>` tag in your output.
Think deeply about the impact of the changes."""


def _format_results(results_dir, task_ids):
    """Format training results from a predictions directory into a readable string."""
    results = load_results_from_dir(results_dir, task_ids)
    if not results:
        return "No results available."

    lines = []
    for task_id, result in results.items():
        if result.get("error"):
            lines.append(f"- {task_id}: ERROR — {result['error']}")
        else:
            acc = result.get("accuracy", 0.0)
            loss = result.get("loss", float("inf"))
            epochs = result.get("epochs_trained", 0)
            lines.append(f"- {task_id}: accuracy={acc:.4f}, loss={loss:.4f}, epochs={epochs}")
    return "\n".join(lines)


def get_diagnose_improvement_prompt(
        entry_id, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[],
        task_ids=None,
    ):
    """
    Build system message and user prompt for post-improvement diagnosis.

    Args:
        entry_id:         Task ID being diagnosed.
        parent_commit:    Parent run ID (before patch).
        root_dir:         Project root directory.
        model_patch_file: Path to the model_patch.diff file.
        out_dir:          DGM output directory.
        run_id:           Current run ID (after patch).
        patch_files:      List of accumulated patches for code context.
        task_ids:         List of task IDs to compare (default: all available).

    Returns:
        (system_message, user_prompt)
    """
    if task_ids is None:
        task_ids = ["mnist", "fashion_mnist"]

    # Get code context
    code_files = ['nn_agent.py', 'nn_bench/tasks.py']
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files)
    model_patch_text = read_file(model_patch_file)

    # Get before/after results
    before_dir = os.path.join(out_dir, parent_commit, 'predictions')
    after_dir = os.path.join(out_dir, run_id, 'predictions')

    before_results = _format_results(before_dir, task_ids)
    after_results = _format_results(after_dir, task_ids)

    system_msg = diagnose_improvement_system_message.format(
        code=code_text,
        model_patch_text=model_patch_text,
    )
    user_prompt = diagnose_improvement_prompt.format(
        before_results=before_results,
        after_results=after_results,
    )

    return system_msg, user_prompt
