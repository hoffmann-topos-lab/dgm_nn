"""
prompts/self_improvement_prompt.py — Funções auxiliares de prompt para o DGM-NN.

Contém helpers compartilhados (leitura de código, leitura de logs) e
o prompt legado de diagnóstico adaptado para o contexto de treinamento
de redes neurais.

O módulo principal de diagnóstico é nn_self_improvement_prompt.py.
Este arquivo fornece funções utilitárias que podem ser importadas
por outros módulos de prompt.
"""

import os

from utils.common_utils import load_json_file, read_file


# ---------------------------------------------------------------------------
# Sumário do agente NN — usado pelo problem_statement
# ---------------------------------------------------------------------------

nn_agent_summary = """# NN Agent Summary

- **Main File**: `nn_agent.py`
  - **Training mode**: `python nn_agent.py --task_id <task_id> --output_file <path>`
    - Defines and trains a neural network on the given task, saves result.json.
  - **Self-improve mode**: `python nn_agent.py --self_improve --problem_statement "..." ...`
    - Uses an LLM with bash/edit tools to modify its own source code.

- **Key helper functions (primary targets for evolution)**:
  - `build_model(input_shape, num_classes)` — Model architecture
  - `get_transforms(task)` — Data augmentation pipeline
  - `get_dataloaders(task, train_transform, val_transform)` — Dataset loading
  - `get_optimizer(model, task)` — Optimizer and LR scheduler
  - `run_training(task_id, output_file)` — Main training loop

- **Tools available**: `bash` (run shell commands), `edit` (modify files).
- **Dependencies**: PyTorch + torchvision already installed.
- **Hard constraints**:
  - Do NOT modify `run_self_improve()` or the argument parsing block.
  - Do NOT use `while True` loops.
  - Changes must keep ALL tasks working.

"""

# ---------------------------------------------------------------------------
# System message template — contém o código atual do agente
# ---------------------------------------------------------------------------

diagnose_system_message = """You are an expert ML engineer reviewing the performance of a neural network training agent.

# NN Agent Code
The current implementation of the agent (including applied patches).
----- NN Agent Code Start -----
{code}
----- NN Agent Code End -----

Your task is to identify ONE concrete, high-impact improvement that would
increase the agent's accuracy on ML benchmarks. The improvement must be
general enough to benefit ALL supported tasks.
"""

# ---------------------------------------------------------------------------
# User prompt — diagnóstico de uma tarefa
# ---------------------------------------------------------------------------

diagnose_prompt = """
# Training Result for Task '{task_id}'

## Training Log
----- Training Log Start -----
{md_log}
----- Training Log End -----

## Training Metrics
- Accuracy: {accuracy}
- Loss: {loss}
- Epochs trained: {epochs_trained}
- Threshold: {threshold}

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "log_summarization": Analyze the training logs and summarize what the training curves reveal (overfitting? underfitting? plateau? learning rate issues?).
- "potential_improvements": Identify 2-3 concrete improvements to `nn_agent.py` that could enhance accuracy across all tasks. Focus on architecture, optimizer, augmentation, or training loop changes.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail.
- "implementation_suggestion": Concrete code-level guidance — which function to change, what code to add/replace.
- "problem_description": Phrase the improvement as a description that a software engineer can implement by reading `nn_agent.py`.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

# ---------------------------------------------------------------------------
# User prompt — agente não gerou patch (empty patches)
# ---------------------------------------------------------------------------

diagnose_prompt_emptypatches = """The NN training agent ran in self-improve mode but produced an empty patch (no code changes were made to `nn_agent.py`). Since the agent is stochastic, it may not always produce a patch. Handle cases where the agent fails to generate code modifications.

Please analyze the log below to identify why no code edits were made.

# Agent Self-Improve Log
----- Log Start -----
{md_log}
----- Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to make the agent more reliable at generating code modifications.
- "improvement_proposal": Choose ONE high-impact improvement and describe it in detail.
- "implementation_suggestion": Concrete code-level guidance referencing the agent's structure.
- "problem_description": Phrase the improvement as a description that a software engineer can implement.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

# ---------------------------------------------------------------------------
# Problem description template
# ---------------------------------------------------------------------------

problem_description_prompt = """# To Implement\n\n{implementation_suggestion}\n\n{problem_description}"""


def get_problem_description_prompt(response_json):
    """Format the LLM diagnosis JSON into a problem_statement for nn_agent.py --self_improve."""
    return nn_agent_summary + problem_description_prompt.format(
        implementation_suggestion=response_json.get("implementation_suggestion", ""),
        problem_description=response_json.get("problem_description", ""),
    )


# ---------------------------------------------------------------------------
# Helpers de leitura de logs e código
# ---------------------------------------------------------------------------

def read_mdlog_file(filepath, filter=True):
    """Read a markdown log file, optionally filtering out error lines."""
    if not filter:
        return read_file(filepath)

    filter_content = [
        'Error in get_response_withtools',
    ]
    filtered_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if not any(line.startswith(fc) for fc in filter_content):
                filtered_lines.append(line.rstrip('\n'))
    return "\n".join(filtered_lines).strip()


def find_selfimprove_eval_logs(entry, out_dir, commit_id='initial', filter=True):
    """
    Find evaluation logs and results for a given task entry in a commit's predictions.
    Returns (md_logs, result_jsons, predicted_patches, eval_results).
    """
    predictions_dir = os.path.join(out_dir, commit_id, 'predictions')
    if not os.path.isdir(predictions_dir):
        return [], [], [], []

    all_preds_folders = [f for f in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, f))]

    # Read result.json files for each prediction folder
    md_logs = []
    predicted_patches = []
    eval_results = []

    for folder in all_preds_folders:
        # Try to read the self-improve log
        log_file = os.path.join(predictions_dir, folder, f"{entry}.md")
        if os.path.exists(log_file):
            md_logs.append(read_mdlog_file(log_file, filter=filter))

        # Try to read result.json
        result_file = os.path.join(predictions_dir, folder, "result.json")
        if os.path.exists(result_file):
            result_data = load_json_file(result_file)
            accuracy = result_data.get("accuracy", 0.0)
            predicted_patches.append(f"accuracy={accuracy}")
            eval_results.append(result_data)

    return md_logs, [], predicted_patches, eval_results


def process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results=None):
    """Process the collected logs, returning the first of each."""
    md_log = md_logs[0] if md_logs else "No logs available."
    eval_log = eval_logs[0] if eval_logs else "No test results available."
    predicted_patch = predicted_patches[0] if predicted_patches else "No predicted patch available."

    # Truncate logs if too long
    if len(md_log) > 250000:
        md_log = md_log[:250000] + "\n<log clipped>"

    eval_result = eval_results[0] if eval_results else "No evaluation result available."
    return md_log, eval_log, predicted_patch, eval_result


def get_current_code(current_dir, code_files, patch_files=None, exclude_files=None):
    """
    Retrieves the contents of specified Python files/directories, optionally
    applying patches. Also allows excluding specific files from the result.

    :param current_dir: Root directory to resolve paths against.
    :param code_files: List of files or directories to include.
    :param patch_files: List of patch files to include at the end of the output.
    :param exclude_files: List of files (relative paths to current_dir) to exclude.
    :return: A string containing all requested code (and patches).
    """
    if patch_files is None:
        patch_files = []
    if exclude_files is None:
        exclude_files = []

    exclude_set = set(exclude_files)
    code_text = []

    for file_path in code_files:
        full_path = os.path.join(current_dir, file_path)

        if file_path in exclude_set:
            continue

        if os.path.isfile(full_path):
            rel_path = os.path.relpath(full_path, current_dir)
            if rel_path not in exclude_set:
                code_text.append(f"# {rel_path}")
                code_text.append(read_file(full_path))

        elif os.path.isdir(full_path):
            for root, _, files in os.walk(full_path):
                for f in files:
                    if f.endswith('.py'):
                        file_full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(file_full_path, current_dir)
                        if rel_path not in exclude_set:
                            code_text.append(f"# {rel_path}")
                            code_text.append(read_file(file_full_path))

    # Add patch files
    for i, patch_file in enumerate(patch_files):
        rel_path = os.path.relpath(patch_file, current_dir)
        if rel_path not in exclude_set:
            code_text.append(f"# Patch {i+1}: {rel_path}")
            code_text.append(read_file(patch_file))

    return "\n".join(code_text)
