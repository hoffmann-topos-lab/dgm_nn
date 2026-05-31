"""
nn_bench/report.py — Agregação de resultados e cálculo do score de archive.

Lê os JSONs de resultado produzidos pelo harness, determina quais tarefas
foram "resolvidas" (accuracy >= threshold) e gera um arquivo de avaliação
no formato esperado por `utils/evo_utils.py:get_all_performance()`.

Estrutura do arquivo de avaliação gerado (compatível com o DGM original):
{
    "resolved_instances": int,
    "submitted_instances": int,
    "resolved_ids":   [task_id, ...],   # accuracy >= threshold
    "unresolved_ids": [task_id, ...],   # accuracy < threshold (mas sem erro)
    "empty_patch_ids": [task_id, ...]   # falhou / não terminou
}
"""

import json
import os
from pathlib import Path

from nn_bench.tasks import get_task


def _read_result(dname: str) -> dict | None:
    """
    Lê o result.json de um diretório de output do harness.
    Retorna None se o arquivo não existir ou estiver corrompido.
    """
    result_file = os.path.join(dname, "result.json")
    if not os.path.exists(result_file):
        return None
    try:
        with open(result_file) as f:
            return json.load(f)
    except Exception:
        return None


def _is_failed(result: dict | None) -> bool:
    """Retorna True se o resultado indica uma falha do agente (empty patch)."""
    if result is None:
        return True
    if result.get("error"):
        return True
    if result.get("epochs_trained", 0) == 0:
        return True
    return False


def _is_resolved(result: dict, task_id: str) -> bool:
    """Retorna True se a accuracy final atingiu o threshold da tarefa."""
    try:
        task = get_task(task_id)
    except ValueError:
        return False
    return result.get("accuracy", 0.0) >= task["threshold"]


def make_report(
    dnames: list,
    run_ids: list,
    output_dir: str,
    dnames_workers: int = 1,  # mantido para compatibilidade de interface
    **kwargs,                  # absorve argumentos extras (ex: dataset_name)
) -> str:
    """
    Agrega os resultados do harness e salva um JSON de avaliação em output_dir.

    Quando há múltiplas avaliações da mesma tarefa (num_evals > 1), usa a
    melhor accuracy entre as repetições (comportamento otimista, consistente
    com o DGM original que acumula avaliações).

    Args:
        dnames: Lista de diretórios de output do harness.
        run_ids: Lista de run IDs correspondentes (usada para nomear o arquivo).
        output_dir: Diretório onde salvar o arquivo de avaliação.
        dnames_workers: Não utilizado (compatibilidade com interface original).

    Returns:
        Caminho do arquivo de avaliação gerado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Agrupa resultados por task_id (pode haver múltiplas avaliações da mesma tarefa)
    best_result_per_task: dict[str, dict] = {}

    for dname in dnames:
        result = _read_result(dname)

        # Infere o task_id a partir do nome do diretório (formato: {task_id}_eval{i})
        dir_name = os.path.basename(dname)
        task_id = dir_name.rsplit("_eval", 1)[0]

        if _is_failed(result):
            # Registra falha apenas se não houver resultado válido ainda
            if task_id not in best_result_per_task:
                best_result_per_task[task_id] = None
            continue

        # Mantém o melhor resultado para a tarefa
        current_best = best_result_per_task.get(task_id)
        if current_best is None or result["accuracy"] > current_best.get("accuracy", 0.0):
            best_result_per_task[task_id] = result

    # Classifica cada tarefa
    resolved_ids   = []
    unresolved_ids = []
    empty_patch_ids = []

    for task_id, result in best_result_per_task.items():
        if _is_failed(result):
            empty_patch_ids.append(task_id)
        elif _is_resolved(result, task_id):
            resolved_ids.append(task_id)
        else:
            unresolved_ids.append(task_id)

    submitted_instances = len(resolved_ids) + len(unresolved_ids) + len(empty_patch_ids)
    resolved_instances  = len(resolved_ids)

    eval_summary = {
        "resolved_instances":  resolved_instances,
        "submitted_instances": submitted_instances,
        "resolved_ids":        sorted(resolved_ids),
        "unresolved_ids":      sorted(unresolved_ids),
        "empty_patch_ids":     sorted(empty_patch_ids),
        # Detalhes adicionais para análise
        "task_details": {
            task_id: {
                "accuracy":       (r["accuracy"]       if r else None),
                "loss":           (r["loss"]            if r else None),
                "epochs_trained": (r["epochs_trained"]  if r else 0),
                "error":          (r.get("error")       if r else "agent_failed"),
            }
            for task_id, r in best_result_per_task.items()
        },
    }

    # Nome do arquivo compatível com get_all_performance() do evo_utils.py:
    # a função busca arquivos .json cujo nome contém `run_keyword` (= model_name_or_path = run_id)
    base_run_id = run_ids[0].rsplit("_", 1)[0] if run_ids else "unknown"
    eval_file = os.path.join(output_dir, f"{base_run_id}_nn_eval.json")

    with open(eval_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"[report] Avaliação salva em: {eval_file}")
    print(f"[report] Resolvidas: {resolved_ids}")
    print(f"[report] Não resolvidas: {unresolved_ids}")
    print(f"[report] Falhas: {empty_patch_ids}")
    print(f"[report] Score: {resolved_instances}/{submitted_instances} = "
          f"{resolved_instances/submitted_instances:.3f}" if submitted_instances else "[report] Score: 0/0")

    return eval_file
