"""
utils/nn_eval_utils.py — Utilitários de avaliação para o DGM-NN.

Funções auxiliares para ler, resumir e comparar resultados de treinamento.
Usado principalmente pelos scripts de análise e pelo prompt de diagnóstico.
"""

import json
import os


def load_result(result_file: str) -> dict | None:
    """Carrega um result.json do harness. Retorna None se inválido."""
    if not os.path.exists(result_file):
        return None
    try:
        with open(result_file) as f:
            return json.load(f)
    except Exception:
        return None


def load_results_from_dir(pred_dname: str, task_ids: list) -> dict:
    """
    Carrega todos os resultados de um diretório de predições do harness.

    Args:
        pred_dname: Diretório base de predições (contém subdirs por tarefa).
        task_ids: Lista de task IDs esperados.

    Returns:
        Dicionário {task_id: result_dict | None}.
    """
    results = {}
    for entry in os.listdir(pred_dname):
        entry_path = os.path.join(pred_dname, entry)
        if not os.path.isdir(entry_path):
            continue
        # Nome do diretório: {task_id}_eval{i}
        task_id = entry.rsplit("_eval", 1)[0]
        result_file = os.path.join(entry_path, "result.json")
        result = load_result(result_file)
        # Mantém o melhor resultado quando há múltiplas avaliações
        if task_id not in results or (
            result and result.get("accuracy", 0) > (results[task_id] or {}).get("accuracy", 0)
        ):
            results[task_id] = result
    return results


def summarize_result(result: dict | None, task_id: str = "") -> str:
    """
    Formata o resultado de uma tarefa como texto legível.
    Usado nos prompts de diagnóstico para descrever o estado do agente.
    """
    if result is None:
        return f"[{task_id}] FALHA: resultado não encontrado."

    error = result.get("error")
    if error:
        return f"[{task_id}] ERRO durante treinamento: {error}"

    acc    = result.get("accuracy", 0.0)
    loss   = result.get("loss", float("inf"))
    epochs = result.get("epochs_trained", 0)
    t      = result.get("train_time_seconds", 0.0)
    hist   = result.get("history", {})

    val_accs   = hist.get("val_accuracy", [])
    val_losses = hist.get("val_loss", [])
    train_losses = hist.get("train_loss", [])

    lines = [
        f"[{task_id}] Accuracy final: {acc:.4f} | Loss final: {loss:.4f} | "
        f"Épocas: {epochs} | Tempo: {t:.1f}s",
    ]

    if val_accs and len(val_accs) >= 2:
        delta = val_accs[-1] - val_accs[0]
        lines.append(f"  Evolução accuracy: {val_accs[0]:.4f} → {val_accs[-1]:.4f} (Δ={delta:+.4f})")

    if train_losses and val_losses and len(train_losses) == len(val_losses):
        gap = val_losses[-1] - train_losses[-1]
        if gap > 0.3:
            lines.append(f"  ⚠ Possível overfitting: gap train/val loss = {gap:.4f}")
        elif train_losses[-1] > 1.5:
            lines.append(f"  ⚠ Possível underfitting: train_loss ainda alto ({train_losses[-1]:.4f})")

    return "\n".join(lines)


def diagnose_training(result: dict | None, task_id: str = "") -> str:
    """
    Gera um diagnóstico estruturado do resultado de treinamento.
    Retorna texto formatado para uso no prompt de self-improvement.
    """
    if result is None or result.get("error"):
        error = (result or {}).get("error", "resultado não encontrado")
        return (
            f"Tarefa '{task_id}' falhou com erro: {error}\n"
            f"O agente não completou o treinamento. Verifique se o dataset está correto "
            f"e se o loop de treino não levanta exceções."
        )

    acc     = result.get("accuracy", 0.0)
    loss    = result.get("loss", float("inf"))
    epochs  = result.get("epochs_trained", 0)
    hist    = result.get("history", {})
    val_accs    = hist.get("val_accuracy", [])
    val_losses  = hist.get("val_loss", [])
    train_losses = hist.get("train_loss", [])

    lines = [f"## Diagnóstico — {task_id}"]
    lines.append(f"- Accuracy final (validação): {acc:.4f}")
    lines.append(f"- Loss final (validação): {loss:.4f}")
    lines.append(f"- Épocas treinadas: {epochs}")

    if val_accs:
        lines.append(f"- Histórico de accuracy: {[round(v, 3) for v in val_accs]}")
    if val_losses:
        lines.append(f"- Histórico de val_loss: {[round(v, 3) for v in val_losses]}")
    if train_losses:
        lines.append(f"- Histórico de train_loss: {[round(v, 3) for v in train_losses]}")

    # Identifica problemas
    problems = []
    if train_losses and val_losses and len(train_losses) == len(val_losses):
        final_gap = val_losses[-1] - train_losses[-1]
        if final_gap > 0.5:
            problems.append(
                f"Overfitting detectado: gap train/val loss = {final_gap:.4f}. "
                f"Considere dropout, batch normalization ou data augmentation."
            )
        if train_losses[-1] > 2.0:
            problems.append(
                f"Underfitting: train_loss = {train_losses[-1]:.4f} ainda alto. "
                f"Considere aumentar capacidade do modelo ou reduzir regularização."
            )

    if val_accs and len(val_accs) >= 3:
        last_3 = val_accs[-3:]
        if max(last_3) - min(last_3) < 0.001:
            problems.append(
                "Convergência prematura: accuracy estabilizou nas últimas 3 épocas. "
                "Considere ajustar learning rate (scheduler) ou aumentar épocas."
            )

    if problems:
        lines.append("\n### Problemas identificados:")
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("\n### Sem problemas óbvios identificados no treinamento.")

    return "\n".join(lines)


def compute_aggregate_score(results: dict, task_ids: list) -> float:
    """
    Calcula um score agregado entre tarefas (média simples de accuracy).
    Retorna 0.0 se nenhum resultado válido.
    """
    scores = []
    for task_id in task_ids:
        result = results.get(task_id)
        if result and not result.get("error") and result.get("epochs_trained", 0) > 0:
            scores.append(result.get("accuracy", 0.0))
    return sum(scores) / len(scores) if scores else 0.0
