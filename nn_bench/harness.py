"""
nn_bench/harness.py — Harness de avaliação do DGM-NN.

Orquestra a execução de nn_agent.py em containers Docker para cada tarefa
da lista fornecida, aplica os patches acumulados do agente e coleta os
JSONs de resultado.

Mantém compatibilidade com self_improve_step.py.
"""

import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker

from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)


def _run_single_task(
    task_id: str,
    model_patch_paths: list,
    pred_dname: str,
    eval_index: int,
    root_dir: str,
    image_name: str = "dgm",
    log_dir: str = None,
) -> str:
    """
    Executa nn_agent.py para uma única tarefa dentro de um container Docker.

    Args:
        task_id: ID da tarefa (ex: "mnist").
        model_patch_paths: Lista de caminhos de patch a aplicar no agente.
        pred_dname: Diretório base onde salvar os resultados.
        eval_index: Índice desta avaliação (para suporte a num_evals > 1).
        root_dir: Diretório raiz do projeto no host.
        image_name: Nome da imagem Docker.

    Returns:
        Caminho do diretório de output desta execução.
    """
    # Configura logger para esta thread (evita "No logger found for thread")
    if log_dir:
        setup_logger(os.path.join(log_dir, f"harness_{task_id}_eval{eval_index}.log"))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"dgm-nn-{task_id}-eval{eval_index}-{ts}"

    out_dir = os.path.join(pred_dname, f"{task_id}_eval{eval_index}")
    os.makedirs(out_dir, exist_ok=True)
    result_file_host = os.path.join(out_dir, "result.json")

    client = docker.from_env()
    remove_existing_container(client, container_name)

    container = build_dgm_container(
        client, root_dir, image_name, container_name, force_rebuild=False
    )
    if container is None:
        safe_log(f"[{task_id}] Falha ao criar container.")
        _write_error_result(result_file_host, task_id, "Falha ao criar container Docker.")
        return out_dir

    container.start()

    try:
        # Aplica patches acumulados ao nn_agent.py dentro do container
        for patch_file in model_patch_paths:
            copy_to_container(container, patch_file, "/dgm/parent_patch.txt")
            exec_result = container.exec_run(
                "/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir="/dgm"
            )
            log_container_output(exec_result)
            container.exec_run("rm /dgm/parent_patch.txt", workdir="/dgm")

        # Executa o agente de treinamento
        result_container_path = f"/tmp/nn_results/{task_id}.json"
        exec_result = container.exec_run(
            [
                "timeout", str(7200),  # 2h de limite absoluto (margem 2x sobre o big de 3600s)
                "python", "/dgm/nn_agent.py",
                "--task_id", task_id,
                "--output_file", result_container_path,
            ],
            workdir="/dgm",
        )
        safe_log(f"[{task_id}] Saída do agente: {exec_result.output.decode()[:500]}")

        # Copia resultado de volta para o host
        try:
            copy_from_container(container, result_container_path, result_file_host)
        except Exception as e:
            safe_log(f"[{task_id}] Erro ao copiar resultado: {e}")
            _write_error_result(result_file_host, task_id, f"Resultado não encontrado: {e}")

    except Exception as e:
        safe_log(f"[{task_id}] Erro durante execução: {e}")
        _write_error_result(result_file_host, task_id, str(e))

    finally:
        cleanup_container(container)

    return out_dir


def _write_error_result(result_file: str, task_id: str, error_msg: str):
    """Salva um JSON de resultado vazio quando o agente falha."""
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    result = {
        "task_id": task_id,
        "accuracy": 0.0,
        "loss": float("inf"),
        "epochs_trained": 0,
        "train_time_seconds": 0.0,
        "history": {"train_loss": [], "val_loss": [], "val_accuracy": []},
        "error": error_msg,
    }
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)


def harness(
    test_task_list: list,
    model_name_or_path: str,
    model_patch_paths: list,
    pred_dname: str,
    num_evals: int = 1,
    num_evals_parallel: int = 1,
    max_workers: int = 4,
    num_samples: int = -1,  # mantido para compatibilidade de interface
) -> list:
    """
    Executa o nn_agent.py em Docker para cada tarefa da lista e coleta resultados.

    Args:
        test_task_list: Lista de task IDs a avaliar (ex: ["mnist", "cifar10"]).
        model_name_or_path: Identificador do run (usado para nomear arquivos).
        model_patch_paths: Patches acumulados a aplicar ao agente.
        pred_dname: Diretório onde salvar os resultados por tarefa.
        num_evals: Número de avaliações repetidas por tarefa.
        num_evals_parallel: Workers paralelos para as avaliações repetidas.
        max_workers: Workers paralelos entre tarefas distintas.
        num_samples: Não utilizado (compatibilidade com interface original).

    Returns:
        Lista de diretórios de output, um por (tarefa × avaliação).
    """
    root_dir = os.path.abspath("./")
    os.makedirs(pred_dname, exist_ok=True)

    # Monta todas as combinações (task_id, eval_index)
    jobs = [
        (task_id, eval_i)
        for task_id in test_task_list
        for eval_i in range(num_evals)
    ]

    dnames = []
    effective_workers = min(max_workers, len(jobs))

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _run_single_task,
                task_id,
                model_patch_paths,
                pred_dname,
                eval_i,
                root_dir,
                log_dir=pred_dname,
            ): (task_id, eval_i)
            for task_id, eval_i in jobs
        }
        for future in as_completed(futures):
            task_id, eval_i = futures[future]
            try:
                out_dir = future.result()
                dnames.append(out_dir)
                safe_log(f"[{task_id} eval{eval_i}] Concluído → {out_dir}")
            except Exception as e:
                safe_log(f"[{task_id} eval{eval_i}] Erro: {e}")

    return dnames
