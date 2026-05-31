import argparse
import datetime
import json
import os
import docker

from llm import create_client, get_response_from_llm, extract_json_between_markers
from nn_bench.harness import harness as nn_harness
from nn_bench.report import make_report
from prompts.nn_self_improvement_prompt import get_diagnose_prompt_nn, get_problem_description_nn
from utils.common_utils import load_json_file
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)

def _get_diagnose_model():
    """Lido em runtime (após load_dotenv) para garantir que .env foi carregado."""
    return os.environ['DIAGNOSE_MODEL']


def diagnose_problem(
    task_id: str,
    parent_commit: str,
    out_dir_base: str,
    root_dir: str,
    patch_files: list = [],
    max_attempts: int = 3,
    run_id: str = None,
) -> str | None:
    """
    Usa um LLM especializado para diagnosticar os logs de treinamento e gerar
    um problem_statement detalhado para o nn_agent.py --self_improve.

    Args:
        task_id:       Task ID (ex: "mnist") ou "solve_empty_patches".
        parent_commit: Run ID do pai (ou "initial").
        out_dir_base:  Diretório base de outputs.
        root_dir:      Diretório raiz do projeto.
        patch_files:   Patches acumulados até este ponto.
        max_attempts:  Tentativas em caso de falha de parse.
        run_id:        Run ID atual (usado para diagnóstico de empty patches).

    Returns:
        String de problem_statement, ou None em caso de falha.
    """
    client_tuple = create_client(_get_diagnose_model())
    try:
        sys_msg, user_prompt = get_diagnose_prompt_nn(
            task_id=task_id,
            commit=parent_commit,
            root_dir=root_dir,
            out_dir=out_dir_base,
            patch_files=patch_files,
            run_id=run_id,
        )
        response, msg_history = get_response_from_llm(
            msg=user_prompt,
            client=client_tuple[0],
            model=client_tuple[1],
            system_message=sys_msg,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Diagnose LLM response (truncated): {response[:500]}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json from diagnose LLM"
        problem_statement = get_problem_description_nn(response_json)
    except Exception as e:
        safe_log(f"Erro ao diagnosticar problema com LLM: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                task_id, parent_commit, out_dir_base, root_dir,
                patch_files=patch_files, max_attempts=max_attempts - 1, run_id=run_id,
            )
        return None
    return problem_statement


def save_metadata(metadata: dict, output_dir: str):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


def run_harness_nn(
    task_id, model_name_or_path, patch_files, num_evals,
    output_dir, metadata, run_id,
    test_more_threshold, test_task_list, test_task_list_more,
):
    safe_log('Start nn harness')
    effective_task_list = [task_id] if test_task_list is None else test_task_list

    dnames = nn_harness(
        test_task_list=effective_task_list,
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        max_workers=min(5, len(effective_task_list)),
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['nn_dnames'] = [str(dn) for dn in dnames]

    safe_log('Start make_report')
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        output_dir=output_dir,
    )

    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    # Avaliação adicional se score passar do threshold
    if (
        overall_performance
        and test_more_threshold is not None
        and test_task_list_more is not None
        and overall_performance.get('total_resolved_instances', 0)
            >= len(effective_task_list) * test_more_threshold
    ):
        safe_log("Start additional evaluation cycle")
        dnames = nn_harness(
            test_task_list=test_task_list_more,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            max_workers=min(5, len(test_task_list_more)),
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            output_dir=output_dir,
        )
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        safe_log("End of additional evaluation")


def self_improve(
    parent_commit='initial',
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=False,
    entry=None,               # task_id (ex: "mnist")
    test_task_list=None,      # None → usa [entry]
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    run_baseline=None,
):
    metadata = {}
    root_dir = os.path.abspath('./')
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit

    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    # Cria e inicia o container Docker
    image_name = "dgm"
    container_name = f"dgm-container-{run_id}"
    client = docker.from_env()
    remove_existing_container(client, container_name)
    container = build_dgm_container(
        client, root_dir, image_name, container_name, force_rebuild=force_rebuild,
    )
    container.start()

    # Aplica patches acumulados do pai
    patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)
    if run_baseline not in ['no_selfimprove']:
        for patch_file in patch_files:
            copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
            exec_result = container.exec_run(
                "/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm'
            )
            log_container_output(exec_result)
            container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')

    # Commit do estado atual para isolar o diff do patch gerado
    container.exec_run("git add --all", workdir='/dgm/')
    exec_result = container.exec_run(
        "git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message' --allow-empty",
        workdir='/dgm/',
    )
    log_container_output(exec_result)
    commit_output = exec_result.output.decode('utf-8')
    # Extrai o hash do commit (formato: "[master abc1234] message")
    try:
        commit_hash = commit_output.split()[1].strip("[]")
    except IndexError:
        # Fallback: pega o hash via git rev-parse
        rev_result = container.exec_run("git rev-parse HEAD", workdir='/dgm/')
        commit_hash = rev_result.output.decode('utf-8').strip()

    # Instala dependências atualizadas (ignora exit code — pip warnings são comuns)
    exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
    if exec_result.output:
        safe_log(f"pip install output: {exec_result.output.decode('utf-8')[-500:]}")

    # Gera o problem_statement para o agente via LLM diagnóstico
    if entry:
        safe_log(f"Task to improve: {entry}")
        problem_statement = diagnose_problem(
            task_id=entry,
            parent_commit=parent_commit,
            out_dir_base=out_dir_base,
            root_dir=root_dir,
            patch_files=patch_files,
            run_id=run_id,
        )
        safe_log(f"problem_statement:\n{problem_statement}")
    else:
        safe_log("No entry provided. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    metadata['entry'] = entry
    metadata['problem_statement'] = problem_statement

    if not problem_statement:
        safe_log("Failed to generate problem statement. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    # Executa o agente de self-improvement dentro do container
    safe_log("Running self-improvement")
    chat_history_file_container = "/dgm/self_evo.md"
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
        "AWS_REGION": os.getenv('AWS_REGION'),
        "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
        "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
        "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "CLAUDE_MODEL": os.getenv('CLAUDE_MODEL', ''),
    }
    cmd = [
        "timeout", "1800",  # 30 min timeout
        "python", "/dgm/nn_agent.py",
        "--self_improve",
        "--problem_statement", problem_statement,
        "--git_dir", "/dgm/",
        "--chat_history_file", chat_history_file_container,
        "--base_commit", commit_hash,
        "--outdir", "/dgm/",
    ]
    exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
    log_container_output(exec_result)

    # Copia arquivos de output de volta para o host
    chat_history_file = os.path.join(output_dir, "self_evo.md")
    try:
        copy_from_container(container, chat_history_file_container, chat_history_file)
    except FileNotFoundError:
        safe_log("Chat history file not found in container.")
    model_patch_file = os.path.join(output_dir, "model_patch.diff")
    try:
        copy_from_container(container, "/dgm/model_patch.diff", model_patch_file)
    except FileNotFoundError:
        safe_log("Model patch file not found in container. Agent failed to generate a patch.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    # Valida o patch gerado
    try:
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file does not exist")
        with open(model_patch_file, 'r') as f:
            if not f.read().strip():
                raise Exception("Model patch file is empty")
    except Exception as e:
        safe_log(f"Failed to read model patch file: {str(e)}")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    patch_files.append(model_patch_file)
    cleanup_container(container)

    # Avalia a performance pós self-improvement
    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id

    if model_patch_exists and model_patch_notempty:
        try:
            run_harness_nn(
                entry, model_name_or_path, patch_files, num_evals,
                output_dir, metadata, run_id,
                test_more_threshold, test_task_list, test_task_list_more,
            )
        except Exception as e:
            safe_log(f"Error during nn harness evaluation: {e}")

    # Diagnóstico pós-melhoria (verifica se o patch foi compilado)
    if post_improve_diagnose:
        metadata['is_compiled'] = is_compiled_self_improve(metadata)
        safe_log(f"is_compiled: {metadata['is_compiled']}")

    save_metadata(metadata, output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for DGM-NN.")
    parser.add_argument('--parent_commit', default="initial", type=str,
                        help='Run ID do pai, ou "initial" para partir do baseline')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str,
                        help='Diretório de output')
    parser.add_argument('--force_rebuild', default=False, action='store_true',
                        help='Força rebuild da imagem Docker')
    parser.add_argument('--num_evals', default=1, type=int,
                        help='Número de avaliações repetidas por tarefa')
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true',
                        help='Pula o diagnóstico pós self-improvement')
    parser.add_argument('--entry', default="mnist", type=str,
                        help='Task ID a melhorar (ex: mnist, cifar10)')
    parser.add_argument('--test_task_list', default=None, type=str,
                        help='Lista de tasks para avaliação (JSON file path)')
    args = parser.parse_args()

    # Copia o agente inicial para o diretório de output
    os.system(f"cp -r initial_nn/ {args.output_dir}")

    test_task_list = None
    if args.test_task_list:
        test_task_list = load_json_file(args.test_task_list)

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        entry=args.entry,
        test_task_list=test_task_list,
    )


if __name__ == "__main__":
    main()
