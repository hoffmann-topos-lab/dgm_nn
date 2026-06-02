import argparse
import datetime
import json
import math
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from dotenv import load_dotenv

load_dotenv()

from self_improve_step import self_improve
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve


_metadata_cache = {}

# Ordem do currículo progressivo: do mais fácil ao mais difícil.
CURRICULUM_LEVELS = ['small', 'medium', 'big']


def load_metadata_cached(output_dir, commit):
    """Carrega o metadata.json de um commit, com cache em memória (ver Gargalo 3a)."""
    path = os.path.join(output_dir, commit, "metadata.json")
    if path not in _metadata_cache:
        _metadata_cache[path] = load_json_file(path)
    return _metadata_cache[path]


def _load_subset(level):
    return load_json_file(f"./nn_bench/subsets/{level}.json")


def _level_mastered(output_dir, archive, task_set):
    """True se algum agente do archive resolveu TODAS as tarefas de task_set."""
    target = set(task_set)
    if not target:
        return True
    for commit in archive:
        try:
            meta = load_metadata_cached(output_dir, commit)
            resolved = set(meta['overall_performance']['total_resolved_ids'])
        except Exception:
            continue
        if target.issubset(resolved):
            return True
    return False


def get_curriculum_tasks(output_dir, archive, max_level='big'):
    """
    Retorna a lista de tarefas 'desbloqueadas' pelo currículo progressivo.

    Começa sempre em 'small'. Um nível mais difícil só é desbloqueado quando
    algum agente do archive já resolveu TODAS as tarefas dos níveis já abertos
    (i.e., dominou o nível atual). Assim o DGM só escala para tarefas mais
    difíceis depois de resolver as mais fáceis.
    """
    levels = CURRICULUM_LEVELS[:CURRICULUM_LEVELS.index(max_level) + 1]
    unlocked = _load_subset(levels[0])
    for level in levels[1:]:
        if not _level_mastered(output_dir, archive, unlocked):
            break
        unlocked = unlocked + _load_subset(level)
    return unlocked


def initialize_run(output_dir, prevrun_dir=None):
    """Inicializa o archive e o número de geração inicial."""
    start_gen_num = 0
    if not prevrun_dir:
        archive = ['initial']
    else:
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1

    # Copia o agente baseline para o diretório de output (apenas na primeira run)
    if not prevrun_dir and not os.path.exists(os.path.join(output_dir, "initial")):
        if os.path.exists("initial_nn"):
            os.system(f"cp -r initial_nn/ {output_dir}/initial")
        else:
            raise RuntimeError(
                "Diretório 'initial_nn/' não encontrado. "
                "Execute a Fase 8 para gerar o baseline antes de iniciar a evolução."
            )

    return archive, start_gen_num


def choose_selfimproves(output_dir, archive, selfimprove_size, method='random',
                        run_baseline=None, active_tasks=None):
    """
    Escolhe os pares (parent_commit, task_id) para os self-improve attempts
    da geração atual.

    Para cada pai escolhido:
    - 25% de chance de selecionar 'solve_empty_patches' se ≥10% das tarefas
      falharam em gerar código (entry vazia).
    - Caso contrário, escolhe um task_id-alvo aleatoriamente.

    `active_tasks` (modo currículo): quando fornecido, o alvo passa a ser
    qualquer tarefa desbloqueada que o pai ainda NÃO resolveu — incluindo
    níveis recém-abertos que ele nunca tentou. Isso faz o agente avançar para
    tarefas mais difíceis só depois de resolver as fáceis e evita o impasse em
    que um pai que já resolveu tudo (unresolved_ids vazio) deixava a geração
    sem nenhuma entrada. Sem `active_tasks`, mantém o comportamento original
    (alvo = apenas tarefas avaliadas e não-resolvidas).
    """
    selfimprove_entries = []

    # Coleta métricas de cada commit do archive
    candidates = {}
    for commit in archive:
        try:
            metadata = load_metadata_cached(output_dir, commit)
            candidates[commit] = {
                'accuracy_score':      metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids':  metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                if parent_commit in candidates:
                    candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            print(f"{commit} não elegível como pai: {e}")
            continue

    if not candidates:
        return selfimprove_entries

    # Seleciona pais de acordo com o método
    if run_baseline == 'no_darwin':
        parent_commits = list(candidates.keys())[-1:]
    elif method == 'score_prop':
        commits = list(candidates.keys())
        scores = [candidates[c]['accuracy_score'] for c in commits]
        scores = [1 / (1 + math.exp(-10 * (s - 0.5))) for s in scores]
        total = sum(scores)
        probs = [s / total for s in scores]
        parent_commits = random.choices(commits, probs, k=selfimprove_size)
    elif method == 'score_child_prop':
        commits = list(candidates.keys())
        scores = [candidates[c]['accuracy_score'] for c in commits]
        scores = [1 / (1 + math.exp(-10 * (s - 0.5))) for s in scores]
        child_w = [1 / (1 + candidates[c]['children_count']) for c in commits]
        probs = [s * w for s, w in zip(scores, child_w)]
        total = sum(probs)
        probs = [p / total for p in probs]
        parent_commits = random.choices(commits, probs, k=selfimprove_size)
    elif method == 'best':
        sorted_commits = sorted(candidates, key=lambda c: candidates[c]['accuracy_score'], reverse=True)
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits += random.choices(parent_commits, k=selfimprove_size - len(parent_commits))
    else:
        parent_commits = random.choices(list(candidates.keys()), k=selfimprove_size)

    # Para cada pai, seleciona a task_id a melhorar
    for parent_commit in parent_commits:
        empty_ids    = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']
        num_total = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

        # Se muitas tarefas falharam em gerar código, tenta corrigir isso
        if num_total > 0 and len(empty_ids) >= 0.1 * num_total and random.random() < 0.25:
            selfimprove_entries.append((parent_commit, 'solve_empty_patches'))
            continue

        # Define as tarefas-alvo a melhorar
        if active_tasks is not None:
            resolved_set = set(resolved_ids)
            target_ids = [t for t in active_tasks if t not in resolved_set]
        else:
            target_ids = unresolved_ids

        # Escolhe uma tarefa-alvo aleatoriamente
        if not target_ids:
            continue
        entry = random.choice(target_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries


def filter_compiled(run_ids, output_dir, num_nn_tasks=[], logger=None):
    """
    Filtra runs que não produziram código válido ou não completaram a avaliação.
    """
    run_ids_compiled = []
    logger.info(f"num_nn_tasks esperado: {num_nn_tasks}")
    for run_id in run_ids:
        metadata = load_metadata_cached(output_dir, run_id)
        logger.info(f"{run_id} metadata: {metadata}")
        if is_compiled_self_improve(metadata, num_swe_issues=num_nn_tasks, logger=logger):
            run_ids_compiled.append(run_id)
    return run_ids_compiled


def get_original_score(output_dir):
    """Retorna o accuracy_score do agente baseline (initial)."""
    metadata = load_metadata_cached(output_dir, "initial")
    return metadata["overall_performance"]["accuracy_score"]


def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    """
    Atualiza o archive com os novos runs de self-improvement.

    - 'keep_better': só adiciona runs que superam o baseline (- noise_leeway).
    - 'keep_all':    adiciona todos.
    """
    if method == 'keep_better':
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_metadata_cached(output_dir, run_id)
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        archive += new_ids
    return archive


def get_full_eval_threshold(output_dir, archive):
    """
    Retorna o segundo maior score do archive (usado como threshold para decidir
    se vale fazer avaliação completa sobre todas as tarefas).
    """
    num_full_eval = sum(
        len(load_json_file(f"./nn_bench/subsets/{size}.json"))
        for size in ['small', 'medium', 'big']
    )

    archive_scores = [get_original_score(output_dir)]
    for run_id in archive:
        metadata = load_metadata_cached(output_dir, run_id)
        total_submitted = metadata["overall_performance"]["total_submitted_instances"]
        if total_submitted < num_full_eval * 0.9:
            continue
        archive_scores.append(metadata["overall_performance"]["accuracy_score"])

    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0]
    return max(threshold, 0.4)


def main():
    parser = argparse.ArgumentParser(description="Darwin Gödel Machine — NN Trainer")
    parser.add_argument("--max_generation",   type=int, default=80,
                        help="Número máximo de gerações de evolução.")
    parser.add_argument("--selfimprove_size", type=int, default=2,
                        help="Número de tentativas de self-improvement por geração.")
    parser.add_argument("--selfimprove_workers", type=int, default=2,
                        help="Workers paralelos para self-improvement.")
    parser.add_argument("--choose_selfimproves_method", type=str, default='score_child_prop',
                        choices=['random', 'score_prop', 'score_child_prop', 'best'],
                        help="Método de seleção de pais.")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Diretório de uma run anterior para retomar.")
    parser.add_argument("--update_archive", type=str, default='keep_all',
                        choices=['keep_better', 'keep_all'],
                        help="Estratégia de atualização do archive.")
    # Argumentos de avaliação
    parser.add_argument("--num_evals", type=int, default=1,
                        help="Avaliações repetidas por tarefa após self-improvement.")
    parser.add_argument("--post_improve_diagnose", default=False, action='store_true',
                        help="Diagnóstico pós self-improvement (verifica se patch compilou).")
    parser.add_argument("--shallow_eval", default=False, action='store_true',
                        help="Avaliação rasa: usa apenas tarefas small (mais rápido).")
    parser.add_argument("--curriculum", default=False, action='store_true',
                        help="Currículo progressivo: só avança para tarefas mais "
                             "difíceis depois que algum agente resolveu TODAS as "
                             "do nível atual. Tem precedência sobre --shallow_eval.")
    parser.add_argument("--curriculum_max_level", type=str, default='big',
                        choices=['medium', 'big'],
                        help="Nível máximo que o currículo pode desbloquear.")
    parser.add_argument("--eval_noise", type=float, default=0.1,
                        help="Margem de ruído para keep_better.")
    parser.add_argument("--no_full_eval", default=False, action='store_true',
                        help="Não executa avaliação completa (big tasks) para nenhum nó.")
    # Baselines
    parser.add_argument("--run_baseline", type=str, default=None,
                        choices=['no_selfimprove', 'no_darwin'],
                        help="Modo de ablação.")
    args = parser.parse_args()

    # Diretório de output desta run
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)
    output_dir = os.path.join("./output_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Inicializa archive
    print(f"[DEBUG] output_dir: {output_dir}")
    archive, start_gen_num = initialize_run(output_dir, prevrun_dir=args.continue_from)
    print(f"[DEBUG] archive: {archive}, start_gen_num: {start_gen_num}")

    # Verifica variáveis de ambiente
    print(f"[DEBUG] OPENAI_API_KEY: {'OK' if os.getenv('OPENAI_API_KEY') else 'FALTANDO'}")
    print(f"[DEBUG] CLAUDE_MODEL: {os.getenv('CLAUDE_MODEL', '(não setado)')}")
    print(f"[DEBUG] DIAGNOSE_MODEL: {os.getenv('DIAGNOSE_MODEL', '(não setado)')}")

    # Listas de tarefas por tamanho
    nn_tasks_sm  = load_json_file("./nn_bench/subsets/small.json")
    nn_tasks_med = load_json_file("./nn_bench/subsets/medium.json")
    print(f"[DEBUG] tasks small: {nn_tasks_sm}, medium: {nn_tasks_med}")

    logger = setup_logger(os.path.join(output_dir, "dgm_outer.log"))
    logger.info(f"Iniciando DGM-NN run {run_id} | args: {vars(args)}")
    logger.info(f"Archive inicial: {archive}")

    test_more_threshold = 0.4  # threshold para avançar para tarefas medium

    for gen_num in range(start_gen_num, args.max_generation):

        # Currículo: lista de tarefas desbloqueadas nesta geração
        if args.curriculum:
            active_tasks = get_curriculum_tasks(
                output_dir, archive, max_level=args.curriculum_max_level
            )
            logger.info(
                f"Gen {gen_num} — currículo ativo ({len(active_tasks)} tarefas): {active_tasks}"
            )
        else:
            active_tasks = None

        # Listas de avaliação efetivas desta geração
        if args.curriculum:
            eval_task_list = active_tasks
            eval_more_threshold = None      # escalada é controlada pelo currículo
            eval_task_list_more = None
        else:
            eval_task_list = nn_tasks_sm
            eval_more_threshold = None if args.shallow_eval else test_more_threshold
            eval_task_list_more = None if args.shallow_eval else nn_tasks_med

        # Seleciona (parent, task_id) para esta geração
        selfimprove_entries = choose_selfimproves(
            output_dir, archive, args.selfimprove_size,
            method=args.choose_selfimproves_method,
            run_baseline=args.run_baseline,
            active_tasks=active_tasks,
        )
        print(f"[DEBUG] Gen {gen_num} — selfimprove_entries: {selfimprove_entries}")
        logger.info(f"Gen {gen_num} — entradas de self-improve: {selfimprove_entries}")

        # Gargalo 1 — calcula o threshold UMA vez por geração (antes era recomputado
        # S× dentro da list-comprehension de futures → O(A·S) leituras por geração).
        full_eval_threshold = None if args.no_full_eval else get_full_eval_threshold(output_dir, archive)

        # Executa self-improvement em paralelo
        selfimprove_ids = []
        with ThreadPoolExecutor(max_workers=args.selfimprove_workers) as executor:
            futures = [
                executor.submit(
                    self_improve,
                    parent_commit=parent_commit,
                    output_dir=output_dir,
                    force_rebuild=False,
                    num_evals=args.num_evals,
                    post_improve_diagnose=args.post_improve_diagnose,
                    entry=entry,
                    test_task_list=eval_task_list,
                    test_more_threshold=eval_more_threshold,
                    test_task_list_more=eval_task_list_more,
                    full_eval_threshold=full_eval_threshold,
                    run_baseline=args.run_baseline,
                )
                for parent_commit, entry in selfimprove_entries
            ]

            for future in as_completed(futures):
                try:
                    metadata = future.result(timeout=5 * 60 * 60)  # 5h timeout
                    selfimprove_ids.append(metadata['run_id'])
                    print(f"[DEBUG] Self-improve concluído: {metadata['run_id']}")
                except TimeoutError:
                    print("[DEBUG] Self-improvement attempt timed out.")
                    logger.error("Self-improvement attempt timed out.")
                    future.cancel()
                except Exception as e:
                    print(f"[DEBUG] Self-improvement step failed: {e}")
                    print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
                    logger.error(f"Self-improvement step failed: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

        # Filtra runs inválidos e atualiza o archive
        logger.info(f"Atualizando archive para geração {gen_num}")
        if args.curriculum:
            num_nn_tasks = [len(active_tasks)]
        elif args.shallow_eval:
            num_nn_tasks = [len(nn_tasks_sm)]
        else:
            num_nn_tasks = [len(nn_tasks_sm), len(nn_tasks_med)]
        selfimprove_ids_compiled = filter_compiled(
            selfimprove_ids, output_dir, num_nn_tasks=num_nn_tasks, logger=logger
        )
        archive = update_archive(
            output_dir, archive, selfimprove_ids_compiled,
            method=args.update_archive, noise_leeway=args.eval_noise,
        )

        # Persiste estado da geração
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps({
                "generation":           gen_num,
                "selfimprove_entries":  selfimprove_entries,
                "children":             selfimprove_ids,
                "children_compiled":    selfimprove_ids_compiled,
                "archive":              archive,
            }, indent=2) + "\n")

        print(f"[DEBUG] Gen {gen_num} concluída | archive: {archive}")
        logger.info(f"Gen {gen_num} concluída | archive: {archive}")


if __name__ == "__main__":
    main()
