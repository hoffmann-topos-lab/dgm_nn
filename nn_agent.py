"""
nn_agent.py — Agente de treinamento de redes neurais do DGM-NN.

Modos de operação:
  - Treinamento (padrão): treina uma rede neural na tarefa especificada e
    salva os resultados em JSON para o harness coletar.
  - Auto-melhoria (--self_improve): usa LLM + ferramentas para editar o
    próprio código, melhorando o pipeline de treinamento.

Este arquivo é o ponto de partida da evolução. O sistema DGM modifica o
`run_training()` e os helpers abaixo ao longo das gerações.
"""

import argparse
import json
import os
import sys
import time


# ─── Modo de auto-melhoria ────────────────────────────────────────────────────

def run_self_improve(args):
    """
    Usa LLM com ferramentas bash/edit para modificar o código do agente.
    Interface compatível com self_improve_step.py.
    """
    from llm_withtools import chat_with_agent

    model = os.environ.get('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')

    initial_msg = f"""\
You are a coding agent improving a neural network training agent (nn_agent.py).

Your task:
{args.problem_statement}

Context about testing:
{args.test_description}

The agent code is in {args.git_dir}.
Focus your changes on the `run_training()` function and its helpers inside nn_agent.py.
Do NOT modify the `run_self_improve()` function or the argument parsing.

After making changes, save the git diff to {args.outdir}/model_patch.diff:
  cd {args.git_dir} && git diff {args.base_commit} > {args.outdir}/model_patch.diff
"""

    log_lines = []
    def log(msg):
        log_lines.append(str(msg))
        print(msg)

    chat_with_agent(initial_msg, model=model, logging=log)

    if args.chat_history_file:
        with open(args.chat_history_file, 'w') as f:
            f.write('\n'.join(log_lines))


# ─── Helpers de treinamento ───────────────────────────────────────────────────
# Estas funções são o alvo da evolução. O DGM as modifica para melhorar a
# accuracy ao longo das gerações.

def build_model(input_shape, num_classes):
    """
    Constrói o modelo de rede neural.
    Baseline: MLP de 2 camadas ocultas (256 → 128).
    """
    import torch.nn as nn

    input_size = 1
    for dim in input_shape:
        input_size *= dim

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def get_transforms(task):
    """
    Retorna as transformações de pré-processamento para treino e validação.
    Baseline: apenas normalização simples.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform, transform  # (train_transform, val_transform)


def get_dataloaders(task, train_transform, val_transform):
    """
    Carrega os datasets e retorna DataLoaders de treino e validação.
    """
    import torchvision
    from torch.utils.data import DataLoader

    dataset_name = task["dataset"]
    data_root = "/tmp/data"
    batch_size = 64

    DATASET_MAP = {
        "mnist":         (torchvision.datasets.MNIST,        {"download": True}),
        "fashion_mnist": (torchvision.datasets.FashionMNIST, {"download": True}),
        "cifar10":       (torchvision.datasets.CIFAR10,      {"download": True}),
        "cifar100":      (torchvision.datasets.CIFAR100,     {"download": True}),
        "svhn":          (torchvision.datasets.SVHN,         {"split": "train", "download": True}),
        "stl10":         (torchvision.datasets.STL10,        {"split": "train", "download": True}),
    }

    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Dataset não suportado: '{dataset_name}'")

    DatasetClass, extra_kwargs = DATASET_MAP[dataset_name]

    if dataset_name in ("svhn", "stl10"):
        train_dataset = DatasetClass(root=data_root, transform=train_transform, **extra_kwargs)
        val_dataset   = DatasetClass(root=data_root, split="test",  transform=val_transform, download=True)
    else:
        train_dataset = DatasetClass(root=data_root, train=True,  transform=train_transform, **extra_kwargs)
        val_dataset   = DatasetClass(root=data_root, train=False, transform=val_transform,   **extra_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=256,         shuffle=False, num_workers=2)

    return train_loader, val_loader


def get_optimizer(model, task):
    """
    Retorna o otimizador. Baseline: Adam com lr=1e-3.
    """
    import torch.optim as optim
    return optim.Adam(model.parameters(), lr=1e-3)


# ─── Loop de treinamento ──────────────────────────────────────────────────────

def run_training(task_id: str, output_file: str):
    """
    Pipeline principal de treinamento. Chama os helpers acima e salva o
    resultado JSON esperado pelo harness.
    """
    import torch
    import torch.nn as nn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nn_bench.tasks import get_task
    task = get_task(task_id)

    result = {
        "task_id": task_id,
        "accuracy": 0.0,
        "loss": float('inf'),
        "epochs_trained": 0,
        "train_time_seconds": 0.0,
        "history": {"train_loss": [], "val_loss": [], "val_accuracy": []},
        "error": None,
    }

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")

        # Inicialização
        train_transform, val_transform = get_transforms(task)
        train_loader, val_loader = get_dataloaders(task, train_transform, val_transform)
        model = build_model(task["input_shape"], task["num_classes"]).to(device)
        optimizer = get_optimizer(model, task)
        criterion = nn.CrossEntropyLoss()

        max_epochs = task["max_epochs"]
        max_time   = task["max_train_time_seconds"]
        start_time = time.time()

        for epoch in range(max_epochs):
            if time.time() - start_time > max_time:
                print(f"Limite de tempo atingido na época {epoch + 1}.")
                break

            # ── Treino ────────────────────────────────────────────────────────
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)

            # ── Validação ─────────────────────────────────────────────────────
            model.eval()
            correct, total, val_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss_sum += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    total   += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss_sum / len(val_loader)
            val_acc  = correct / total

            result["history"]["train_loss"].append(round(train_loss, 4))
            result["history"]["val_loss"].append(round(val_loss, 4))
            result["history"]["val_accuracy"].append(round(val_acc, 4))
            result["epochs_trained"] = epoch + 1

            print(f"Época {epoch+1:02d}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        elapsed = time.time() - start_time
        result["accuracy"]           = result["history"]["val_accuracy"][-1] if result["history"]["val_accuracy"] else 0.0
        result["loss"]               = result["history"]["val_loss"][-1]      if result["history"]["val_loss"]     else float('inf')
        result["train_time_seconds"] = round(elapsed, 2)

    except Exception as e:
        result["error"] = str(e)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)

    # Salva resultado
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResultado salvo em: {output_file}")
    print(json.dumps(result, indent=2))
    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="nn_agent — Agente de treinamento DGM-NN")

    # Modo de treinamento
    parser.add_argument("--task_id",     type=str, default=None,
                        help="ID da tarefa (ex: mnist, cifar10)")
    parser.add_argument("--output_file", type=str, default="/tmp/nn_result.json",
                        help="Caminho do arquivo JSON de resultado")

    # Modo de auto-melhoria (compatível com self_improve_step.py)
    parser.add_argument("--self_improve",      action="store_true",
                        help="Ativa modo de auto-melhoria via LLM")
    parser.add_argument("--problem_statement", type=str, default="",
                        help="Descrição do problema a melhorar")
    parser.add_argument("--git_dir",           type=str, default="/dgm/",
                        help="Diretório git do agente")
    parser.add_argument("--chat_history_file", type=str, default=None,
                        help="Arquivo para salvar o histórico de chat")
    parser.add_argument("--base_commit",       type=str, default="HEAD",
                        help="Commit base para o diff")
    parser.add_argument("--outdir",            type=str, default="/dgm/",
                        help="Diretório de saída dos patches")
    parser.add_argument("--test_description",  type=str, default="",
                        help="Descrição do contexto de avaliação")

    args = parser.parse_args()

    if args.self_improve:
        run_self_improve(args)
    else:
        if not args.task_id:
            parser.error("--task_id é obrigatório no modo de treinamento")
        run_training(args.task_id, args.output_file)


if __name__ == "__main__":
    main()
