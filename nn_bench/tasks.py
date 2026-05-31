"""
Definições das tarefas de treinamento para o DGM-NN.

Cada tarefa especifica o dataset, métrica, threshold de "resolvida" e
limites de tempo/época. O nn_agent.py recebe o task_id como problem_statement
e deve salvar um JSON de resultado com a estrutura definida em get_result_schema().
"""

TASKS = {
    # -------------------------------------------------------------------------
    # Small: treino rápido (~1-2 min). Ponto de entrada da evolução.
    # -------------------------------------------------------------------------
    "mnist": {
        "task_id": "mnist",
        "description": (
            "Classificar dígitos manuscritos (0-9) a partir de imagens em escala de cinza 28x28. "
            "Dataset: MNIST. Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.99."
        ),
        "dataset": "mnist",
        "dataset_source": "torchvision",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "metric": "accuracy",
        "threshold": 0.99,
        "max_train_time_seconds": 240,
        "max_epochs": 20,
        "size": "small",
    },
    "fashion_mnist": {
        "task_id": "fashion_mnist",
        "description": (
            "Classificar peças de vestuário (10 categorias) a partir de imagens em escala de cinza 28x28. "
            "Dataset: Fashion-MNIST. Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.90."
        ),
        "dataset": "fashion_mnist",
        "dataset_source": "torchvision",
        "input_shape": [1, 28, 28],
        "num_classes": 10,
        "metric": "accuracy",
        "threshold": 0.90,
        "max_train_time_seconds": 240,
        "max_epochs": 20,
        "size": "small",
    },

    # -------------------------------------------------------------------------
    # Medium: treino moderado (~5-10 min). Requer arquitetura mais sofisticada.
    # -------------------------------------------------------------------------
    "cifar10": {
        "task_id": "cifar10",
        "description": (
            "Classificar imagens coloridas (10 classes: avião, carro, pássaro, gato, cervo, "
            "cachorro, sapo, cavalo, navio, caminhão) a partir de imagens RGB 32x32. "
            "Dataset: CIFAR-10. Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.85."
        ),
        "dataset": "cifar10",
        "dataset_source": "torchvision",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "metric": "accuracy",
        "threshold": 0.85,
        "max_train_time_seconds": 1200,
        "max_epochs": 50,
        "size": "medium",
    },
    "cifar100": {
        "task_id": "cifar100",
        "description": (
            "Classificar imagens coloridas em 100 classes finas agrupadas em 20 superclasses, "
            "a partir de imagens RGB 32x32. "
            "Dataset: CIFAR-100. Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.55."
        ),
        "dataset": "cifar100",
        "dataset_source": "torchvision",
        "input_shape": [3, 32, 32],
        "num_classes": 100,
        "metric": "accuracy",
        "threshold": 0.55,
        "max_train_time_seconds": 1200,
        "max_epochs": 50,
        "size": "medium",
    },

    # -------------------------------------------------------------------------
    # Big: treino longo (~20-30 min). Alta resolução ou maior volume de dados.
    # -------------------------------------------------------------------------
    "svhn": {
        "task_id": "svhn",
        "description": (
            "Classificar dígitos de fotos reais de placas de rua (Street View House Numbers). "
            "Dataset: SVHN (split 'train' completo, ~73k imagens). Imagens RGB 32x32, 10 classes (0-9). "
            "Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.92."
        ),
        "dataset": "svhn",
        "dataset_source": "torchvision",
        "input_shape": [3, 32, 32],
        "num_classes": 10,
        "metric": "accuracy",
        "threshold": 0.92,
        "max_train_time_seconds": 3600,
        "max_epochs": 30,
        "size": "big",
    },
    "stl10": {
        "task_id": "stl10",
        "description": (
            "Classificar imagens de alta resolução em 10 classes (avião, pássaro, carro, gato, cervo, "
            "cachorro, cavalo, macaco, navio, caminhão) a partir de imagens RGB 96x96. "
            "Dataset: STL-10. Métrica: accuracy no conjunto de validação. "
            "Threshold para 'resolvida': accuracy >= 0.75."
        ),
        "dataset": "stl10",
        "dataset_source": "torchvision",
        "input_shape": [3, 96, 96],
        "num_classes": 10,
        "metric": "accuracy",
        "threshold": 0.75,
        "max_train_time_seconds": 3600,
        "max_epochs": 30,
        "size": "big",
    },
}


def get_task(task_id: str) -> dict:
    """Retorna a configuração de uma tarefa pelo ID."""
    if task_id not in TASKS:
        raise ValueError(f"Tarefa desconhecida: '{task_id}'. Disponíveis: {list(TASKS.keys())}")
    return TASKS[task_id]


def get_result_schema() -> dict:
    """
    Esquema esperado do JSON de resultado que o nn_agent.py deve salvar.

    O harness lê este arquivo após a execução do agente para calcular o score.
    """
    return {
        "task_id": str,           # ID da tarefa (ex: "mnist")
        "accuracy": float,        # Accuracy final no conjunto de validação (0.0 a 1.0)
        "loss": float,            # Loss final no conjunto de validação
        "epochs_trained": int,    # Número de épocas efetivamente treinadas
        "train_time_seconds": float,  # Tempo total de treinamento em segundos
        "history": {              # Métricas por época (para diagnóstico)
            "train_loss": list,       # [float, ...]
            "val_loss": list,         # [float, ...]
            "val_accuracy": list,     # [float, ...]
        },
        "error": str,             # Mensagem de erro, ou null se bem-sucedido
    }
