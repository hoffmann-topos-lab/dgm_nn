"""
prompts/testrepo_prompt.py — Descrição de como testar o agente NN.

Fornece instruções sobre como executar o nn_agent.py e verificar
os resultados de treinamento.
"""


def get_test_description():
    """
    Returns a description of how to test the NN training agent.
    Used by the coding agent to understand how to verify changes.
    """
    description = (
        'The NN training agent can be tested by running:\n'
        '  `cd /dgm/ && python nn_agent.py --task_id <task_id> --output_file /tmp/result.json`\n\n'
        'Available tasks: mnist, fashion_mnist, cifar10, cifar100, svhn, stl10.\n\n'
        'The result.json file will contain accuracy, loss, epochs_trained, and training history.\n'
        'A task is considered "resolved" if accuracy >= the task threshold '
        '(e.g., 0.99 for MNIST, 0.90 for Fashion-MNIST, 0.70 for CIFAR-10).\n\n'
        'To run a quick validation, test with MNIST first (fastest task):\n'
        '  `python nn_agent.py --task_id mnist --output_file /tmp/result.json`\n'
        '  Then check: `cat /tmp/result.json` — verify accuracy and error fields.'
    )
    return description.strip()
