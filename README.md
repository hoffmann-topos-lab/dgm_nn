# Darwin Gödel Machine — Neural Network Trainer (DGM-NN)

Um sistema de evolução aberta onde agentes baseados em LLM se auto-aprimoram iterativamente, modificando o próprio código-fonte de um agente de treinamento de redes neurais e avaliando os resultados em benchmarks de ML.

Adaptado do [DGM original](https://arxiv.org/abs/2505.22954) (Sakana AI), que evolui um agente de coding para SWE-bench. Aqui, o que evolui é o scaffolding de treinamento de redes neurais (`nn_agent.py`): arquitetura, otimizador, data augmentation, loop de treino, etc. Os pesos do LLM permanecem congelados.

## Como funciona

```
DGM_outer.py  →  self_improve_step.py  →  nn_agent.py  →  nn_bench/ harness
     ↑                                                           |
     └──────────── atualização do archive (mantém os melhores) ─┘
```

1. O **archive** mantém os agentes com melhor accuracy (seleção Darwiniana)
2. A cada geração, um LLM analisa os logs de treinamento do agente pai e propõe melhorias
3. O agente modifica seu próprio código de treinamento (auto-referência Gödeliana)
4. O harness executa o agente modificado em Docker e mede accuracy/loss
5. Se o filho supera o pai, entra no archive

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Opcional: para rodar scripts de análise (analysis/)
pip install -r requirements_dev.txt
```

Docker deve estar rodando e acessível ao usuário atual.

Copie `.env.example` para `.env` e configure as API keys do provedor LLM escolhido.

## Uso

```bash
# Rodar o loop de evolução
python DGM_outer.py

# Flags principais
python DGM_outer.py \
  --max_generation 20 \
  --selfimprove_size 2 \
  --selfimprove_workers 1 \
  --continue_from <output_dir>  # Retomar uma run anterior
  --shallow_eval                # Avaliação rasa (1 tarefa)
  --run_baseline no_selfimprove # Ablação: sem auto-melhoria

# Rodar testes
pytest tests/
```

Outputs são salvos em `output_dgm/`.

## Estrutura do Projeto

```
DGM/
├── DGM_outer.py                    # Loop de evolução principal
├── self_improve_step.py            # Step de auto-melhoria (LLM + harness)
├── nn_agent.py                     # Agente de treinamento (evoluído pelo DGM)
├── llm.py                          # Cliente LLM unificado (Anthropic, OpenAI, etc.)
├── llm_withtools.py                # Loop agêntico com tool use
├── Dockerfile                      # Container com PyTorch CPU + datasets
├── requirements.txt
├── .env.example                    # Template de variáveis de ambiente
├── nn_bench/                       # Harness de avaliação
│   ├── harness.py                  # Execução Docker por tarefa
│   ├── tasks.py                    # Definição das tarefas (datasets, thresholds)
│   ├── report.py                   # Agregação de resultados
│   └── subsets/                    # small.json, medium.json, big.json
├── initial_nn/                     # Agente semente (baseline)
│   └── logs/
├── prompts/
│   ├── tooluse_prompt.py           # Prompt de tool use
│   ├── nn_self_improvement_prompt.py  # Diagnóstico de logs de treinamento
│   └── (legados do DGM original, mantidos como referência)
├── tools/
│   ├── bash.py                     # Ferramenta bash para o agente
│   └── edit.py                     # Ferramenta de edição para o agente
├── utils/
│   ├── docker_utils.py             # Gerenciamento de containers
│   ├── git_utils.py                # Operações git (diff, patch)
│   ├── evo_utils.py                # Lógica de archive/evolução
│   ├── common_utils.py             # Utilitários genéricos
│   └── nn_eval_utils.py            # Métricas e diagnóstico de treinamento
├── analysis/                       # Scripts de análise (referência do original)
├── tests/                          # Testes unitários
```

## Tarefas de Treinamento

| ID | Subset | Threshold | Tempo máx. |
|---|---|---|---|
| `mnist` | small | 99% | 2 min |
| `fashion_mnist` | small | 90% | 2 min |
| `cifar10` | medium | 85% | 10 min |
| `cifar100` | medium | 55% | 10 min |
| `svhn` | big | 92% | 30 min |
| `stl10` | big | 75% | 30 min |

## Aviso de Seguranca

> **Atenção**: Este sistema executa código não-confiável gerado por modelos de linguagem. Todo código gerado roda dentro de containers Docker para isolamento, mas riscos residuais existem. Use por sua conta e risco.

## Créditos

Baseado no [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) (Zhang et al., 2025, Sakana AI).

```bibtex
@article{zhang2025darwin,
  title={Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents},
  author={Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal={arXiv preprint arXiv:2505.22954},
  year={2025}
}
```
