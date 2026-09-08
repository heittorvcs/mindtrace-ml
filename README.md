# mindtrace-ml

Ciclo de vida do modelo de classificação comportamental do
[MindTrace](https://github.com/RodrigoOrvate/MindTrace) — engenharia de features,
treino, validação e exportação para ONNX.

Este repositório é deliberadamente separado do aplicativo Qt. O MindTrace continua
responsável por capturar, inferir pose e exportar dados; aqui mora tudo o que
depende de rótulos humanos e de avaliação estatística. A fronteira entre os dois
são arquivos CSV e, no sentido inverso, um `.onnx`.

```
MindTrace (Qt/C++)                    mindtrace-ml (Python)
  exportBehaviorFeatures()  ──CSV──▶   features + rótulos
  BoutEditorPanel           ──CSV──▶       │
                                           ▼
  InferenceEngine           ◀──ONNX──  modelo treinado
```

## Estado atual

Implementado o **harness de avaliação**: mede o classificador de regras existente
contra rótulos humanos, em dois níveis (quadro e bout). Ainda não há treino — o
primeiro número a produzir é o desempenho da linha de base.

## Decisão de modelagem: multi-etiqueta

Os comportamentos **coocorrem**. Um rato pode cheirar o objeto enquanto se apoia
nele: isso é `sniffing` e `rearing` no mesmo quadro. Por isso cada comportamento
é uma trilha binária independente, e não uma classe de um conjunto exclusivo.

Essa decisão tem uma consequência mensurável. A cadeia de regras do MindTrace
(`BehaviorScanner::classifySimple()`) é uma cascata por prioridade que emite
**exatamente um rótulo por quadro** — logo, em todo quadro com coocorrência ela
perde ao menos um comportamento, por construção. O harness reporta esse déficit
separadamente (`unreachable_pct`), porque é um limite estrutural, não um limiar
mal calibrado.

## Contrato de dados

### Features — vindas do MindTrace

Saída de `InferenceController::exportBehaviorFeatures()`, sem alteração. UTF-8
com BOM, 23 colunas:

```
frame, move_nose, move_body, bp_sum, bp_mean, bp_min, bp_max,
roll2s_mean, roll2s_sum, roll5s_mean, roll5s_sum, roll6s_mean, roll6s_sum,
roll7_5s_mean, roll7_5s_sum, roll15s_mean, roll15s_sum,
prob_sum, prob_mean, low_prob_01, low_prob_05, low_prob_075, rule_label
```

**`frame` não é contíguo.** O `BehaviorScanner` só registra o quadro quando ao
menos um ponto corporal tem confiança ≥ 0,75; o resto é omitido. Todo código aqui
projeta intervalos sobre os quadros efetivamente presentes, nunca sobre um range.
A função `coverage()` reporta quanto da sessão sobreviveu a esse limiar.

### Rótulos — vindos da anotação humana

Um bout por linha. O mesmo intervalo pode aparecer sob comportamentos diferentes.

```csv
session_id,animal_id,field,behavior,start_frame,end_frame,annotator
S001,R01,0,sniffing,140,210,ana
S001,R01,0,rearing,180,210,ana
S001,R01,0,unscorable,300,320,ana
```

| Campo | Observação |
|---|---|
| `behavior` | `walking`, `sniffing`, `grooming`, `resting`, `rearing` ou `unscorable` |
| `start_frame` / `end_frame` | inclusivos, na numeração da coluna `frame` das features |
| `unscorable` | oclusão ou pose perdida — excluído das métricas, não é uma sexta classe |

Marcar `unscorable` importa: forçar o anotador a escolher entre cinco opções
quando o vídeo está ocluso envenena o *ground truth* com adivinhação.

### Manifesto de sessão

```csv
session_id,animal_id,apparatus,fps,pose_model,lighting
```

O `fps` real é obrigatório: os limiares do MindTrace são em pixels **por quadro**,
e as janelas móveis assumem 30 fps fixo em código, então sessões gravadas a taxas
diferentes não são diretamente comparáveis.

## Uso

```bash
python scripts/evaluate_baseline.py \
    --features data/example/S001_campo1.csv \
    --labels   data/example/S001_campo1_labels.csv \
    --field 0
```

Os arquivos em `data/example/` são **sintéticos**, para verificar o harness antes
de existirem dados reais. Não interprete os números deles como resultado.

O relatório traz, nesta ordem: cobertura da sessão, taxa de coocorrência, teto
estrutural das regras, desempenho por quadro e desempenho por bout.

## Por que métricas de bout

Para etologia, o desfecho é contagem de episódios e tempo total — não acerto
quadro a quadro. Um modelo com 85% de acerto por quadro pode fragmentar um bout
em três e destruir a contagem. O harness reporta os dois níveis, e `count_error`
com `time_error_pct` costumam ser os números que importam para o artigo.

## Regras não negociáveis

1. **Divisão por animal.** `split_by_animal()` sorteia animais, nunca quadros nem
   sessões. Quadros vizinhos são quase idênticos: divisão aleatória entrega
   acurácia altíssima que é puro vazamento. `check_leakage()` deve sempre
   retornar lista vazia.
2. **Nunca acurácia global.** Com `resting` ocupando perto da metade dos quadros,
   um modelo que não faz nada pontua bem. Use F1 por comportamento.
3. **Rótulos são o ativo durável.** Modelo de pose, features e classificador são
   descartáveis e refazíveis; rotulagem especializada não. Rótulos ficam ancorados
   em intervalos de quadro, independentes de qual modelo de pose os gerou.

## Próximos passos

- [ ] Features de janela curta com variância (0,2 s / 0,4 s / 1 s) — `grooming` é
      definido por variância de focinho, que hoje não é calculada
- [ ] Converter limiares de px/quadro para px/segundo
- [ ] Classificadores binários por comportamento + suavização temporal
- [ ] Exportação ONNX compatível com o carregador já existente no MindTrace

## Desenvolvimento

```bash
python -m pytest tests/ -q
```
