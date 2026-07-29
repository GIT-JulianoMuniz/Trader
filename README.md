# Trader — Descoberta Estatística de Padrões de Topos e Fundos

Framework de pesquisa quantitativa para descobrir, **sem indicadores**,
padrões recorrentes de topos/fundos (fractais, swings, BOS/CHoCH,
rompimentos falsos, sweeps de liquidez, simetria de movimentos) com
validação estatística rigorosa treino (2024) / validação (2025) / teste
out-of-sample (2026).

## Como usar

1. Coloque o CSV de candles (menor timeframe disponível, cobrindo
   2024–2026) exportado do MT5 dentro de `data/`. Formatos aceitos:
   - MT5 padrão: `<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,...`
   - Genérico: `datetime,open,high,low,close,volume`
   - Ticks (bid/ask): use `src.data_loader.ticks_to_candles` para converter
     para o menor timeframe antes de rodar a descoberta.

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Rode a descoberta:
   ```
   python run_discovery.py --csv data/SEUATIVO_M1_2024_2026.csv --symbol SEUATIVO
   ```

4. O relatório (ranking de padrões aprovados, com métricas de treino,
   validação e teste out-of-sample) é salvo em `reports/<symbol>.md`.

## Estrutura do código

- `src/data_loader.py` — carrega OHLC/ticks, converte ticks em candles,
  divide por ano (treino/validação/teste).
- `src/market_structure.py` — fractais, swing highs/lows, BOS/CHoCH,
  rompimentos falsos, amplitude/velocidade entre swings. **Nenhum
  indicador** — só máximas, mínimas e sequência de velas.
- `src/patterns.py` — motor de mineração de padrões: fundos ascendentes +
  rompimento, topo/fundo duplo, sweep de liquidez + rejeição, simetria de
  movimentos. Cada ocorrência é validada andando para frente nos candles
  reais (sem look-ahead), calculando MAE/MFE e o resultado real
  (alvo/stop/timeout).
- `src/metrics.py` — taxa de acerto, profit factor, expectância em R,
  IC 95% (Wilson), p-valor via bootstrap.
- `src/pipeline.py` — orquestra tudo e só aprova um padrão se ele tiver:
  ≥500 ocorrências, profit factor > 1.30, expectância positiva, p < 0.05
  e estabilidade entre 2024/2025/2026 — nas três fases, não só numa.

## Regras do projeto (ver `.claude/PROJECT_STATE.md`)

- Nunca usar indicadores derivados (RSI, MACD, médias móveis, Bandas de
  Bollinger, osciladores).
- Toda descoberta precisa sobreviver a validação out-of-sample real
  (2026), não é permitido reportar padrões só porque "parecem" funcionar
  no treino.

## Adicionar novos padrões

Para minerar um novo tipo de estrutura (ex: topo triplo, W, M, V, N,
consolidação seguida de expansão), adicione uma função `detect_*(df,
swings) -> list[dict]` em `src/patterns.py` retornando ocorrências com
`entry_time`, `entry_idx`, `direction`, `entry_price`, `stop_price`,
`target_price`, e registre em `PATTERN_REGISTRY`. O pipeline cuida
automaticamente do cálculo de resultado, métricas e validação
treino/val/teste.
