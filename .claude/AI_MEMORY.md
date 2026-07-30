Project:

Institutional Trading Framework

Rules:

Never add indicators.

Never modify setup.

Always optimize implementation.

Languages:

MQL5

Python

Pine Script

Data:

Tick data (forex) local no PC do usuário: C:\bktest\forex

Esta sessão roda em container remoto na nuvem, sem acesso a esse caminho local.
Para rodar a descoberta de padrões (src/pipeline.py, run_discovery.py), os
arquivos de C:\bktest\forex precisam ser enviados para este repositório
(ex: pasta data/) via git push, ou anexados diretamente no chat.

Formato esperado: ticks (bid/ask ou bid) por ativo, cobrindo 2024-2026.
Usar src.data_loader.ticks_to_candles() para converter ticks em candles no
menor timeframe antes de rodar run_discovery.py, se necessário.
