"""Concordância entre uma medida automática e a manual, sessão a sessão.

A pergunta "o automático é igual ao manual?" tem três respostas diferentes, e
um artigo precisa das três:

- **Viés** — em média, o automático mede a mais ou a menos? Teste t pareado e
  Wilcoxon. Não achar diferença **não** prova igualdade: com poucas sessões
  quase nada dá significativo.
- **Equivalência** — a diferença média cabe numa margem que o laboratório
  aceita? Dois testes unilaterais (TOST): equivalente se o intervalo de 90% da
  diferença média cabe inteiro em ±margem. A margem é decisão científica, e
  por isso também se reporta a menor margem que os dados sustentam.
- **Concordância sessão a sessão** — a média pode bater com cada sessão errando
  muito para cima ou para baixo. Limites de Bland-Altman (onde caem 95% das
  diferenças de uma sessão) e o coeficiente de concordância de Lin, que ao
  contrário da correlação de Pearson cai também quando há viés.
"""

import numpy as np
from scipy import stats


def concordance(manual: np.ndarray, automatic: np.ndarray) -> float:
    """Coeficiente de concordância de Lin: 1 só se os pontos caem na diagonal."""
    covariance = np.mean((manual - manual.mean()) * (automatic - automatic.mean()))
    return 2 * covariance / (manual.var() + automatic.var() + (manual.mean() - automatic.mean()) ** 2)


def agreement(manual, automatic, margin_fraction: float = 0.15) -> dict:
    manual, automatic = np.asarray(manual, float), np.asarray(automatic, float)
    n = len(manual)
    difference = automatic - manual
    mean, sd = difference.mean(), difference.std(ddof=1)
    se = sd / np.sqrt(n)
    reference = manual.mean()
    margin = margin_fraction * reference

    t90 = stats.t.ppf(0.95, n - 1)
    t95 = stats.t.ppf(0.975, n - 1)
    low90, high90 = mean - t90 * se, mean + t90 * se
    # TOST: as duas hipóteses nulas são "diferença ≤ −margem" e "≥ +margem".
    p_lower = 1 - stats.t.cdf((mean + margin) / se, n - 1)
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)

    return {
        "n": n,
        "manual_mean": reference,
        "automatic_mean": automatic.mean(),
        "bias": mean,
        "bias_pct": 100 * mean / reference,
        "ci95": (mean - t95 * se, mean + t95 * se),
        "paired_t_p": stats.ttest_rel(automatic, manual).pvalue,
        "wilcoxon_p": stats.wilcoxon(automatic, manual).pvalue if n >= 5 and np.any(difference) else np.nan,
        "margin": margin,
        "tost_p": max(p_lower, p_upper),
        "smallest_margin_pct": 100 * max(abs(low90), abs(high90)) / reference,
        "limits_of_agreement": (mean - 1.96 * sd, mean + 1.96 * sd),
        "limits_pct": (100 * (mean - 1.96 * sd) / reference, 100 * (mean + 1.96 * sd) / reference),
        "ccc": concordance(manual, automatic),
        "pearson_r": stats.pearsonr(manual, automatic)[0] if n > 2 else np.nan,
        # Sessões sem nenhum episódio marcado ficam de fora: não há erro relativo a zero.
        "mean_abs_pct_error": 100 * np.nanmean(np.abs(difference) / np.where(manual > 0, manual, np.nan)),
    }
