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


def total_error(manual, automatic, sessions=None) -> dict:
    """Erro do total por sessão: o MSE e de que ele é feito.

    O MSE se decompõe em viés² mais espalhamento (a variância das diferenças).
    Viés baixo com espalhamento alto quer dizer que a média bate, mas cada sessão
    erra para um lado. Em contagem, o RMSE pesa mais as sessões com muitos
    episódios, por isso vai junto o erro relativo mediano (com o total marcado
    de pelo menos 1, para uma sessão sem episódios não dividir por zero). Com
    `sessions`, diz também qual sessão pesa mais no MSE.
    """
    manual, automatic = np.asarray(manual, float), np.asarray(automatic, float)
    error = automatic - manual
    mse = float(np.mean(error ** 2))
    bias, spread = float(error.mean()), float(error.var())
    result = {
        "n": len(error),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": bias,
        "spread": spread,
        "bias_share": bias ** 2 / mse if mse else 0.0,
        "manual_mean": float(manual.mean()),
        "relative_median": float(np.median(np.abs(error) / np.maximum(manual, 1.0))),
    }
    if sessions is not None and mse:
        share = error ** 2 / np.sum(error ** 2)
        worst = int(np.argmax(share))
        result["worst_session"], result["worst_share"] = list(sessions)[worst], float(share[worst])
    return result


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
