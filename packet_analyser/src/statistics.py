import numpy as np
import pandas as pd

from scipy import stats


def compute_moments(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    count = len(values)
    zero_count = int(np.count_nonzero(values == 0))

    return {
        "n": count,
        "min": float(values.min()) if count else np.nan,
        "max": float(values.max()) if count else np.nan,
        "median": float(np.median(values)) if count else np.nan,
        "q1": (
            float(np.percentile(values, 25))
            if count else np.nan
        ),
        "q3": (
            float(np.percentile(values, 75))
            if count else np.nan
        ),
        "mean": float(values.mean()) if count else np.nan,
        "second_raw_moment": (
            float((values * values).mean())
            if count else np.nan
        ),
        # Variance et écart-type d'échantillon.
        "variance": (
            float(values.var(ddof=1))
            if count > 1 else np.nan
        ),
        "std": (
            float(values.std(ddof=1))
            if count > 1 else np.nan
        ),
        "zero_count": zero_count,
        "zero_fraction": (
            zero_count / count
            if count else np.nan
        ),
    }


def compute_interarrival_times(packets):
    # Calculer les intervalles pour une seule capture à la fois.
    if (
        "capture_id" in packets
        and packets["capture_id"].nunique() > 1
    ):
        raise ValueError(
            "Calculer les intervalles séparément pour chaque capture."
        )

    selected = packets.sort_values(
        "timestamp",
        kind="stable",
    )

    # Garder les intervalles nuls.
    return selected["timestamp"].diff().dropna().to_numpy()


def fit_distributions(values, simulations=999, discrete=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not isinstance(simulations, int) or simulations < 1:
        raise ValueError(
            "Le nombre de simulations doit être un entier positif."
        )

    if (values < 0).any():
        raise ValueError(
            "Les tailles et les intervalles doivent être positifs ou nuls."
        )

    if discrete and (
        (values < 1) | (values != np.floor(values))
    ).any():
        raise ValueError(
            "Les tailles IP doivent être des entiers strictement positifs."
        )

    original_count = len(values)
    zero_count = int(np.count_nonzero(values == 0))

    information = {
        "n_total": original_count,
        "zero_count_total": zero_count,
        "zero_fraction_total": (
            zero_count / original_count
            if original_count else None
        ),
    }

    # Seuil pratique, pas une limite mathématique des ajustements.
    if len(values) < 10:
        return [{
            **information,
            "status": "insufficient_data",
            "reason": (
                "Moins de 10 valeurs : ajustement non effectué "
                "par choix méthodologique."
            ),
        }]

    if np.ptp(values) == 0:
        return [{
            **information,
            "status": "constant_data",
            "reason": (
                "Toutes les valeurs sont identiques : "
                "distribution ponctuelle observée, "
                "ajustements non effectués."
            ),
        }]

    # Limiter le coût des simulations.
    # Les moments sont calculés séparément sur toutes les valeurs.
    if len(values) > 2000:
        values = np.random.default_rng(42).choice(
            values,
            2000,
            replace=False,
        )

    if np.ptp(values) == 0:
        return [{
            **information,
            "status": "constant_sample",
            "n_test": len(values),
        }]

    # Références exploratoires, sans supposer qu'elles conviennent.
    names = (
        ["poisson", "geom"]
        if discrete
        else ["norm", "expon"]
    )

    results = []

    for name in names:
        distribution = getattr(stats, name)

        if discrete:
            parameters, statistic, p_value = fit_discrete(
                values,
                name,
                simulations,
            )

        else:
            # L'exponentielle commence à zéro seconde.
            known = {"loc": 0.0} if name == "expon" else None

            result = stats.goodness_of_fit(
                distribution,
                values,
                known_params=known,
                statistic="ks",
                n_mc_samples=simulations,
                rng=np.random.default_rng(42),
            )

            parameters = {
                key: float(value)
                for key, value
                in result.fit_result.params._asdict().items()
            }

            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        results.append({
            "distribution": name,
            **information,
            "sampled": original_count > len(values),
            "n_test": len(values),
            "zero_fraction_test": float(
                np.mean(values == 0)
            ),
            "parameters": parameters,
            "data_type": (
                "discrete" if discrete else "continuous"
            ),
            "ks": statistic,
            "p_value": p_value,
            "simulations": simulations,
            "assumption": (
                "Comparaison exploratoire sous hypothèse "
                "d’indépendance. Une valeur p élevée ne prouve "
                "pas la loi. Pour les temps, les zéros, arrondis "
                "et dépendances limitent le test continu. "
                "La gigue lissée est aussi dépendante dans le temps. "
                "Poisson et géométrique sont des références simples, "
                "pas des modèles physiques imposés aux tailles IP."
            ),
        })

    return results


def discrete_parameters(values, name):
    # Estimer le paramètre à partir de la moyenne.
    mean = float(np.mean(values))

    if name == "poisson":
        return {"mu": mean}

    if name == "geom":
        return {"p": 1.0 / mean}

    raise ValueError(
        f"Loi discrète non prise en charge : {name}"
    )


def discrete_distance(values, distribution, parameters):
    # Comparer les distributions cumulées avant et après chaque valeur.
    points, counts = np.unique(
        values,
        return_counts=True,
    )

    observed_after = np.cumsum(counts) / len(values)

    observed_before = (
        np.cumsum(counts) - counts
    ) / len(values)

    after = distribution.cdf(
        points,
        **parameters,
    )

    before = distribution.cdf(
        points - 1,
        **parameters,
    )

    return float(max(
        np.max(np.abs(observed_after - after)),
        np.max(np.abs(observed_before - before)),
    ))


def fit_discrete(values, name, simulations):
    distribution = getattr(stats, name)
    parameters = discrete_parameters(values, name)

    observed = discrete_distance(
        values,
        distribution,
        parameters,
    )

    rng = np.random.default_rng(42)
    exceedances = 0

    # Simuler la loi et réestimer son paramètre à chaque essai.
    # Ne pas utiliser la valeur p du KS continu sur des tailles entières.
    for _ in range(simulations):
        sample = distribution.rvs(
            **parameters,
            size=len(values),
            random_state=rng,
        )

        fitted = discrete_parameters(
            sample,
            name,
        )

        distance = discrete_distance(
            sample,
            distribution,
            fitted,
        )

        exceedances += distance >= observed

    p_value = (1 + exceedances) / (simulations + 1)

    return parameters, observed, float(p_value)


def compute_jitter(path):
    """Gigue RTP selon la récurrence de la RFC 3550."""

    dataframe = pd.read_csv(path)

    required = [
        "capture_id",
        "stream_id",
        "arrival_time",
        "rtp_timestamp",
        "clock_hz",
    ]

    if not set(required).issubset(dataframe.columns):
        raise ValueError(
            f"CSV RTP : colonnes requises {required}"
        )

    if dataframe[required].isna().any().any():
        raise ValueError("Valeurs RTP manquantes.")

    numeric = [
        "arrival_time",
        "rtp_timestamp",
        "clock_hz",
    ]

    dataframe[numeric] = dataframe[numeric].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(
        dataframe[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("Valeurs RTP non finies.")

    rows = []

    # Un flux correspond à une source RTP et une direction.
    groups = dataframe.groupby(
        ["capture_id", "stream_id"]
    )

    for (capture, stream), packets in groups:
        # Garder l'ordre d'arrivée.
        packets = packets.sort_values(
            "arrival_time",
            kind="stable",
        )

        if (
            packets["clock_hz"].nunique() != 1
            or packets["clock_hz"].iloc[0] <= 0
        ):
            raise ValueError(
                "Une fréquence RTP positive et constante "
                "par flux est requise."
            )

        rate = float(packets["clock_hz"].iloc[0])

        arrivals = packets["arrival_time"].to_numpy(
            dtype=float
        )

        timestamps = packets["rtp_timestamp"].to_numpy(
            dtype=float
        )

        invalid = (
            (timestamps < 0)
            | (timestamps >= 2**32)
            | (timestamps != np.floor(timestamps))
        )

        if invalid.any():
            raise ValueError(
                "Les horodatages RTP doivent être "
                "des entiers non signés sur 32 bits."
            )

        jitter = 0.0

        for index in range(1, len(packets)):
            # Gérer le retour du compteur RTP à zéro.
            delta_rtp = (
                int(timestamps[index])
                - int(timestamps[index - 1])
                + 2**31
            ) % 2**32 - 2**31

            # Convertir l'intervalle d'arrivée en unités RTP.
            delta_arrival = (
                arrivals[index] - arrivals[index - 1]
            ) * rate

            delta = delta_arrival - delta_rtp

            jitter += (abs(delta) - jitter) / 16.0

            rows.append({
                "capture_id": capture,
                "stream_id": stream,
                "arrival_time": arrivals[index],
                "jitter_seconds": jitter / rate,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "capture_id",
            "stream_id",
            "arrival_time",
            "jitter_seconds",
        ],
    )