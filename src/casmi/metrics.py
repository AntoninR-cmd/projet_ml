def _validate_inputs(true_keys, ranked_keys, k):
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k doit être un entier.")

    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    if len(true_keys) != len(ranked_keys):
        raise ValueError(
            "true_keys et ranked_keys doivent contenir le même nombre de requêtes."
        )

    if len(true_keys) == 0:
        raise ValueError("Il faut au moins une requête pour calculer la métrique.")

    if any(key is None for key in true_keys):
        raise ValueError("Chaque requête doit avoir une vraie clé moléculaire.")


def mrr_at_k(true_keys, ranked_keys, k=25):
    """Calcule le MRR@k à partir des vraies clés et des listes classées."""
    _validate_inputs(true_keys, ranked_keys, k)

    reciprocal_ranks = []

    for true_key, candidates in zip(true_keys, ranked_keys):
        reciprocal_rank = 0.0

        for rank, candidate in enumerate(candidates[:k], start=1):
            if candidate == true_key:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(true_keys, ranked_keys, k):
    """Calcule la proportion de requêtes dont la bonne clé apparaît dans le top k."""
    _validate_inputs(true_keys, ranked_keys, k)

    hits = 0

    for true_key, candidates in zip(true_keys, ranked_keys):
        if any(candidate == true_key for candidate in candidates[:k]):
            hits += 1

    return hits / len(true_keys)