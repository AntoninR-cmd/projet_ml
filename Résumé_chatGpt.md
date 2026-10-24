# Projet CASMI 2026 – Identification de molécules à partir de spectres MS/MS

## 1. Contexte du projet

Le projet s’appuie sur la compétition Kaggle :

**Enveda CASMI 2026 – Molecule ID From Mass Spectra**

L’objectif est d’identifier la structure chimique 2D d’une molécule à partir d’un ou plusieurs spectres de masse en tandem MS/MS.

### Problématique retenue

> Dans quelle mesure peut-on apprendre, à partir de spectres de masse MS/MS, une représentation permettant d’identifier et de classer des structures moléculaires candidates, y compris pour des molécules absentes du jeu d’apprentissage ?

Cette problématique est liée aux sciences du vivant, notamment à la métabolomique et à l’identification de molécules présentes dans des échantillons biologiques.

---

# 2. Données disponibles

Le jeu d’entraînement contient :

- 2 539 608 spectres MS/MS
- environ 275 810 molécules distinctes
- environ 2,8 Go de données

Un individu correspond à un spectre MS/MS.

Une même molécule peut avoir plusieurs spectres, obtenus dans différentes conditions expérimentales.

Le test final contient environ :

- 400 molécules
- 1 500 spectres

Le problème n’est donc pas une simple classification entre les molécules présentes dans le train : certaines molécules du test peuvent ne jamais avoir été observées dans les données d’apprentissage.

---

# 3. Principales variables

Les principales variables étudiées sont les suivantes.

## Structure de la molécule

### `normalized_smiles`

Représentation textuelle standardisée de la structure moléculaire.

Exemple :

```text
CC(=O)Oc1ccccc1C(=O)O
```

Disponible dans le train et utilisée comme vérité terrain.

### `inchikey`

Identifiant standardisé complet de la structure.

### `inchikey14`

Premiers 14 caractères de l’InChIKey.

Il représente principalement la connectivité 2D de la molécule.

Il est particulièrement utile pour :

- identifier les molécules distinctes ;
- regrouper les différents spectres d’une même molécule ;
- éviter les fuites d’information lors de la validation ;
- calculer la métrique locale.

---

# 4. Compréhension du spectre MS/MS

Le principe général est :

```text
Molécule
   ↓
Ionisation
   ↓
Ion précurseur
   ↓
Collision
   ↓
Fragmentation
   ↓
Fragments détectés
   ↓
Spectre MS/MS
```

Un spectre contient essentiellement des couples :

```text
m/z du fragment
intensité du fragment
```

Les deux principales colonnes sont :

### `ms2_mzs`

Liste des rapports masse/charge des fragments détectés.

Exemple :

```python
[77.03, 91.05, 105.04, 121.03]
```

### `ms2_normalized_intensities`

Intensités relatives correspondant exactement aux mêmes pics.

Exemple :

```python
[0.12, 0.63, 1.0, 0.43]
```

Le pic d’intensité `1.0` est le plus intense du spectre après normalisation.

---

# 5. Précurseur et adduct

## `precursor_mz`

Rapport masse/charge de l’ion sélectionné avant fragmentation.

Il ne correspond pas forcément directement à la masse de la molécule neutre.

## `adduct`

Décrit la forme ionisée observée.

Exemples :

```text
[M+H]+
[M-H]-
[M+Na]+
[2M+Ca-H]+
[M-C4H11O2]+
[M-H5O3]+
```

`M` représente la molécule neutre.

Exemple :

```text
[M+H]+
```

signifie que la molécule a gagné un proton.

---

# 6. Reconstruction de la masse neutre

L’idée est d’utiliser :

```text
precursor_mz
+
adduct
```

pour retrouver approximativement la masse neutre de la molécule.

Cette information sera ensuite très utile pour filtrer les structures candidates.

## Masse d’une formule chimique

Pour une formule comme :

```text
H5O3
```

la masse monoisotopique est :

\[
5 \times m(H) + 3 \times m(O)
\]

Même principe pour :

```text
C4H11O2
```

\[
4m(C) + 11m(H) + 2m(O)
\]

On utilise les masses monoisotopiques des éléments.

---

# 7. Parsing automatique des adducts

Plutôt que de créer manuellement une règle pour chaque adduct, l’objectif est de lire automatiquement leur structure.

Exemple :

```text
[2M+Ca-H]+
```

doit être décomposé en :

```text
2M
+Ca
-H
```

Avec Python :

```python
import re

inside = "2M+Ca-H"

parts = re.findall(r"[+-]?[^+-]+", inside)

print(parts)
```

Résultat :

```python
['2M', '+Ca', '-H']
```

---

# 8. Expressions régulières utilisées

Le module Python utilisé est :

```python
import re
```

Une expression régulière permet de rechercher des motifs dans du texte.

Le préfixe :

```python
r"..."
```

signifie `raw string`.

Il évite que Python interprète certains caractères comme `\`.

Exemple :

```python
r"\d+"
```

signifie :

> un ou plusieurs chiffres.

Pour extraire les éléments chimiques :

```python
re.findall(
    r"([A-Z][a-z]?)(\d*)",
    "C4H11O2"
)
```

renvoie :

```python
[
    ("C", "4"),
    ("H", "11"),
    ("O", "2")
]
```

---

# 9. Fonction actuelle de calcul de masse neutre

La fonction repose sur RDKit pour obtenir automatiquement la masse monoisotopique des éléments.

```python
import re
from rdkit import Chem

ELECTRON_MASS = 0.000548579909065
PERIODIC_TABLE = Chem.GetPeriodicTable()


def get_atom_mass(symbol):

    atomic_number = PERIODIC_TABLE.GetAtomicNumber(symbol)

    if atomic_number == 0:
        raise ValueError(f"Atome inconnu : {symbol}")

    return PERIODIC_TABLE.GetMostCommonIsotopeMass(
        atomic_number
    )


def formula_mass(formula):

    match = re.fullmatch(
        r"(\d*)([A-Za-z0-9]+)",
        formula
    )

    if match is None:
        raise ValueError(
            f"Formule non reconnue : {formula}"
        )

    global_coef = (
        int(match.group(1))
        if match.group(1)
        else 1
    )

    formula_body = match.group(2)

    elements = re.findall(
        r"([A-Z][a-z]?)(\d*)",
        formula_body
    )

    mass = 0.0

    for symbol, count in elements:

        count = int(count) if count else 1

        mass += (
            get_atom_mass(symbol)
            * count
        )

    return global_coef * mass


def get_neutral_mass(precursor_mz, adduct):

    match = re.search(
        r"\[(.*?)\]",
        adduct
    )

    if match is None:
        raise ValueError(
            f"Adduct invalide : {adduct}"
        )

    inside = match.group(1)

    parts = re.findall(
        r"[+-]?[^+-]+",
        inside
    )

    if len(parts) == 0:
        raise ValueError(
            f"Adduct vide : {adduct}"
        )

    m_part = parts[0]

    m_match = re.fullmatch(
        r"(\d*)M",
        m_part
    )

    if m_match is None:
        raise ValueError(
            f"Premier bloc non reconnu comme M : {m_part}"
        )

    coeff_mol = (
        int(m_match.group(1))
        if m_match.group(1)
        else 1
    )

    relative_sum = 0.0

    for part in parts[1:]:

        if part.startswith("+"):
            sign = +1
            formula = part[1:]

        elif part.startswith("-"):
            sign = -1
            formula = part[1:]

        else:
            sign = +1
            formula = part

        relative_sum += (
            sign
            * formula_mass(formula)
        )

    charge_match = re.search(
        r"\](\d*)([+-])$",
        adduct
    )

    if charge_match is None:
        raise ValueError(
            f"Charge non reconnue : {adduct}"
        )

    charge_magnitude = (
        int(charge_match.group(1))
        if charge_match.group(1)
        else 1
    )

    charge_sign = (
        +1
        if charge_match.group(2) == "+"
        else -1
    )

    z = charge_sign * charge_magnitude

    ion_mass = (
        abs(z)
        * precursor_mz
    )

    neutral_mass = (
        ion_mass
        - relative_sum
        + z * ELECTRON_MASS
    ) / coeff_mol

    return neutral_mass
```

---

# 10. Test du parser sur les adducts

Avant d’appliquer la fonction à toutes les observations, il faut la tester sur tous les adducts uniques.

```python
adducts_unique = (
    meta["adduct"]
    .dropna()
    .unique()
)

for adduct in adducts_unique:

    try:

        mass = get_neutral_mass(
            precursor_mz=500.0,
            adduct=adduct
        )

        print(
            f"OK {adduct:25s} -> {mass:.6f}"
        )

    except Exception as e:

        print(
            f"ERREUR {adduct:25s} -> {e}"
        )
```

L’objectif est de détecter les syntaxes d’adduct non encore prises en charge.

---

# 11. Ajout de la masse neutre au dataset

Lorsque le parser est validé :

```python
meta["neutral_mass"] = meta.apply(
    lambda row: get_neutral_mass(
        row["precursor_mz"],
        row["adduct"]
    ),
    axis=1
)
```

Pour une version robuste :

```python
def safe_get_neutral_mass(row):

    try:

        return get_neutral_mass(
            row["precursor_mz"],
            row["adduct"]
        )

    except Exception:

        return np.nan
```

Puis :

```python
meta["neutral_mass"] = meta.apply(
    safe_get_neutral_mass,
    axis=1
)
```

---

# 12. Valeurs manquantes

Les variables étudiées présentant des valeurs manquantes étaient notamment :

```python
cols_na = [
    "instrument_type",
    "base_peak_intensity",
    "collision_energy_ev",
    "collision_energy_orig"
]
```

Un test a montré que :

```python
train[cols_na].isna().all(axis=1).sum()
==
train[cols_na].isna().any(axis=1).sum()
```

est vrai.

Cela signifie que lorsqu’une de ces quatre variables est manquante, elles le sont toutes.

Une analyse par bibliothèque avait donné :

```text
drug_plus     100 % NA
enveda-180      0 % NA
```

Attention toutefois à vérifier que cette observation est bien faite sur l’ensemble du dataset et pas seulement sur un row group.

Conclusion actuelle :

- ne pas supprimer automatiquement les observations ;
- considérer le missingness comme potentiellement lié à la source des données ;
- tester plusieurs stratégies pendant la validation.

---

# 13. `base_peak_intensity`

La `base_peak_intensity` représente l’intensité absolue du pic le plus intense avant normalisation.

Elle ne peut pas être reconstruite à partir de :

```text
ms2_mzs
ms2_normalized_intensities
precursor_mz
```

car la normalisation détruit l’échelle absolue.

Par exemple :

```text
[0.1, 0.5, 1.0]
```

peut correspondre à :

```text
[10, 50, 100]
```

ou :

```text
[100, 500, 1000]
```

---

# 14. Validation croisée

Il ne faut surtout pas utiliser un KFold classique sur les lignes.

Une même molécule peut posséder plusieurs spectres.

Si certains spectres d’une molécule étaient dans le train et d’autres dans la validation, le score serait artificiellement optimiste.

La validation est donc réalisée avec :

```python
GroupKFold
```

et :

```python
groups = meta["inchikey14"]
```

Exemple :

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(
    n_splits=5
)

meta["fold"] = -1

for fold, (_, val_idx) in enumerate(
    gkf.split(
        meta,
        groups=meta["inchikey14"]
    )
):

    meta.loc[
        val_idx,
        "fold"
    ] = fold
```

Vérification :

```python
for fold in range(5):

    train_keys = set(
        meta.loc[
            meta.fold != fold,
            "inchikey14"
        ]
    )

    val_keys = set(
        meta.loc[
            meta.fold == fold,
            "inchikey14"
        ]
    )

    print(
        fold,
        len(train_keys & val_keys)
    )
```

Le résultat attendu est :

```text
0
```

pour chaque fold.

---

# 15. Pourquoi ne pas prédire directement `inchikey14` ?

Une classification classique ferait :

```text
spectre
→
classe parmi 275 000 molécules
```

Mais cette stratégie ne permet pas de prédire une molécule jamais vue pendant l’apprentissage.

Avec le GroupKFold :

```text
molécules train
∩
molécules validation
=
∅
```

Un classifieur classique ne pourrait donc jamais prédire la bonne classe de validation.

---

# 16. Cible retenue : Morgan fingerprint

La première approche retenue consiste à prédire une représentation de la structure chimique.

Pipeline :

```text
spectre MS/MS
        ↓
modèle
        ↓
Morgan fingerprint
```

Un Morgan fingerprint est un vecteur binaire représentant différents motifs chimiques présents dans la molécule.

Exemple :

```text
001001000101001...
```

On peut utiliser :

```python
FP_SIZE = 1024
```

ou plus tard :

```python
FP_SIZE = 2048
```

---

# 17. Construction des fingerprints

```python
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 1024

morgan_generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=2,
        fpSize=FP_SIZE
    )
)


def smiles_to_fp(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = (
        morgan_generator
        .GetFingerprint(mol)
    )

    arr = np.zeros(
        FP_SIZE,
        dtype=np.uint8
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    return arr
```

---

# 18. Calculer une seule fois le fingerprint par molécule

Il serait inutile de recalculer le fingerprint pour les 2,5 millions de spectres.

On crée d’abord :

```python
molecules = (
    meta[
        [
            "inchikey14",
            "normalized_smiles"
        ]
    ]
    .drop_duplicates(
        "inchikey14"
    )
    .reset_index(drop=True)
)
```

Puis :

```python
molecules["fingerprint"] = (
    molecules[
        "normalized_smiles"
    ]
    .apply(smiles_to_fp)
)
```

Enfin :

```python
fp_by_key = dict(
    zip(
        molecules["inchikey14"],
        molecules["fingerprint"]
    )
)
```

On peut alors retrouver directement le fingerprint cible d’un spectre à partir de son `inchikey14`.

---

# 19. Binning des spectres

Les spectres ont un nombre variable de pics.

Exemple :

```text
spectre A : 40 pics
spectre B : 125 pics
spectre C : 300 pics
```

Un MLP ou XGBoost ne peut pas directement utiliser ces listes de tailles différentes.

On transforme donc chaque spectre en un vecteur de taille fixe.

---

# 20. Exemple concret de binning

Spectre :

| m/z | intensité |
|---:|---:|
| 12.1 | 0.2 |
| 18.7 | 1.0 |
| 24.2 | 0.4 |
| 37.9 | 0.7 |

Avec des bins de 10 Da :

```text
[0,10[
[10,20[
[20,30[
[30,40[
```

On obtient :

```text
[0, 1.0, 0.4, 0.7]
```

Si plusieurs pics tombent dans le même bin, on peut conserver l’intensité maximale.

---

# 21. Paramètres envisagés pour le binning

Première baseline :

```python
MAX_MZ = 1200
BIN_WIDTH = 0.5
```

Nombre de variables :

\[
1200 / 0.5 = 2400
\]

Chaque spectre devient donc :

```text
2400 variables
```

---

# 22. Fonction de binning

```python
def bin_spectrum(
    mzs,
    intensities,
    max_mz=1200,
    bin_width=0.5
):

    n_bins = int(
        np.ceil(
            max_mz / bin_width
        )
    )

    x = np.zeros(
        n_bins,
        dtype=np.float32
    )

    mzs = np.asarray(mzs)
    intensities = np.asarray(
        intensities
    )

    mask = (
        (mzs >= 0)
        &
        (mzs < max_mz)
    )

    mzs = mzs[mask]
    intensities = (
        intensities[mask]
    )

    bins = (
        mzs / bin_width
    ).astype(int)

    np.maximum.at(
        x,
        bins,
        intensities
    )

    return x
```

---

# 23. Construction de X et y

Pour un fold donné :

```python
VALID_FOLD = 0

train_fold = meta[
    meta["fold"] != VALID_FOLD
]

val_fold = meta[
    meta["fold"] == VALID_FOLD
]
```

Puis :

```python
X_train = np.stack([
    bin_spectrum(
        mzs,
        intensities
    )
    for mzs, intensities
    in zip(
        train_fold["ms2_mzs"],
        train_fold[
            "ms2_normalized_intensities"
        ]
    )
])
```

Même chose pour validation.

Les cibles :

```python
y_train = np.stack([
    fp_by_key[key]
    for key
    in train_fold["inchikey14"]
])
```

Ainsi :

```text
X_train
=
spectres binned

y_train
=
Morgan fingerprints
```

---

# 24. Ajout de la masse neutre comme variable

Une première possibilité est d’ajouter :

```text
neutral_mass
```

au vecteur X.

Comme la masse est de l’ordre de plusieurs centaines tandis que les intensités sont comprises entre 0 et 1, il faut la normaliser.

Exemple :

```python
train_mass = (
    train_fold["neutral_mass"]
    .fillna(0)
    .to_numpy(
        dtype=np.float32
    )
    / 1200
)
```

Puis :

```python
X_train = np.column_stack([
    X_train,
    train_mass
])
```

Le vecteur contient alors :

```text
2400 bins
+
1 masse neutre
```

---

# 25. Attention à la mémoire

Créer directement :

```text
2 539 608 × 2400
```

valeurs en `float32` demanderait environ :

```text
24 Go
```

rien que pour X.

Il ne faudra donc pas créer toute la matrice en mémoire.

Approche prévue :

- prototype sur un sous-échantillon ;
- puis PyTorch ;
- chargement par batches ;
- transformation du spectre au moment du chargement.

---

# 26. Modèle initial envisagé

Première baseline :

```text
MS/MS binned
+
neutral_mass
        ↓
MLP
        ↓
Morgan fingerprint prédit
```

Le réseau produira un vecteur de probabilités :

```text
vrai :
0 1 0 1 0 ...

prévu :
0.03 0.92 0.10 0.81 0.02 ...
```

Une loss possible est :

```text
Binary Cross Entropy
```

car chaque bit du fingerprint constitue une cible binaire.

---

# 27. Pourquoi la loss n’est pas la métrique finale

La BCE permet d’entraîner le modèle mais la compétition demande une liste de molécules.

Il faudra donc transformer le fingerprint prédit en candidats moléculaires.

Pipeline complet :

```text
spectre
   ↓
modèle
   ↓
fingerprint prédit
   ↓
liste de molécules candidates
   ↓
similarité
   ↓
classement
   ↓
Top 25
   ↓
MRR@25
```

---

# 28. Candidate database

Pour le premier prototype, la base de candidats sera constituée des molécules uniques du dataset.

```python
candidate_db = (
    meta[
        [
            "inchikey14",
            "normalized_smiles"
        ]
    ]
    .drop_duplicates(
        "inchikey14"
    )
    .reset_index(drop=True)
)
```

Cela permet de créer une validation dite `closed-world` :

> la structure de validation est présente dans la base de candidats, mais aucun spectre de cette molécule n’a été utilisé pour entraîner le modèle.

---

# 29. Filtrage par masse

La base contient environ 275 000 molécules.

Il serait inefficace de comparer chaque fingerprint prédit avec toutes les molécules.

La masse neutre permet de filtrer les candidats :

```text
275 000 molécules
       ↓
filtre par masse
       ↓
quelques centaines / milliers
       ↓
comparaison des fingerprints
```

Une tolérance en ppm pourra être utilisée.

---

# 30. PubChem et COCONUT

À terme, la base locale de candidats sera insuffisante.

Certaines molécules du test ne sont pas présentes parmi les structures du train.

Les bases externes envisagées sont :

```text
PubChem
COCONUT
```

COCONUT est notamment spécialisée dans les produits naturels.

Pipeline futur :

```text
train
+
COCONUT
+
PubChem
      ↓
candidate database
```

L’ajout de ces bases augmente la couverture des structures possibles, mais rend également le classement plus difficile car le nombre de candidats augmente fortement.

---

# 31. Trois niveaux de difficulté

Le problème peut être vu comme trois situations.

## Cas 1

La molécule possède déjà un spectre connu.

Approche :

```text
spectral library search
```

## Cas 2

La molécule n’a pas de spectre connu mais sa structure est présente dans une base.

Approche :

```text
spectre
→
fingerprint
→
candidate retrieval
```

C’est le cas principalement simulé par notre validation actuelle.

## Cas 3

La structure elle-même est inconnue des bases.

Il faudra alors utiliser une approche de génération :

```text
spectrum → SMILES
```

ou :

```text
spectrum → graph
```

Ce cas sera traité plus tard.

---

# 32. MRR@25

Pour chaque molécule, on peut proposer jusqu’à 25 candidats.

Si la bonne molécule est :

```text
rang 1
```

score :

\[
1
\]

Si elle est :

```text
rang 2
```

score :

\[
1/2
\]

Si elle est :

```text
rang 3
```

score :

\[
1/3
\]

Si elle n’est pas présente dans les 25 premiers :

\[
0
\]

Fonction locale :

```python
def reciprocal_rank_at_25(
    true_inchikey14,
    predicted_inchikey14
):

    for rank, candidate in enumerate(
        predicted_inchikey14[:25],
        start=1
    ):

        if candidate == true_inchikey14:
            return 1.0 / rank

    return 0.0
```

MRR :

```python
def mrr_at_25(
    y_true,
    y_pred
):

    scores = []

    for true_key, candidates in zip(
        y_true,
        y_pred
    ):

        scores.append(
            reciprocal_rank_at_25(
                true_key,
                candidates
            )
        )

    return np.mean(scores)
```

---

# 33. Candidate Recall

Le MRR ne suffit pas pour comprendre où le pipeline échoue.

Il faut également mesurer si la vraie molécule est présente dans les candidats.

Métriques envisagées :

```text
Recall@25
Recall@100
Recall@500
MRR@25
```

Exemple :

```text
Recall@500 = 92 %
```

signifie que la vraie molécule apparaît parmi les 500 candidats pour 92 % des molécules.

Le ranker ne peut jamais identifier correctement une structure absente de sa liste de candidats.

---

# 34. Agrégation des différents spectres d’une molécule

Une molécule peut avoir plusieurs spectres.

Exemple :

```text
Molécule X

spectre 1
spectre 2
spectre 3
```

Le modèle donnera :

```text
FP1
FP2
FP3
```

Une première méthode simple consiste à calculer :

\[
FP_X =
\frac{FP_1 + FP_2 + FP_3}{3}
\]

Puis ce fingerprint agrégé est utilisé pour rechercher les candidats.

Des méthodes plus avancées pourront être testées plus tard.

---

# 35. Modèles envisagés

Ordre de progression envisagé :

### Baseline 1

Similarité spectrale / nearest neighbour.

### Baseline 2

Spectres binned + MLP.

### Modèle 3

Spectre → Morgan fingerprint.

### Modèle 4

Embeddings DreaMS.

### Modèle 5

Transformer sur les pics `(m/z, intensité)`.

### Modèle 6

Dual encoder :

```text
spectrum encoder
+
molecule encoder
```

avec apprentissage contrastif.

### Ranking final

Utilisation possible de :

```text
LightGBM
XGBoost
LambdaMART
```

pour classer les candidats selon plusieurs features :

- erreur de masse ;
- similarité du fingerprint ;
- similarité spectrale ;
- adduct ;
- neutral losses ;
- embedding DreaMS ;
- formule moléculaire prédite ;
- score du modèle.

---

# 36. Ce qu’il ne faut pas faire

Ne pas utiliser directement :

```text
inchikey14
```

comme classe d’un Random Forest ou XGBoost.

Ne pas utiliser les variables dérivées de la vraie structure comme features.

Variables à risque de fuite :

```text
normalized_smiles
inchikey
inchikey14
molecular_formula
precursor_error_ppm
```

`precursor_error_ppm` est notamment calculé à partir de la structure connue et ne sera pas disponible de la même manière lorsqu’on cherche justement la structure.

---

# 37. État actuel du projet

Les étapes suivantes ont déjà été réalisées ou définies :

- téléchargement et chargement des données ;
- compréhension générale des variables ;
- étude des valeurs manquantes ;
- compréhension du rôle du précurseur et des adducts ;
- création des folds groupés par `inchikey14` ;
- définition de la stratégie de validation ;
- choix d’une première cible sous forme de Morgan fingerprint ;
- définition du binning MS/MS ;
- conception d’un parser générique des adducts ;
- calcul de la masse neutre à partir de `precursor_mz` et `adduct` ;
- réflexion sur la candidate database ;
- définition du calcul du MRR@25 ;
- définition du pipeline global.

---

# 38. Pipeline actuellement retenu

```text
              DONNÉES MS/MS
                    │
                    ▼
             Prétraitement
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Binning MS/MS      precursor_mz
                              +
                            adduct
                              │
                              ▼
                       neutral_mass
          │                   │
          └─────────┬─────────┘
                    ▼
                   MLP
                    │
                    ▼
          Morgan fingerprint
               prédit
                    │
                    ▼
          Candidate database
                    │
           filtrage par masse
                    │
                    ▼
        similarité fingerprint
                    │
                    ▼
               TOP 25
                    │
                    ▼
                MRR@25
```

---

# 39. Prochaines étapes

Les prochaines étapes logiques sont :

1. valider le parser sur tous les adducts uniques ;
2. ajouter `neutral_mass` au dataset ;
3. créer les Morgan fingerprints des molécules uniques ;
4. tester le binning sur quelques spectres ;
5. construire un premier `X_train`, `y_train`, `X_val`, `y_val` sur un sous-échantillon ;
6. entraîner un premier MLP ;
7. mesurer la qualité du fingerprint prédit ;
8. construire la candidate database locale ;
9. filtrer les candidats par masse ;
10. calculer le Top 25 ;
11. calculer le MRR@25 ;
12. faire tourner le pipeline sur un fold complet ;
13. passer ensuite à un DataLoader PyTorch pour exploiter tout le dataset ;
14. comparer progressivement plusieurs architectures ;
15. ajouter éventuellement COCONUT puis PubChem.

---

# 40. Objectif immédiat

L’objectif à court terme n’est pas encore d’obtenir le meilleur score Kaggle.

Il est d’obtenir un pipeline complet et reproductible :

```text
spectre MS/MS
→
prétraitement
→
modèle
→
représentation chimique
→
candidats
→
Top 25
→
MRR@25
```

Une fois cette première baseline obtenue, chaque amélioration pourra être évaluée objectivement avec la même validation croisée.