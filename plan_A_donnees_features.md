# Projet CASMI 2026 — Feuille de route du membre A

**Rôle : Données & features**

Ce document est autonome. La partie 1 (commune aux trois membres) explique le problème et toute la démarche depuis zéro. La partie 2 détaille ce que **toi** tu dois faire, chercher et livrer, sprint par sprint.

---

# PARTIE 1 — Démarche commune (identique dans les trois documents)

## 1.1 Le problème

La compétition Kaggle *Enveda CASMI 2026 – Molecule ID From Mass Spectra* demande d'identifier la **structure chimique 2D** de petites molécules détectées dans des extraits biologiques, à partir de leurs **spectres de masse en tandem (MS/MS)**.

- **Entrée :** pour chaque molécule inconnue, un ou plusieurs spectres. Un spectre, c'est une liste de pics `(m/z, intensité)` et des métadonnées : `precursor_mz`, `adduct`, mode d'ionisation, énergie de collision, type d'instrument.
- **Sortie attendue :** jusqu'à 25 structures candidates (au format SMILES), classées de la plus probable à la moins probable.
- **Score :** MRR@25. Pour chaque molécule, on gagne 1 si la bonne structure est en 1re position, 1/2 en 2e, 1/3 en 3e… et 0 si elle n'est pas dans le top 25. Le score final est la moyenne sur toutes les molécules.

Ordres de grandeur des données :

| Fichier | Contenu |
|---|---|
| `train.parquet` | ~2,5 millions de spectres, ~276 000 molécules distinctes, 11 bibliothèques sources, ~2,8 Go |
| `test.parquet` | ~1 200 spectres pour 400 molécules, sans la structure |
| `sample_submission.csv` | 400 lignes : `molecule_id`, `smiles` (25 SMILES séparés par `;`) |

Une molécule est identifiée par son **`inchikey14`** : les 14 premiers caractères de son InChIKey, qui codent sa connectivité 2D. Une même molécule possède en général plusieurs spectres, acquis dans des conditions différentes.

## 1.2 Pourquoi ce n'est pas une classification classique

L'idée naïve serait de traiter chaque `inchikey14` comme une classe et d'entraîner un classifieur à 276 000 classes. Cela ne marche pas, pour deux raisons :

1. **Le test contient des molécules absentes du train.** Un classifieur ne peut répondre que par une molécule qu'il a déjà vue.
2. **Notre validation doit imiter cette situation.** On y met des molécules dont aucun spectre n'a été vu à l'entraînement. Un classifieur y obtiendrait 0 par construction.

**L'idée clé : on ne prédit pas l'identité de la molécule, on prédit une description de sa structure.**

Cette description est un **fingerprint de Morgan** : un vecteur de bits (par exemple 2048). Chaque bit indique à peu près la présence d'un petit motif chimique, c'est-à-dire l'environnement d'un atome jusqu'à une certaine distance (un cycle aromatique, un groupe acide, un azote dans un cycle…). La bibliothèque RDKit le calcule de façon déterministe à partir du SMILES.

Une molécule nouvelle est presque toujours faite de motifs que le modèle a vus des milliers de fois dans d'autres molécules. Il peut donc généraliser. Le problème devient une **classification multi-label** : 2048 petites questions oui/non (« ce motif est-il présent ? ») au lieu d'une seule question à 276 000 réponses.

> **Analogie du portrait-robot.** Un témoin n'a jamais vu le suspect dans un fichier, mais il le décrit : lunettes, barbe, cicatrice. La police garde les personnes de la bonne taille dans son fichier, puis compare leurs photos au portrait. Ici, la taille, c'est la **masse** ; le portrait, c'est le **fingerprint prédit** ; le fichier, c'est un **catalogue de structures connues**.

## 1.3 Le pipeline complet

```
Spectres d'une molécule inconnue
   │
   ├─ (1) precursor_mz + adduct ─────────► masse neutre M de la molécule
   │                                          │
   │                       (2) catalogue de structures connues,
   │                           filtré à M ± tolérance (en ppm)
   │                                          │
   │                                          ▼
   │                           candidats (quelques dizaines à quelques milliers)
   │                           + leur fingerprint calculé par RDKit
   │                                          │
   ├─ (3) MODÈLE : spectre ──► fingerprint prédit
   │                                          │
   └──────────────── (4) similarité (Tanimoto) entre le fingerprint prédit
                         et celui de chaque candidat
                                              │
                                              ▼
                              (5) tri ► top 25 ► MRR@25
```

Trois remarques fondamentales :

- **Seule l'étape 3 est de l'apprentissage automatique.** Les étapes 1, 2, 4 et 5 sont déterministes. On construit donc d'abord tout le pipeline avec un « modèle » trivial, puis on remplace l'étape 3 par des modèles de plus en plus sérieux.
- **La masse seule ne départage pas les isomères.** Deux molécules de même formule brute ont exactement la même masse. C'est le spectre (les fragments) qui doit faire la différence.
- **Si la bonne molécule n'est pas dans le catalogue, aucune méthode de ce type ne peut la trouver.** On mesure donc deux choses séparément : le *rappel du filtre* (la vraie molécule est-elle parmi les candidats ?) et le *MRR* (est-elle bien classée ?).

## 1.4 La validation

- **Split par molécule.** 10 % des `inchikey14`, avec tous leurs spectres, forment la validation. Aucun spectre d'une molécule de validation n'est utilisé pour entraîner un modèle.
- **Catalogue de validation (« monde fermé »).** Il contient toutes les structures du train **et** de la validation, comme une base de données de structures connues. La bonne réponse y est, mais le modèle n'a jamais vu ses spectres.
- **Sous-ensemble `dev`.** Environ 2 000 molécules de validation pour itérer vite. La validation complète sert aux résultats officiels.
- **Test Kaggle.** Aucune liste de candidats n'est fournie avec les données. Il faudra un catalogue externe (COCONUT, PubChem…), si les règles de la compétition l'autorisent (à vérifier par C).
- **Fuites de données à éviter.** Les colonnes dérivées de la vraie structure ne doivent **jamais** servir d'entrée à un modèle : `normalized_smiles`, `inchikey`, `inchikey14`, `molecular_formula`, `precursor_error_ppm`. D'ailleurs, elles n'existent pas dans le test.

## 1.5 Progression des méthodes

| Niveau | Méthode | À quoi elle sert |
|---|---|---|
| Baseline 0 | Filtre par masse, puis tri par erreur de masse | **Plancher** : ce que la masse seule permet |
| Oracle | Classement avec le *vrai* fingerprint | **Plafond** : le mieux possible avec cette représentation |
| Baseline 1 | Plus proches voisins spectraux, sans entraînement | Premier usage du spectre |
| Modèle 1 | MLP : spectre → fingerprint | Premier modèle appris |
| Améliorations | Features, architecture, fusion de scores, catalogue externe | Selon les résultats |

Toute méthode est évaluée sur le même split, avec les mêmes métriques, et son résultat est ajouté à `reports/results.csv`.

## 1.6 L'abstraction centrale : le « scorer »

Pour que les trois membres puissent travailler en parallèle, **toutes les méthodes prennent la même forme** : une fonction qui reçoit les spectres d'une molécule inconnue et la liste de ses candidats, et qui renvoie un score par candidat.

```python
def scorer(query_spectra: pd.DataFrame, cand_idx: np.ndarray) -> np.ndarray:
    """Renvoie un score par candidat. Plus grand = plus probable."""
```

- Baseline 0 : score = −|erreur de masse en ppm|.
- Modèles à fingerprint : score = Tanimoto(fingerprint prédit, fingerprint du candidat).

La fonction `evaluate(scorer)` de C s'occupe de tout le reste : récupérer les candidats, trier, garder le top 25, calculer les métriques.

## 1.7 Organisation de l'équipe

### Rôles

| Membre | Rôle | Fichiers dont il est propriétaire |
|---|---|---|
| **A** | Données & features | `config.py`, `io.py`, `split.py`, `features.py`, `dataset.py` |
| **B** | Chimie, catalogue & modèle | `mass.py`, `catalog.py`, `fingerprints.py`, `models/mlp.py` |
| **C** | Évaluation, classement & soumission | `metrics.py`, `evaluate.py`, `ranking.py`, `submission.py`, `models/knn.py` |

### Les sprints

| Sprint | Objectif | A | B | C |
|---|---|---|---|---|
| **0. Mise en place** | Environnement, données, bases théoriques | Dépôt, environnement, chargement des données | Fiche chimie, prise en main de RDKit | Fiche Kaggle, conventions d'équipe, tests |
| **1. Socle sans ML** | Une première évaluation de bout en bout | Dictionnaire des données, NA, train vs test, nettoyage, split figé | Masse neutre, catalogue, filtre par masse | Métriques, évaluation, baseline 0, première soumission |
| **2. Représentations** | Préparer l'entrée et la sortie du modèle | Binning des spectres, features de métadonnées | Fingerprints du catalogue, analyse des cibles | Classement par Tanimoto, agrégation, oracle |
| **3. Premiers modèles** | Battre la baseline 0 | Chargement des données à grande échelle | MLP spectre → fingerprint | Plus proches voisins spectraux, comparaison |
| **4. Améliorations** | Une expérience par personne | Ablations de features et stratégies NA | Amélioration du modèle | Fusion de scores, catalogue externe, soumission finale |
| **5. Restitution** | Rapport et conférence | Partie données | Partie chimie et modèle | Partie résultats |

La durée de chaque sprint est à fixer ensemble selon votre calendrier.

### Le cycle d'un sprint

1. **Réunion de lancement (30 min)** : relire les tâches, confirmer les interfaces (noms de fonctions, colonnes, fichiers).
2. **Travail individuel** sur une branche. Si tu dépends du travail d'un autre qui n'est pas prêt, écris une **version provisoire** (un *stub*) qui respecte l'interface, et remplace-la au merge.
3. **Réunion de revue (1 h)** : chacun présente son résultat en 5 minutes, on discute, on relit et on merge. Les décisions sont notées dans `docs/decisions.md`.

### Règles git

- `main` doit toujours fonctionner. Une branche par tâche, nommée `s<sprint>/<membre>-<sujet>` : `s1/A-split`, `s2/C-oracle`…
- Merge uniquement par *pull request*, relue par un autre membre.
- Le code réutilisable va dans `src/casmi/*.py`. Les notebooks servent seulement à explorer, et chacun a son dossier `notebooks/A`, `notebooks/B`, `notebooks/C`. Installez `nbstripout` pour retirer les sorties des notebooks avant chaque commit.
- Jamais de données dans git : `data/` est dans `.gitignore`.
- Pendant un sprint, chacun ne modifie que ses propres fichiers. Pour changer le fichier d'un autre, on le lui demande.

### Arborescence du dépôt

```
casmi/
├── data/                    # hors git
│   ├── raw/                 # train.parquet, test.parquet, sample_submission.csv
│   └── processed/           # fichiers produits par l'équipe
├── src/casmi/
│   ├── config.py            # chemins et constantes (A)
│   ├── io.py                # chargement des données (A)
│   ├── split.py             # split train / val / dev (A)
│   ├── features.py          # binning et features de métadonnées (A)
│   ├── dataset.py           # chargement par batches pour PyTorch (A)
│   ├── mass.py              # adducts et masse neutre (B)
│   ├── catalog.py           # catalogue et filtre par masse (B)
│   ├── fingerprints.py      # fingerprints (B)
│   ├── metrics.py           # MRR, rappel (C)
│   ├── ranking.py           # similarité, agrégation (C)
│   ├── evaluate.py          # évaluation de bout en bout (C)
│   ├── submission.py        # fichier de soumission Kaggle (C)
│   └── models/
│       ├── mlp.py           # (B)
│       └── knn.py           # (C)
├── notebooks/A/  notebooks/B/  notebooks/C/
├── tests/                   # tests pytest
├── reports/                 # results.csv, figures
├── docs/                    # fiches, dictionnaire des données, décisions
├── journal_ia/              # un journal par membre
├── requirements.txt
└── README.md
```

### Contrat d'interfaces

Ce sont les points de rencontre entre vos travaux. On les modifie seulement en réunion.

**Fichiers produits**

| Fichier | Auteur | Sprint | Contenu |
|---|---|---|---|
| `data/processed/meta.parquet` | A | 1 | Une ligne par spectre du train, sans les pics : `row_id` (position dans `train.parquet`), `inchikey14`, `normalized_smiles`, `ingest_lib`, `ionization_mode`, `instrument_type`, `adduct`, `precursor_mz`, `collision_energy_ev`, `num_peaks`, `keep` (booléen, filtre qualité) |
| `data/processed/split.parquet` | A | 1 | `inchikey14`, `split` (`"train"` ou `"val"`), `dev` (booléen) |
| `data/processed/catalog.parquet` | B | 1 | `cat_idx`, `inchikey14`, `smiles`, `formula`, `exact_mass` |
| `data/processed/catalog_fp.npy` | B | 2 | Matrice `(n_catalogue, 2048)` ; la ligne `i` correspond à `cat_idx = i` |
| `reports/results.csv` | C | 1 | Une ligne par expérience : date, auteur, méthode, paramètres, split, métriques |

**Fonctions**

```python
# A — io.py, split.py (sprints 0–1)
load_meta(columns=None) -> pd.DataFrame            # une ligne par spectre, avec row_id
load_spectra(row_ids) -> pd.DataFrame              # row_id, ms2_mzs, ms2_normalized_intensities
load_split() -> pd.DataFrame                       # inchikey14, split, dev
load_test() -> pd.DataFrame                        # le test complet

# A — features.py (sprint 2), dataset.py (sprint 3)
bin_spectrum(mzs, intensities, bin_width, max_mz) -> np.ndarray   # float32
meta_features(df) -> np.ndarray
make_loader(split, batch_size) -> torch DataLoader

# B — mass.py, catalog.py (sprint 1)
parse_adduct(adduct) -> (n_mol, delta, z)
compute_neutral_masses(df) -> np.ndarray           # NaN si adduct non interprétable
load_catalog() -> pd.DataFrame
get_candidates(neutral_mass, ppm=10) -> np.ndarray  # tableau de cat_idx

# B — fingerprints.py (sprint 2), models/mlp.py (sprint 3)
smiles_to_fp(smiles) -> np.ndarray                 # uint8, taille 2048
load_catalog_fps() -> np.ndarray                   # (n_catalogue, 2048)
predict_fp(query_spectra) -> np.ndarray            # (n_spectres, 2048), probabilités

# C — metrics.py, evaluate.py, ranking.py (sprints 1–2)
mrr_at_k(true_keys, ranked_keys, k=25) -> float
recall_at_k(true_keys, ranked_keys, k) -> float
build_queries(split="dev") -> pd.DataFrame         # spectres + query_id + true_inchikey14
evaluate(scorer, split="dev", ppm=10) -> dict
log_result(name, params, metrics)
tanimoto_scores(pred_fp, cand_fps) -> np.ndarray
aggregate_fps(fps) -> np.ndarray                   # plusieurs spectres -> un fingerprint
```

## 1.8 Le journal IA (pour la conférence)

La conférence finale porte sur **l'utilisation de l'IA dans le projet**. Sa matière première, c'est ce que vous notez au fil de l'eau. Chacun tient `journal_ia/journal_<membre>.md` et ajoute une entrée à chaque usage significatif d'un assistant IA :

```markdown
## 2026-xx-xx — <tâche>
- Outil : ChatGPT / Claude / Copilot…
- Ce que j'ai demandé : (résumé ou prompt exact)
- Ce que j'ai gardé :
- Ce que j'ai vérifié, et comment :
- Erreur ou approximation détectée : (oui/non, laquelle)
- Temps gagné ou perdu (estimation) :
- Verdict : utile / trompeur / à vérifier systématiquement
```

Les erreurs détectées sont les entrées les plus précieuses : une affirmation fausse, un code qui accepte silencieusement des cas invalides, une conclusion tirée d'un échantillon non représentatif. Ce sont elles qui rendront la conférence intéressante.

## 1.9 Glossaire

| Terme | Définition |
|---|---|
| **Spectre MS/MS** | Liste des fragments obtenus en cassant un ion sélectionné : chaque pic = un rapport masse/charge (m/z) et une intensité |
| **m/z** | Rapport masse/charge d'un ion, en daltons (Da) pour une charge 1 |
| **Ion précurseur** | L'ion sélectionné avant fragmentation ; `precursor_mz` est son m/z |
| **Adduct** | Forme ionisée de la molécule : `[M+H]+` = molécule + un proton ; `[2M+Na]+` = deux molécules + un ion sodium |
| **Masse neutre** | Masse monoisotopique de la molécule M elle-même, retrouvée à partir du précurseur et de l'adduct |
| **ppm** | Partie par million : erreur relative de masse. 10 ppm sur 300 Da = 0,003 Da |
| **Isomères** | Molécules de même formule brute (donc même masse) mais de structures différentes |
| **SMILES** | Écriture textuelle d'une structure : `CC(=O)O` = acide acétique |
| **InChIKey / inchikey14** | Identifiant standardisé d'une structure ; les 14 premiers caractères codent la connectivité 2D |
| **Fingerprint de Morgan** | Vecteur de bits décrivant les motifs chimiques présents |
| **Tanimoto** | Similarité entre deux fingerprints : bits communs / bits présents dans l'un ou l'autre (entre 0 et 1) |
| **MRR@25** | Moyenne de 1/rang de la bonne réponse, 0 si elle est hors du top 25 |
| **Rappel@k** | Proportion de molécules dont la bonne réponse est dans les k premières |
| **Fuite de données** | Information de la réponse qui s'infiltre dans les entrées et gonfle artificiellement le score |

---

# PARTIE 2 — Tes tâches : membre A, données & features

## 2.0 Ta mission en résumé

Tu es le garant de **la qualité et de la compréhension des données**, et de **ce que le modèle reçoit en entrée**. Concrètement :

- tout le monde charge les données grâce à toi (sprint 0) ;
- tu documentes chaque colonne et tu décides de ce qu'on garde (sprint 1) ;
- tu fabriques le split que toute l'équipe utilise (sprint 1) ;
- tu transformes un spectre (liste de longueur variable) en vecteur de taille fixe (sprint 2) ;
- tu permets d'entraîner sur des millions de spectres sans saturer la mémoire (sprint 3) ;
- tu mesures l'effet de tes choix de préparation sur le score (sprint 4).

Tes livrables bloquent les autres au sprint 1 (`meta.parquet`, `split.parquet`) : livre-les en priorité, quitte à les améliorer ensuite.

Chaque tâche suit le même format : **Objectif**, **Pourquoi**, **À chercher / comprendre**, **Étapes proposées**, **Livrable**, **Pièges**, **À présenter en réunion**.

---

## Sprint 0 — Mise en place

### Tâche A0.1 — Créer le dépôt et l'environnement

**Objectif.** Un dépôt GitHub que les trois membres clonent, et un environnement Python identique pour tous.

**Pourquoi.** Si chacun a ses versions de bibliothèques et ses chemins, rien ne fonctionnera au premier merge.

**À chercher / comprendre.**
- Créer un environnement virtuel (`python -m venv`) ou un environnement conda. RDKit s'installe aujourd'hui avec `pip install rdkit`.
- Rédiger un `requirements.txt` : `pandas`, `pyarrow`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `rdkit`, `torch`, `kagglehub`, `pytest`, `nbstripout`, `ipykernel`.
- Ce que fait `nbstripout --install` et pourquoi les sorties de notebook posent problème dans git.
- Rendre `src/casmi` importable : un `pyproject.toml` minimal puis `pip install -e .`, ou à défaut un ajout au `sys.path` en haut des notebooks.

**Étapes proposées.**
1. Créer le dépôt, l'arborescence de la partie 1.7 et un `.gitignore` (`data/`, `*.parquet`, `*.npy`, `__pycache__/`, `.venv/`).
2. Écrire le `README.md` : installation en 5 commandes maximum, téléchargement des données, lancement des tests.
3. Faire tester l'installation par B et C sur leurs machines.

**Livrable.** Dépôt cloné et installé chez les trois membres ; `README.md` ; `requirements.txt`.

**Pièges.** Committer les données par erreur (2,8 Go). Des chemins absolus propres à ta machine dans le code : tout chemin doit passer par `config.py`.

### Tâche A0.2 — Télécharger et charger les données

**Objectif.** `config.py` et `io.py` qui permettent à chacun de charger exactement ce dont il a besoin.

**Pourquoi.** `train.parquet` fait 2,8 Go, et les listes de pics en occupent l'essentiel. Le charger entièrement en pandas peut saturer la RAM d'un ordinateur portable. Il faut savoir lire seulement certaines colonnes, ou seulement certaines lignes.

**À chercher / comprendre.**
- Télécharger avec `kagglehub.competition_download(...)` (il faut avoir accepté les règles de la compétition sur Kaggle) vers `data/raw/`.
- Le format parquet : stockage par colonnes et par *row groups*. Dans `pyarrow.parquet` : `ParquetFile`, `.metadata` (nombre de lignes, de row groups, statistiques `null_count` par colonne), `read_row_group`, `iter_batches`, et le paramètre `columns=` de `pd.read_parquet`.
- Comment lire les pics d'une liste de lignes précises sans tout charger. Une piste : savoir dans quel row group se trouve chaque `row_id`, puis lire seulement ces row groups.

**Étapes proposées.**
1. `config.py` : `DATA_RAW`, `DATA_PROCESSED`, `TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_PATH`, `SEED = 42`.
2. `io.py` :
   - `load_meta(columns=None)` : toutes les colonnes sauf les pics par défaut, plus une colonne `row_id` égale à la position de la ligne dans le fichier ;
   - `load_spectra(row_ids)` : renvoie `row_id`, `ms2_mzs`, `ms2_normalized_intensities` pour ces lignes ;
   - `load_test()`.
3. Mesurer le temps et la mémoire de `load_meta()` (`df.memory_usage(deep=True)`).

**Livrable.** `src/casmi/config.py`, `src/casmi/io.py`, et un test qui vérifie que `load_spectra([0, 5])` renvoie bien les lignes 0 et 5.

**Pièges.** `row_id` doit être stable : c'est la clé qui relie toutes les tables de l'équipe. Ne la recalcule pas après un filtrage.

**À présenter en réunion.** Le temps de chargement de `load_meta()` et sa taille en mémoire. Le temps pour lire 1 000 spectres pris au hasard.

---

## Sprint 1 — Comprendre et préparer les données

### Tâche A1.1 — Le dictionnaire des données

**Objectif.** Un tableau qui décrit chaque colonne du train (18) et du test (12).

**Pourquoi.** On ne peut pas choisir des features sans savoir ce que veut dire chaque colonne, ni si elle existe au moment de prédire.

**À chercher / comprendre.** Pour chaque colonne :
- signification (demande à B pour les notions de chimie) ;
- type (texte, nombre, liste) et un exemple de valeur ;
- pourcentage de valeurs manquantes dans le train et dans le test ;
- présente dans le test ? (oui / non) ;
- utilisable comme feature ? (oui / non / à risque de fuite), avec la raison.

Les cas à regarder de près : la différence entre `adduct` et `adduct_orig` ; entre `collision_energy_ev` (liste en eV) et `collision_energy_orig` (texte brut) ; `base_peak_intensity` (intensité absolue du pic le plus haut avant normalisation) ; `precursor_error_ppm`, qui est calculée à partir de la vraie structure et constitue donc une fuite.

**Étapes proposées.**
1. Lister les colonnes et les types depuis le schéma parquet.
2. Compter les NA **sur tout le fichier** à partir des métadonnées parquet (c'est instantané).
3. Remplir le tableau dans `docs/dictionnaire_donnees.md`.

**Livrable.** `docs/dictionnaire_donnees.md`.

### Tâche A1.2 — Les valeurs manquantes

**Objectif.** Comprendre d'où viennent les NA et proposer, pour chaque colonne concernée, des stratégies à tester plus tard.

**Pourquoi.** Supprimer toutes les lignes avec un NA ferait perdre une grande partie des données. Imputer sans comprendre peut injecter du bruit. Et l'absence d'une valeur peut elle-même être une information, par exemple sur la bibliothèque d'origine.

**À chercher / comprendre.**
- Les colonnes concernées : `instrument_type`, `base_peak_intensity`, `collision_energy_ev`, `collision_energy_orig`, `precursor_error_ppm`.
- **Les NA sont-ils simultanés ?** Si plusieurs colonnes manquaient toujours ensemble, elles auraient exactement le même nombre de NA. Compare les comptes.
- Le lien avec `ingest_lib` : quelles bibliothèques ne renseignent pas quoi ?
- Pourquoi `collision_energy_ev` peut manquer alors que `collision_energy_orig` est rempli (unités non converties ?).
- Les NA dans le **test** : une colonne toujours remplie dans le test mais souvent vide dans le train mérite une attention particulière.

**Étapes proposées.**
1. Charger seulement ces colonnes et `ingest_lib`, sur tout le train (c'est léger).
2. Faire la table des combinaisons de NA (`df[cols].isna().value_counts()`).
3. Croiser avec `ingest_lib` (`pd.crosstab`).
4. Pour chaque colonne, écrire 2 ou 3 stratégies candidates : supprimer la feature ; imputer et ajouter une variable indicatrice « était manquant » ; imputer par bibliothèque… Ces stratégies seront comparées au sprint 4.

**Livrable.** `notebooks/A/02_valeurs_manquantes.ipynb` et une section « Valeurs manquantes » dans le dictionnaire.

**Pièges.** **Les lignes du fichier sont regroupées par bibliothèque.** Un échantillon tiré d'un seul row group ne contient que deux ou trois bibliothèques et donne des conclusions fausses sur l'ensemble. Travaille sur tout le fichier, ou tire un échantillon aléatoire réparti sur tous les row groups. Vérifie toute conclusion en comparant aux comptes exacts.

**À présenter en réunion.** Le schéma des NA par bibliothèque, et les stratégies proposées.

### Tâche A1.3 — Comparer le train et le test

**Objectif.** Savoir à quoi ressemblent les spectres du test par rapport à ceux du train.

**Pourquoi.** Si le test ne ressemble qu'à une partie du train (un instrument, quelques adducts, une plage de masses), notre validation devrait surtout mesurer la performance sur cette partie. Sinon, on optimise pour autre chose que le vrai objectif.

**À chercher / comprendre.** Comparer les distributions de :
- `ionization_mode`, `adduct`, `instrument_type` ;
- `collision_energy_ev` (une ou plusieurs valeurs par spectre ?) ;
- `precursor_mz`, `num_peaks`, `base_peak_intensity` ;
- nombre de spectres par molécule (train : de 1 à plus de 1 000 ; test : environ 3) ;
- la bibliothèque du train qui ressemble le plus au test.

**Étapes proposées.**
1. Histogrammes superposés (train en proportion, test en proportion).
2. Un tableau de synthèse « ce qui est pareil, ce qui diffère ».
3. Des recommandations pour la validation. Exemples : limiter le nombre de spectres par molécule de validation pour imiter le test ; ne garder dans les requêtes de validation que les adducts présents dans le test.

**Livrable.** `notebooks/A/03_train_vs_test.ipynb` et une note de recommandations dans `docs/decisions.md`.

**À présenter en réunion.** Trois graphiques maximum, et tes recommandations pour la validation. C en a besoin pour construire les requêtes.

### Tâche A1.4 — Le nettoyage

**Objectif.** Une colonne `keep` qui marque les spectres utilisables pour l'entraînement.

**Pourquoi.** Des spectres avec des étiquettes douteuses (mauvais adduct, structure incohérente avec la masse) apprennent au modèle des choses fausses.

**À chercher / comprendre.**
- Les adducts non interprétables (B te donnera la liste : ceux pour lesquels `compute_neutral_masses` renvoie NaN).
- Les spectres avec très peu de pics.
- Les spectres dont la masse du précurseur est incohérente avec la structure annoncée. `precursor_error_ppm` peut servir **à nettoyer le train**, car il s'agit de la qualité de l'étiquette. Il ne doit jamais servir **de feature**.
- La différence entre nettoyer les données d'entraînement (autorisé, avec des informations de structure) et filtrer les requêtes de validation (seulement avec des informations disponibles dans le test, sinon la validation devient plus facile que le test).

**Étapes proposées.**
1. Écrire chaque règle de nettoyage séparément et compter combien de spectres elle retire.
2. Vérifier qu'aucune règle ne supprime toute une bibliothèque par accident.
3. Produire `meta.parquet` avec la colonne `keep`.

**Livrable.** `data/processed/meta.parquet` ; la liste des règles et leurs effets dans `docs/decisions.md`.

**Pièges.** Ne pas recalculer `row_id` après filtrage : `keep` est une colonne, pas une suppression.

### Tâche A1.5 — Le split figé

**Objectif.** `split.parquet`, utilisé par toute l'équipe pendant tout le projet.

**Pourquoi.** Si chacun tire son propre split, les scores ne sont pas comparables. Et si un spectre d'une molécule de validation se retrouve dans l'entraînement, le score est faussement optimiste.

**À chercher / comprendre.**
- Pourquoi on sépare **par molécule** (`inchikey14`) et non par spectre.
- Un tirage reproductible : `np.random.default_rng(SEED)`.
- Pour aller plus loin : le split par *scaffold* (squelette de Bemis-Murcko, disponible dans RDKit). Les molécules qui partagent le même squelette vont du même côté. La validation devient plus dure, car le train ne contient plus de « quasi-jumeaux » des molécules de validation. À proposer en option, à discuter avec B (qui mesure ces similarités au sprint 2).

**Étapes proposées.**
1. Tirer 10 % des `inchikey14` pour `split = "val"`, le reste en `"train"`.
2. Parmi les molécules de validation, tirer environ 2 000 molécules pour `dev = True`.
3. Vérifications automatiques (dans un test pytest) : aucune molécule des deux côtés ; tailles attendues ; fichier identique si on relance le script.
4. Comparer les distributions (bibliothèques, adducts) entre train et validation.

**Livrable.** `src/casmi/split.py`, `data/processed/split.parquet`, `load_split()`.

**À présenter en réunion.** Nombre de molécules et de spectres dans chaque partie ; les distributions comparées.

---

## Sprint 2 — Transformer un spectre en vecteur

### Tâche A2.1 — Le binning des spectres

**Objectif.** `bin_spectrum(mzs, intensities, bin_width, max_mz)` qui transforme une liste de pics de longueur variable en vecteur de taille fixe.

**Pourquoi.** Un MLP ou un algorithme de plus proches voisins a besoin d'entrées de même taille. Le binning découpe l'axe m/z en cases régulières. Chaque case reçoit l'intensité des pics qui y tombent.

Exemple avec des cases de 10 Da : les pics (12,1 ; 0,2), (18,7 ; 1,0), (24,2 ; 0,4) donnent le vecteur `[0 ; 1,0 ; 0,4 ; …]` (case 0–10, case 10–20, case 20–30…). Si deux pics tombent dans la même case, on garde le maximum ou la somme.

**À chercher / comprendre.**
- **Largeur des cases.** Fine (0,01 Da) : précis, mais vecteur énorme et très creux. Large (1 Da) : compact, mais des fragments différents se mélangent. Regarde la précision des instruments du jeu de données.
- **`max_mz`.** Regarde la distribution des m/z des pics et de `precursor_mz`. Les adducts `[2M+…]` ont des précurseurs élevés.
- **Agrégation** dans une case : maximum ou somme ?
- **Transformation des intensités** : racine carrée ou logarithme, pour que les petits pics comptent aussi.
- **Normalisation** du vecteur (norme L2), utile pour la similarité cosinus de C.
- **Les pertes neutres** (`precursor_mz − m/z du fragment`) : en chimie, ce qui est *perdu* est souvent aussi informatif que le fragment. Option : un deuxième vecteur binné pour les pertes.
- Outils : `np.maximum.at`, `np.add.at`, `np.digitize`, et les matrices creuses `scipy.sparse.csr_matrix`.

**Étapes proposées.**
1. Écrire la fonction, avec des tests : pic exactement sur une frontière de case, pic au-delà de `max_mz`, spectre vide, deux pics dans la même case.
2. Visualiser un spectre brut et son vecteur binné, l'un au-dessus de l'autre.
3. Mesurer le nombre moyen de cases non nulles par spectre selon la largeur choisie.

**Livrable.** `src/casmi/features.py` (`bin_spectrum`) et ses tests.

**Pièges.** Des m/z négatifs ou NaN. Des indices de case qui dépassent la taille du vecteur.

### Tâche A2.2 — Les features de métadonnées

**Objectif.** `meta_features(df)` qui encode les métadonnées utiles d'un spectre.

**Pourquoi.** Un même fragment ne veut pas dire la même chose en mode positif et en mode négatif, ou selon l'énergie de collision. Et la masse neutre renseigne sur la taille de la molécule.

**À chercher / comprendre.**
- `ionization_mode` : 0 ou 1.
- `adduct` : *one-hot* sur les adducts les plus fréquents, plus une catégorie « autre ».
- Masse neutre (fonction de B, `compute_neutral_masses`), mise à l'échelle (par exemple divisée par 1 000).
- `collision_energy_ev` : c'est une liste, donc moyenne, minimum ou maximum, plus un indicateur de NA.
- `instrument_type` : *one-hot*, avec une catégorie « inconnu ».
- `num_peaks` : en logarithme.
- **Interdit** : toute colonne listée comme fuite en partie 1.4.

**Étapes proposées.**
1. Les encodages (catégories retenues, moyennes, échelles) sont **appris sur le train uniquement**, puis appliqués à la validation et au test.
2. Renvoyer aussi la liste des noms de features, pour pouvoir les interpréter.

**Livrable.** `meta_features(df)` dans `features.py`, et ses tests.

**Pièges.** Un adduct du test absent des catégories apprises doit tomber dans « autre » sans erreur.

### Tâche A2.3 — Les matrices pour les premiers modèles

**Objectif.** Des matrices X prêtes à l'emploi pour B et C : une pour le sous-ensemble `dev`, une pour un échantillon du train (par exemple 200 000 spectres).

**Pourquoi.** B et C doivent commencer à modéliser sans attendre le chargement à grande échelle du sprint 3.

**À chercher / comprendre.** La mémoire : 2,5 millions de spectres × 2 400 cases × 4 octets ≈ 24 Go en dense, impossible. Mais un spectre n'a que quelques dizaines à quelques centaines de pics, donc une matrice **creuse** est beaucoup plus légère. Calcule l'ordre de grandeur.

**Étapes proposées.**
1. Construire X (creuse) pour `dev` et pour l'échantillon train, avec un tableau `row_id` aligné.
2. Sauvegarder (`scipy.sparse.save_npz`).

**Livrable.** `data/processed/X_dev.npz`, `X_train_sample.npz` et les `row_id` correspondants.

**À présenter en réunion.** Tes choix de binning et leur justification ; la taille des matrices.

---

## Sprint 3 — Passer à l'échelle

### Tâche A3.1 — Le chargement par batches

**Objectif.** `make_loader(split, batch_size)` qui fournit à B des batches `(x, y)` sur tout le train, où `y` est le fingerprint de la molécule.

**Pourquoi.** Le MLP doit voir des millions de spectres, ce qui ne tient pas d'un bloc en mémoire.

**À chercher / comprendre.**
- `torch.utils.data.Dataset` et `IterableDataset`, `DataLoader`, `num_workers`.
- Deux stratégies à comparer :
  1. **précalculer** toute la matrice creuse du train une fois (quelques Go ?), puis découper en batches ;
  2. **lire et binner à la volée** par row group.
- **Le mélange (shuffle).** Les row groups sont triés par bibliothèque : sans mélange, chaque époque voit les bibliothèques l'une après l'autre, ce qui perturbe l'apprentissage. Mélange l'ordre des row groups et utilise un tampon de mélange.
- `y` : retrouver le fingerprint via `inchikey14` → `cat_idx` → `load_catalog_fps()` (fonctions de B).

**Étapes proposées.**
1. Mesurer la vitesse (spectres par seconde) des deux stratégies.
2. Vérifier qu'aucun spectre de validation n'est servi en mode `split="train"`, et que `keep` est respecté.

**Livrable.** `src/casmi/dataset.py`.

### Tâche A3.2 — L'échantillonnage par molécule

**Objectif.** Éviter que les molécules qui ont des centaines de spectres dominent l'entraînement.

**Pourquoi.** Si une molécule a 1 000 spectres et une autre 3, le modèle apprend surtout la première.

**À chercher / comprendre.** Plafonner le nombre de spectres par molécule et par époque ; ou pondérer chaque spectre par 1/(nombre de spectres de sa molécule).

**Livrable.** Une option `max_spectra_per_molecule` dans `make_loader`, à comparer avec B.

---

## Sprint 4 — Expériences sur les features

### Tâche A4.1 — Les ablations

**Objectif.** Mesurer l'effet de chaque choix de préparation sur le MRR.

**Pourquoi.** C'est la démarche scientifique du projet : une modification à la fois, même split, même métrique. Le tableau qui en résulte est une pièce maîtresse du rapport.

**Expériences possibles** (une modification à la fois par rapport à la configuration de référence) :
- largeur des cases (par exemple 0,1 / 0,5 / 1 Da) ;
- transformation des intensités (aucune / racine / log) ;
- avec ou sans features de métadonnées ;
- avec ou sans vecteur des pertes neutres ;
- les stratégies de NA définies au sprint 1 ;
- plafond de spectres par molécule ;
- split aléatoire contre split par scaffold (si retenu).

**Étapes proposées.** Avec le modèle le plus rapide à réentraîner (plus proches voisins de C, ou petit MLP de B), sur le sous-ensemble `dev`. Chaque résultat va dans `reports/results.csv` via `log_result`.

**Livrable.** Le tableau des ablations et un graphique.

**Pièges.** Choisir les paramètres sur `dev` puis rapporter le score final sur toute la validation, pas l'inverse.

---

## Sprint 5 — Restitution

- Rédiger la partie « Données » du rapport : description, valeurs manquantes, nettoyage, split, différences train/test, ablations.
- Préparer 2 ou 3 diapositives : les pièges des données et ce que l'IA vous a aidé (ou non) à voir.
- Relire ton journal IA et choisir les 2 meilleures anecdotes.

## Idées d'entrées pour ton journal IA

- Une conclusion sur les valeurs manquantes proposée par un assistant : était-elle vraie sur tout le dataset, ou seulement sur un échantillon ?
- Un code de chargement généré : tenait-il en mémoire ? Était-il correct sur les cas limites ?
- Les choix de binning suggérés : sur quoi reposaient-ils, et les as-tu vérifiés ?

## Récapitulatif de tes livrables

| Sprint | Livrable | Qui l'attend |
|---|---|---|
| 0 | Dépôt, `README.md`, `requirements.txt`, `config.py`, `io.py` | Tout le monde |
| 1 | `dictionnaire_donnees.md`, analyse NA, comparaison train/test | Tout le monde |
| 1 | `meta.parquet`, `split.parquet`, `load_split()` | B et C |
| 2 | `bin_spectrum`, `meta_features`, `X_dev.npz`, `X_train_sample.npz` | B et C |
| 3 | `make_loader` | B |
| 4 | Tableau des ablations | Rapport |
