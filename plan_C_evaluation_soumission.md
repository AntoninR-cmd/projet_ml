# Projet CASMI 2026 — Feuille de route du membre C

**Rôle : Évaluation, classement & soumission**

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

# PARTIE 2 — Tes tâches : membre C, évaluation, classement & soumission

## 2.0 Ta mission en résumé

Tu es **le garant de la mesure**. Sans toi, personne ne sait si une idée améliore quoi que ce soit. Concrètement :

- tu connais les règles exactes de la compétition et tu poses les conventions de l'équipe (sprint 0) ;
- tu écris les métriques et la fonction `evaluate` que tout le monde utilise, et tu produis le premier score (sprint 1) ;
- tu écris le classement par similarité, et tu calcules le plafond théorique avec l'oracle (sprint 2) ;
- tu construis la baseline par plus proches voisins et tu compares toutes les méthodes (sprint 3) ;
- tu combines les scores, tu prépares le catalogue externe et tu soumets sur Kaggle (sprint 4) ;
- tu assembles les résultats pour le rapport (sprint 5).

Ta fonction `evaluate` du sprint 1 est utilisée par A et B ensuite : elle doit être juste, testée et simple d'emploi.

Chaque tâche suit le même format : **Objectif**, **Pourquoi**, **À chercher / comprendre**, **Étapes proposées**, **Livrable**, **Pièges**, **À présenter en réunion**.

---

## Sprint 0 — Les règles du jeu

### Tâche C0.1 — La fiche Kaggle

**Objectif.** `docs/fiche_kaggle.md` : tout ce que la compétition impose, en une page.

**Pourquoi.** Une métrique mal comprise peut faire optimiser la mauvaise chose pendant des semaines. Et utiliser une donnée externe interdite rendrait la soumission invalide.

**À chercher / comprendre.** Lis les onglets *Overview*, *Evaluation*, *Data*, *Rules*, et parcours *Discussion*. Réponds précisément à :
1. La définition exacte de la métrique. Comment un SMILES prédit est-il comparé à la vraie structure : par `inchikey14`, par SMILES canonique, autrement ? Que se passe-t-il avec un SMILES invalide ? avec des doublons dans la liste ?
2. Le format exact de la soumission (colonnes, séparateur, nombre maximal de propositions).
3. **Les données externes** (PubChem, COCONUT…) et les modèles pré-entraînés sont-ils autorisés ? Sous quelles conditions ?
4. Les dates, le nombre de soumissions par jour, la taille d'équipe, la séparation du classement public et privé.
5. Ce qui est dit de l'origine des molécules du test.

**Livrable.** `docs/fiche_kaggle.md`, présentée en 5 minutes à la réunion de lancement du sprint 1.

### Tâche C0.2 — Les conventions de l'équipe

**Objectif.** Que le travail à trois soit fluide dès le premier merge.

**Étapes proposées.**
1. `CONTRIBUTING.md` : le workflow git de la partie 1.7, écrit pas à pas (créer une branche, committer, ouvrir une pull request, relire, merger).
2. `.github/pull_request_template.md` : ce que fait la PR, comment elle a été testée, résultats obtenus.
3. `pytest` : le dossier `tests/` avec un premier test d'exemple, et la commande pour lancer les tests dans le `README.md` (coordonne-toi avec A).
4. Le format de `reports/results.csv` (colonnes : `date`, `auteur`, `methode`, `parametres`, `split`, `ppm`, `mrr@25`, `recall@1`, `recall@5`, `recall@10`, `recall@25`, `rappel_filtre`, `n_candidats_median`, `commentaire`) et une fonction `log_result` qui y ajoute une ligne.
5. Les modèles de fichiers : `journal_ia/_modele.md` et `docs/decisions.md`.

**Livrable.** Ces fichiers, mergés dans `main`.

---

## Sprint 1 — Mesurer

### Tâche C1.1 — Les métriques

**Objectif.** `metrics.py` : rang réciproque, `mrr_at_k`, `recall_at_k`.

**Pourquoi.** C'est le thermomètre du projet. Une erreur ici fausse toutes les conclusions.

**À chercher / comprendre.**
- Rang réciproque d'une requête : 1/rang de la bonne réponse dans la liste, 0 si elle est absente des k premiers.
- MRR@k : la moyenne sur toutes les requêtes.
- Rappel@k : la proportion de requêtes dont la bonne réponse est dans les k premiers.
- Les cas limites, à aligner sur la définition Kaggle (C0.1) : liste vide, liste de moins de k éléments, doublons dans la liste.

**Étapes proposées.** Écrire les fonctions, puis des tests avec des cas calculés à la main. Par exemple : bonne réponse en position 1, 2 et absente sur trois requêtes → MRR = (1 + 0,5 + 0) / 3 = 0,5.

**Livrable.** `src/casmi/metrics.py` et `tests/test_metrics.py`.

### Tâche C1.2 — Les requêtes de validation

**Objectif.** `build_queries(split)` qui transforme la validation en « molécules inconnues », comme celles du test.

**Pourquoi.** Dans le test, une requête est une molécule (`molecule_id`) avec environ 3 spectres. La validation doit lui ressembler, sinon le score mesuré ne prédit pas le score Kaggle.

**À chercher / comprendre.**
- Une requête = une molécule de validation (`inchikey14`) et ses spectres, pris dans `meta.parquet` avec le split de A.
- Les recommandations de A (sprint 1) : certaines molécules ont des centaines de spectres, alors que le test en a environ 3. Faut-il plafonner le nombre de spectres par requête (tirage aléatoire reproductible) ? Ne garder que les adducts présents dans le test ?
- La masse neutre d'une requête : chaque spectre donne la sienne (fonction de B). Il faut les combiner, par exemple avec la médiane, en ignorant les NaN. À décider avec B.

**Livrable.** `build_queries` dans `evaluate.py`, avec les colonnes `query_id`, `true_inchikey14`, `row_id`, et les métadonnées des spectres.

### Tâche C1.3 — La fonction d'évaluation

**Objectif.** `evaluate(scorer, split="dev", ppm=10)` qui fait tourner tout le pipeline et renvoie les métriques.

**Pourquoi.** Toute l'équipe évaluera ses méthodes avec cette fonction. Elle garantit que les comparaisons sont justes.

**Ce que fait `evaluate`, pour chaque requête :**
1. calculer la masse neutre de la requête ;
2. récupérer les candidats avec `get_candidates` (B) ;
3. appeler `scorer(query_spectra, cand_idx)` ;
4. trier les candidats par score décroissant, avec un départage **reproductible** des égalités (graine fixe) ;
5. garder les 25 premiers `inchikey14`.

Puis, sur l'ensemble : MRR@25, rappel@1, @5, @10 et @25, **rappel du filtre** (vraie molécule parmi les candidats ?), nombre médian de candidats, temps de calcul. Renvoyer aussi un tableau détaillé par requête (rang obtenu, nombre de candidats…) pour l'analyse d'erreurs.

**Étapes proposées.**
1. Tant que A et B n'ont pas livré, travailler avec des versions provisoires : un split tiré toi-même, une masse neutre limitée à `[M+H]+` et `[M-H]-`, un catalogue fait des clés uniques. Remplace-les au merge.
2. Vérifier la vitesse sur `dev` (environ 2 000 requêtes) : quelques minutes au maximum, sinon vectoriser.

**Livrable.** `src/casmi/evaluate.py` et un test sur un mini-jeu de données inventé dont on connaît la réponse.

**Pièges.** Des égalités de score départagées dans l'ordre du catalogue : le résultat dépend alors de l'ordre des lignes, pas de la méthode. Toujours départager aléatoirement avec une graine.

### Tâche C1.4 — La baseline 0

**Objectif.** Le premier score du projet.

**Pourquoi.** C'est le plancher. Toute méthode qui utilise le spectre doit faire mieux. Il montre aussi ce que la masse seule permet, et donc pourquoi le spectre est nécessaire.

**Étapes proposées.**
1. `scorer_aleatoire` : score aléatoire, c'est-à-dire un ordre au hasard parmi les candidats.
2. `scorer_masse` : score = −|erreur de masse en ppm| entre la masse de la requête et la masse exacte du candidat.
3. Les évaluer sur `dev` puis sur toute la validation, pour plusieurs tolérances, et logger les résultats.

**À présenter en réunion.** Le MRR des deux baselines, le rappel du filtre, le nombre de candidats. Comment expliques-tu que la baseline masse fasse à peine mieux que l'aléatoire, ou nettement mieux ? (Pense aux isomères.)

### Tâche C1.5 — La première soumission

**Objectif.** `submission.py` qui produit un fichier valide pour Kaggle, et une première soumission.

**Pourquoi.** Vérifier tôt que toute la chaîne fonctionne jusqu'au bout, et éviter de découvrir un problème de format la veille de l'échéance.

**Étapes proposées.**
1. Charger le test (`load_test` de A), regrouper par `molecule_id`, appliquer le même pipeline avec le catalogue actuel (celui du train).
2. Convertir les 25 `cat_idx` en SMILES, les joindre par `;`, respecter l'ordre et les identifiants de `sample_submission.csv`.
3. Vérifier automatiquement : 400 lignes, tous les `molecule_id` présents, au plus 25 SMILES par ligne, SMILES lisibles par RDKit.
4. Soumettre la baseline 0.

**À présenter en réunion.** Le score obtenu. S'il est quasi nul, c'est normal et instructif : les molécules du test sont probablement absentes du catalogue du train. C'est l'argument pour le catalogue externe du sprint 4.

---

## Sprint 2 — Classer

### Tâche C2.1 — La similarité de Tanimoto

**Objectif.** `tanimoto_scores(pred_fp, cand_fps)` dans `ranking.py`.

**Pourquoi.** C'est l'étape 4 du pipeline : comparer le fingerprint prédit à celui de chaque candidat.

**À chercher / comprendre.**
- Pour deux vecteurs binaires : Tanimoto = (bits à 1 dans les deux) / (bits à 1 dans au moins un).
- Le modèle ne renvoie pas des bits mais des **probabilités** entre 0 et 1. La version continue s'écrit, pour une prédiction p et un candidat c : `Σ p·c / (Σ p² + Σ c − Σ p·c)`. Alternative à comparer : la similarité cosinus.
- Le calcul vectorisé : une prédiction contre une matrice de candidats en une seule opération NumPy.

**Livrable.** `tanimoto_scores` et ses tests (vecteurs identiques → 1 ; disjoints → 0 ; un cas calculé à la main).

### Tâche C2.2 — L'agrégation des spectres

**Objectif.** `aggregate_fps(fps)` : passer de plusieurs fingerprints prédits (un par spectre) à un seul par molécule.

**Pourquoi.** Une requête a plusieurs spectres (modes, adducts, énergies différents). Chacun apporte une information partielle.

**À chercher / comprendre.** Moyenne simple, maximum, moyenne pondérée (par le nombre de pics, par exemple). Autre option : calculer un score par spectre puis moyenner les **scores** plutôt que les fingerprints. Écris le code pour pouvoir changer de stratégie facilement : ce sera comparé au sprint 3.

**Livrable.** `aggregate_fps` dans `ranking.py`.

### Tâche C2.3 — L'oracle

**Objectif.** Le score maximal atteignable avec cette représentation, et la précision dont le modèle aura besoin.

**Pourquoi.** Si même le vrai fingerprint ne classe pas la bonne molécule en tête, le problème vient de la représentation (molécules aux fingerprints identiques, isomères très proches), pas du modèle. L'oracle sépare ces deux sources d'erreur.

**Étapes proposées.**
1. `scorer_oracle` : utilise le vrai fingerprint de la requête (via `true_inchikey14` → `cat_idx` → `load_catalog_fps()` de B). Évalue-le : c'est le plafond.
2. **Oracle bruité** : dégrade le vrai fingerprint, par exemple en inversant 1 %, 2 %, 5 %, 10 %, 20 % des bits, ou en ajoutant du bruit aux probabilités. Trace la courbe du MRR en fonction du bruit.
3. Coordonne-toi avec B, qui analyse en parallèle les fingerprints identiques et le pouvoir discriminant dans la fenêtre de masse.

**Livrable.** `notebooks/C/02_oracle.ipynb` et la courbe MRR / niveau de bruit.

**À présenter en réunion.** Les trois chiffres qui encadrent tout le projet : le rappel du filtre (plafond absolu), le MRR de l'oracle (plafond de la représentation) et le MRR de la baseline 0 (plancher). La courbe bruitée dit à quel point le modèle de B doit être précis pour être utile : c'est une excellente figure pour la conférence.

---

## Sprint 3 — Une baseline sans apprentissage, et la comparaison

### Tâche C3.1 — Les plus proches voisins spectraux

**Objectif.** `models/knn.py` : un scorer qui utilise le spectre sans rien entraîner.

**Pourquoi.** L'hypothèse : des spectres semblables viennent de molécules semblables. C'est une baseline simple, interprétable, et souvent difficile à battre. Si le MLP ne la dépasse pas, c'est une information importante.

**Principe, pour chaque spectre d'une requête :**
1. calculer sa similarité cosinus avec les spectres du train (matrices binnées et normalisées fournies par A) ;
2. garder les k plus similaires ;
3. fingerprint prédit = moyenne des fingerprints de leurs molécules, pondérée par la similarité ;
4. puis `aggregate_fps` et `tanimoto_scores`, comme pour le MLP.

**À chercher / comprendre.**
- Le produit de matrices creuses par blocs (`X_requete @ X_train.T`), ou `sklearn.neighbors.NearestNeighbors(metric="cosine", algorithm="brute")`.
- Restreindre les voisins au même mode d'ionisation ?
- L'effet de k (1, 5, 10, 50) et de la taille de la base de spectres (100 000, 500 000, tout le train).
- Pour aller plus loin : la bibliothèque `matchms` et la similarité « cosinus modifié », qui tient compte des décalages de masse entre spectres de molécules proches.

**Pièges.** La mémoire : une matrice de similarité 2 000 × 2 millions en dense ne tient pas. Travaille par blocs et ne garde que le top k à chaque bloc. Et vérifie qu'aucun spectre de validation n'est dans la base de voisins.

**Livrable.** `src/casmi/models/knn.py`, ses résultats dans `results.csv`.

### Tâche C3.2 — Comparaison et analyse d'erreurs

**Objectif.** Le tableau comparatif de toutes les méthodes, et la compréhension de leurs échecs.

**Étapes proposées.**
1. Brancher le MLP de B (`predict_fp`) dans un scorer.
2. Tableau : baseline aléatoire, baseline 0, k plus proches voisins, MLP, oracle, sur la même validation.
3. Analyse d'erreurs à partir du tableau par requête :
   - quelle part des échecs vient du filtre (vraie molécule absente des candidats) ?
   - le score varie-t-il selon l'adduct, le mode d'ionisation, la bibliothèque, le nombre de candidats ?
   - le score dépend-il de la nouveauté de la molécule (similarité maximale avec le train, calculée par B) ?
4. Regarder 5 échecs concrets : dessiner la vraie molécule et le premier candidat proposé. Se ressemblent-elles ?

**Livrable.** `notebooks/C/03_comparaison.ipynb`, le tableau et 3 ou 4 graphiques.

**À présenter en réunion.** Le tableau comparatif et la principale cause d'échec.

---

## Sprint 4 — Combiner et soumettre

### Tâche C4.1 — La fusion des scores

**Objectif.** Combiner plusieurs signaux en un score final.

**Pourquoi.** Le MLP, les plus proches voisins et l'erreur de masse portent des informations différentes. Leur combinaison est souvent meilleure que chacun pris seul.

**Étapes proposées.**
1. Score final = somme pondérée des scores (Tanimoto du MLP, Tanimoto des voisins, erreur de masse normalisée).
2. Choisir les poids sur `dev` (recherche en grille), puis rapporter le résultat sur le reste de la validation, jamais sur les données qui ont servi à choisir.
3. Option avancée : un modèle d'apprentissage du classement (*learning to rank*), par exemple `LGBMRanker` de LightGBM avec l'objectif `lambdarank`. Les features décrivent chaque couple (requête, candidat) : scores, erreur ppm, nombre de candidats, adduct… Attention : ce modèle doit être entraîné sur des requêtes dont les spectres n'ont pas servi à entraîner le MLP. Coupe la validation en deux moitiés.

**Livrable.** Le scorer combiné et ses résultats.

### Tâche C4.2 — Le catalogue externe

**Objectif.** Un catalogue élargi pour le test, si les règles le permettent (fiche C0.1).

**Pourquoi.** Les molécules du test viennent d'extraits biologiques et ne sont probablement pas dans le catalogue du train. Sans catalogue externe, le score Kaggle reste proche de 0, quelle que soit la qualité du modèle.

**À chercher / comprendre.**
- **COCONUT** : une base ouverte de produits naturels, la plus pertinente vu l'origine du test. **PubChem** : bien plus grande, à réserver à un sous-ensemble si le temps le permet.
- La standardisation des structures avec RDKit (lecture, SMILES canonique, conserver le plus gros fragment si la structure contient un sel), puis `inchikey14`, formule, masse exacte, fingerprint. Utilise les fonctions de B pour garantir les mêmes calculs.
- La fusion avec le catalogue du train, en dédoublonnant par `inchikey14`.

**Étapes proposées.**
1. Construire le catalogue élargi.
2. Mesurer l'augmentation du nombre de candidats par requête.
3. **Réévaluer la validation avec ce catalogue élargi** : les molécules supplémentaires sont autant de leurres. Le score baissera, mais il sera plus réaliste.

**Livrable.** `data/processed/catalog_ext.parquet` et ses fingerprints, et la comparaison des scores de validation avec les deux catalogues.

### Tâche C4.3 — La soumission finale

**Étapes proposées.** La meilleure méthode, avec le catalogue élargi. Vérification automatique du format. Notez le score public et comparez-le au score de validation. L'écart mesure à quel point la validation ressemble au test.

---

## Sprint 5 — Restitution

- Rédiger la partie « Évaluation et résultats » du rapport : protocole, baselines, oracle, tableau comparatif, analyse d'erreurs, score Kaggle.
- Rassembler les figures clés de l'équipe : courbe tolérance/candidats (B), courbe de l'oracle bruité, tableau comparatif, analyse d'erreurs.
- Proposer une trame pour la conférence de 20 minutes, par exemple :
  1. le problème et pourquoi ce n'est pas une classification (3 min) ;
  2. la démarche et le pipeline (4 min) ;
  3. les résultats en trois chiffres puis le tableau comparatif (4 min) ;
  4. l'IA dans le projet : où elle a accéléré, où elle s'est trompée, comment vous l'avez vérifiée, avec les meilleures anecdotes des trois journaux (7 min) ;
  5. limites et conclusion (2 min).

## Idées d'entrées pour ton journal IA

- Le code de métrique généré : correspondait-il exactement à la définition Kaggle (doublons, égalités, liste trop courte) ?
- Le code des plus proches voisins : tenait-il en mémoire à l'échelle réelle ?
- Les interprétations de résultats proposées par un assistant : étaient-elles soutenues par vos chiffres, ou plausibles mais non vérifiées ?

## Récapitulatif de tes livrables

| Sprint | Livrable | Qui l'attend |
|---|---|---|
| 0 | `docs/fiche_kaggle.md`, `CONTRIBUTING.md`, modèles de fichiers, `log_result` | Tout le monde |
| 1 | `metrics.py`, `build_queries`, `evaluate`, baseline 0, première soumission | A et B |
| 2 | `tanimoto_scores`, `aggregate_fps`, oracle et oracle bruité | B (sprint 3) |
| 3 | Plus proches voisins, tableau comparatif, analyse d'erreurs | Tout le monde |
| 4 | Fusion des scores, catalogue externe, soumission finale | Rapport |
| 5 | Partie résultats, trame de la conférence | Tout le monde |
