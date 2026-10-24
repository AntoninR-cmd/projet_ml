# Projet CASMI 2026 — Feuille de route du membre B

**Rôle : Chimie, catalogue & modèle**

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

# PARTIE 2 — Tes tâches : membre B, chimie, catalogue & modèle

## 2.0 Ta mission en résumé

Tu es **le chimiste de l'équipe** et le responsable du **premier modèle appris**. Concrètement :

- tu expliques à l'équipe les notions de spectrométrie de masse (sprint 0) ;
- tu retrouves la masse de la molécule à partir du précurseur et de l'adduct (sprint 1) ;
- tu construis le catalogue de structures et le filtre par masse, c'est-à-dire l'étape qui produit les candidats (sprint 1) ;
- tu fabriques la cible du modèle, les fingerprints, et tu l'analyses (sprint 2) ;
- tu entraînes le MLP qui prédit le fingerprint à partir du spectre (sprint 3) ;
- tu l'améliores à partir de la littérature (sprint 4).

Tes livrables du sprint 1 (`compute_neutral_masses`, `get_candidates`) bloquent C : livre-les en priorité, même dans une première version simple.

Chaque tâche suit le même format : **Objectif**, **Pourquoi**, **À chercher / comprendre**, **Étapes proposées**, **Livrable**, **Pièges**, **À présenter en réunion**.

---

## Sprint 0 — Les bases

### Tâche B0.1 — La fiche chimie de l'équipe

**Objectif.** Une fiche de 1 à 2 pages, `docs/fiche_chimie.md`, qui permet à A et C de comprendre les données.

**Pourquoi.** Personne ne peut prendre de bonnes décisions sur les données sans savoir ce qu'est un adduct ou pourquoi deux molécules peuvent avoir la même masse.

**À chercher / comprendre.** La fiche doit répondre à ces questions, avec un exemple chiffré chaque fois que possible :
1. Comment fonctionne une expérience de MS/MS : ionisation (électrospray), sélection du précurseur, fragmentation par collision, détection des fragments.
2. Pourquoi `precursor_mz` n'est pas la masse de la molécule. Que signifient `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[2M+Na]+`, `[M-H2O+H]+` ?
3. Mode positif et mode négatif : qu'est-ce qui change ?
4. Masse monoisotopique et masse moyenne : laquelle utilise-t-on ici, et pourquoi ?
5. Qu'est-ce qu'une erreur en ppm, et quelle précision atteignent les instruments modernes ?
6. Qu'est-ce qu'un isomère, et pourquoi la masse ne suffit-elle pas à identifier une molécule ?
7. Que représentent un SMILES, une InChIKey, et ses 14 premiers caractères ? Pourquoi deux stéréoisomères ont-ils le même `inchikey14` ?
8. Qu'apporte l'énergie de collision au spectre ?

**Étapes proposées.** Lire deux ou trois sources fiables (cours universitaires de spectrométrie de masse, documentation des bases MassBank ou GNPS). Si tu utilises un assistant IA pour les explications, vérifie chaque affirmation chiffrée dans une source et note-le dans ton journal.

**Livrable.** `docs/fiche_chimie.md`, présentée en 10 minutes à la réunion de lancement du sprint 1.

### Tâche B0.2 — Prendre en main RDKit

**Objectif.** Savoir passer d'un SMILES à tout ce dont l'équipe a besoin.

**À chercher / comprendre.**
- `Chem.MolFromSmiles` (renvoie `None` si le SMILES est invalide), `Chem.MolToSmiles` (forme canonique).
- `Descriptors.ExactMolWt` (masse monoisotopique), `rdMolDescriptors.CalcMolFormula` (formule brute).
- `Chem.MolToInchiKey`.
- `Draw.MolToImage` pour dessiner.
- `rdFingerprintGenerator.GetMorganGenerator` et `DataStructs.BulkTanimotoSimilarity` (utilisés au sprint 2).

**Étapes proposées.**
1. Sur un échantillon de quelques milliers de spectres **répartis sur tout le fichier** (pas seulement le début, qui ne contient que quelques bibliothèques), recalculer l'InChIKey depuis `normalized_smiles` et comparer ses 14 premiers caractères à la colonne `inchikey14`.
2. Compter combien de `normalized_smiles` différents existent pour un même `inchikey14` (des stéréoisomères ?).
3. Afficher 3 molécules avec un de leurs spectres, pour illustrer la fiche chimie.

**Livrable.** `notebooks/B/00_rdkit.ipynb`.

---

## Sprint 1 — La masse et les candidats

### Tâche B1.1 — Le parser d'adducts et la masse neutre

**Objectif.** `parse_adduct(adduct)` et `compute_neutral_masses(df)` dans `mass.py`.

**Pourquoi.** C'est l'étape 1 du pipeline. Toute la recherche de candidats repose sur une masse neutre juste. Une erreur ici fait disparaître la bonne réponse avant même que le modèle intervienne.

**À chercher / comprendre.**
- Le jeu de données contient environ 120 adducts différents. Liste-les tous avec leur fréquence, sur tout le fichier.
- La structure générale d'un adduct : `[nM ± X ± Y]z±`
  - `n` : nombre de molécules (1 si absent) ;
  - `± X` : atomes ou groupes ajoutés ou perdus (`+H`, `+Na`, `-H2O`, `+CH2O2`…), éventuellement avec un coefficient (`-2H2O`) ;
  - `z` : charge, avec son signe (`+`, `2+`, `-`…).
- La formule : la masse de l'ion vaut `|z| × m/z`. Elle est égale à `n × M + (masse des ajouts) − (masse des pertes) − z × m_e`, où `m_e` ≈ 0,000549 Da est la masse de l'électron (un ion positif a perdu des électrons). On isole M :

  `M = (|z| × m/z − ajouts + pertes + z × m_e) / n`

  Vérification sur `[M+H]+` : M = m/z − m(H) + m_e = m/z − m(proton). C'est cohérent.
- Les masses monoisotopiques des éléments : `Chem.GetPeriodicTable().GetMostCommonIsotopeMass(...)`.
- Les expressions régulières en Python (module `re`) : `re.fullmatch`, `re.findall`, groupes de capture.

**Étapes proposées.**
1. Écrire `parse_adduct` qui renvoie `(n, delta, z)` où `delta` est la somme signée des masses ajoutées et perdues, et qui **lève une erreur** pour tout ce qui n'est pas interprétable.
2. Écrire `compute_neutral_masses(df)` : parser chaque adduct distinct une seule fois, puis appliquer la formule de façon vectorisée. Renvoyer NaN pour les adducts non interprétables.
3. Tests unitaires avec des valeurs calculées à la main. Par exemple, avec `precursor_mz = 500` :
   - `[M+H]+` → 498,992724 ;
   - `[M-H]-` → 501,007276 ;
   - un adduct invalide → NaN.
4. **Validation sur les données réelles.** Sur un échantillon réparti sur toutes les bibliothèques, comparer la masse neutre calculée à `ExactMolWt` du SMILES. Tracer la distribution des erreurs en ppm, par bibliothèque et par adduct. Lister les adducts où la plupart des erreurs sont grandes : c'est soit un bug de ton parser, soit des étiquettes douteuses.

**Livrable.** `src/casmi/mass.py`, ses tests, `notebooks/B/01_masse_neutre.ipynb`.

**Pièges.**
- Une expression régulière trop permissive qui accepte silencieusement des adducts invalides (par exemple `[Cat]2+`, lu comme du calcium suivi d'un caractère ignoré). Un parser doit échouer bruyamment plutôt que de renvoyer une masse fausse.
- Oublier le signe de la masse de l'électron, ou la diviser par `n`.
- Le cas `[M]+` (molécule déjà chargée) : compare avec `ExactMolWt` et regarde si l'écart est systématique.

**À présenter en réunion.** Le pourcentage de spectres dont l'erreur est inférieure à 5, 10 et 20 ppm, par bibliothèque. La liste des adducts non interprétables (pour le nettoyage de A).

### Tâche B1.2 — Le catalogue de structures

**Objectif.** `catalog.parquet` et `load_catalog()`.

**Pourquoi.** C'est le « fichier de photos » dans lequel on cherche. En validation, il contient toutes les molécules du train et de la validation.

**À chercher / comprendre.**
- Une ligne par `inchikey14`. Si plusieurs SMILES existent pour la même clé, en choisir un de façon reproductible (le plus fréquent, par exemple).
- Pour chaque structure : formule brute et masse exacte calculées **depuis le SMILES** (et non depuis le précurseur, qui contient l'erreur de mesure).
- Les SMILES que RDKit ne sait pas lire : combien, et que faire ?
- Le temps de calcul sur environ 276 000 molécules ; `multiprocessing` si c'est trop lent.

**Étapes proposées.**
1. Extraire les couples uniques (`inchikey14`, `normalized_smiles`) de `meta.parquet` (ou directement du train si A n'a pas encore livré).
2. Calculer formule et masse, attribuer `cat_idx = 0, 1, 2…`.
3. Sauvegarder.

**Livrable.** `src/casmi/catalog.py`, `data/processed/catalog.parquet`.

### Tâche B1.3 — Le filtre par masse

**Objectif.** `get_candidates(neutral_mass, ppm=10)`, et le choix argumenté de la tolérance.

**Pourquoi.** La tolérance est un compromis :
- trop serrée, la vraie molécule est exclue à cause de l'erreur de mesure, et rien ne pourra la rattraper ;
- trop large, il y a des milliers de candidats, et le classement devient plus difficile.

**À chercher / comprendre.**
- Une recherche rapide : trier les masses du catalogue une fois, puis `np.searchsorted` pour trouver l'intervalle `[M − δ, M + δ]` avec `δ = M × ppm × 10⁻⁶`.
- Le nombre d'**isomères** par formule dans le catalogue : c'est le nombre minimal de candidats que la masse ne pourra jamais départager.

**Étapes proposées.** Sur les spectres de validation (split de A, ou ton propre tirage provisoire) :
1. pour chaque tolérance (2, 5, 10, 20, 50 ppm), calculer le **rappel du filtre** : la proportion de spectres dont la vraie molécule est parmi les candidats ;
2. pour chaque tolérance, la distribution du **nombre de candidats** (médiane, 90e percentile, maximum) ;
3. tracer les deux courbes ensemble et proposer une tolérance par défaut ;
4. identifier les spectres dont l'erreur est énorme et les signaler à A pour le nettoyage.

**Livrable.** `get_candidates` dans `catalog.py`, ses tests, `notebooks/B/02_filtre_masse.ipynb`, et la tolérance retenue dans `docs/decisions.md`.

**Pièges.** Une molécule de validation a souvent plusieurs spectres, donc plusieurs masses neutres un peu différentes. Discute avec C de la façon de les combiner en une seule masse par molécule (la médiane, par exemple).

**À présenter en réunion.** La courbe « rappel du filtre / nombre de candidats » selon la tolérance. C'est l'une des figures clés du projet.

---

## Sprint 2 — La cible du modèle

### Tâche B2.1 — Les fingerprints

**Objectif.** `smiles_to_fp(smiles)`, et la matrice `catalog_fp.npy` pour tout le catalogue.

**Pourquoi.** C'est la cible que le modèle apprendra à prédire, et la représentation qui servira à comparer les candidats.

**À chercher / comprendre.**
- Le principe de l'algorithme de Morgan (équivalent de l'ECFP) : on décrit l'environnement de chaque atome jusqu'à un rayon r, puis chaque environnement est « haché » vers un numéro de bit. Rayon 2 ≈ ECFP4.
- Les **collisions** : avec 2048 bits, deux motifs différents peuvent tomber sur le même bit.
- Les paramètres : rayon (1, 2, 3), taille (1024, 2048, 4096), version binaire ou à comptage.
- Une alternative interprétable : les **clés MACCS** (166 motifs définis à la main, par exemple « contient un soufre »). Moins précises, mais chaque bit a un sens lisible, ce qui est précieux pour expliquer le modèle à l'oral.
- Le stockage : 276 000 × 2048 octets ≈ 565 Mo en `uint8`. `np.packbits` divise par 8 sur disque.

**Étapes proposées.**
1. Fixer les paramètres de référence (rayon 2, 2048 bits, binaire) et les noter dans `config.py`.
2. Calculer les fingerprints du catalogue, chronométrer, sauvegarder.
3. Écrire `load_catalog_fps()` et un test : `smiles_to_fp(smiles du cat_idx i)` doit être égal à la ligne `i` de la matrice.

**Livrable.** `src/casmi/fingerprints.py`, `data/processed/catalog_fp.npy`.

### Tâche B2.2 — Analyser les cibles

**Objectif.** Comprendre ce que le modèle devra apprendre, et à quel point la validation est difficile.

**Pourquoi.** Ces analyses expliquent les résultats à venir. Un bit présent dans 0,01 % des molécules sera très difficile à prédire. Une validation pleine de « quasi-jumeaux » des molécules du train sera trop facile.

**À chercher / comprendre.**
1. **Fréquence des bits** dans le catalogue : combien sont presque toujours à 0 ? presque toujours à 1 ? Quelle est la proportion moyenne de bits à 1 ? (C'est important pour la loss du sprint 3.)
2. **Fingerprints identiques** : combien de molécules différentes partagent exactement le même fingerprint ? Ces molécules seront impossibles à départager par le classement.
3. **Nouveauté de la validation** : pour chaque molécule `dev`, la similarité de Tanimoto maximale avec une molécule du train (`BulkTanimotoSimilarity`). Une distribution concentrée vers 1 signifie que la validation contient surtout des analogues proches de molécules connues.
4. **Pouvoir discriminant dans la fenêtre de masse** : pour chaque molécule `dev`, combien de ses candidats ont un Tanimoto supérieur à 0,8 ou 0,9 avec elle ? C prépare l'« oracle » en parallèle, coordonnez-vous pour ne pas faire deux fois le même calcul.

**Livrable.** `notebooks/B/03_analyse_fingerprints.ipynb`, avec 3 ou 4 graphiques commentés.

**À présenter en réunion.** La distribution de nouveauté de la validation, et la question à trancher avec A : faut-il un split par scaffold, plus exigeant ?

### Tâche B2.3 — Préparer l'entraînement

**Objectif.** Une boucle d'entraînement PyTorch qui fonctionne, testée sur des données factices.

**Pourquoi.** Au sprint 3, tu n'auras pas de temps à perdre sur la mécanique de PyTorch.

**À chercher / comprendre.** `torch.nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.BCEWithLogitsLoss`, `torch.optim.Adam`, la boucle époque / batch / `loss.backward()` / `optimizer.step()`, le mode `model.eval()` avec `torch.no_grad()`, la sauvegarde des poids. Où trouver un GPU gratuit : Kaggle Notebooks (quota hebdomadaire) ou Google Colab.

**Livrable.** Un squelette `models/mlp.py` qui s'entraîne sur des tenseurs aléatoires sans erreur.

---

## Sprint 3 — Le premier modèle appris

### Tâche B3.1 — Le MLP spectre → fingerprint

**Objectif.** Un modèle entraîné, et la fonction `predict_fp(query_spectra)` qui renvoie, pour chaque spectre, une probabilité par bit.

**Pourquoi.** C'est l'étape 3 du pipeline, le cœur du projet.

**Architecture de départ (à faire évoluer).**
```
entrée : spectre binné (A) + features de métadonnées (A)
  → Linear(n_entrée, 2048) → ReLU → Dropout(0,2)
  → Linear(2048, 2048)     → ReLU → Dropout(0,2)
  → Linear(2048, 2048)     → 2048 logits (un par bit)
loss : BCEWithLogitsLoss (une classification binaire par bit)
```

**À chercher / comprendre.**
- Pourquoi `BCEWithLogitsLoss` plutôt qu'une sigmoïde suivie de `BCELoss` (stabilité numérique).
- **Le déséquilibre des classes** : la plupart des bits valent 0. Un modèle qui prédit « tout à 0 » obtient une loss faible mais ne sert à rien. Surveille les bonnes métriques, et regarde le paramètre `pos_weight`.
- L'early stopping : arrêter quand la loss de validation remonte.
- Comment mesurer un fingerprint prédit, en dehors du MRR :
  - Tanimoto moyen entre la prédiction seuillée à 0,5 et le vrai fingerprint ;
  - précision et rappel des bits à 1.

**Étapes proposées.**
1. Entraîner d'abord sur `X_train_sample` (fourni par A au sprint 2) et évaluer sur `X_dev`. Vérifier que le modèle peut sur-apprendre un tout petit échantillon (si la loss ne descend pas sur 1 000 spectres, il y a un bug).
2. Écrire `predict_fp`, puis construire le scorer avec les fonctions de C : `aggregate_fps` pour combiner les spectres d'une molécule, `tanimoto_scores` pour noter les candidats. Lancer `evaluate(scorer)`.
3. Passer à tout le train avec `make_loader` de A.
4. Sauvegarder les poids (hors git, dans `data/processed/models/`) et les hyperparamètres utilisés.

**Livrable.** `src/casmi/models/mlp.py`, `notebooks/B/04_mlp.ipynb`, une ligne dans `results.csv`.

**Pièges.**
- Fuite : un spectre de validation dans l'entraînement.
- Les statistiques de normalisation des entrées calculées sur tout le jeu au lieu du train seul.
- Les données triées par bibliothèque, non mélangées.
- Comparer des scores obtenus sur des splits différents.

**À présenter en réunion.** Les courbes de loss (train et validation), le Tanimoto moyen, le MRR comparé à la baseline 0 et à l'oracle de C.

---

## Sprint 4 — Améliorer le modèle

### Tâche B4.1 — Une ou deux pistes d'amélioration

**Objectif.** Tester une ou deux idées, mesurer leur effet sur le MRR, et l'expliquer.

**Pistes possibles** (en choisir une ou deux, une modification à la fois) :
- gérer le déséquilibre : `pos_weight`, *focal loss*, ou ignorer les bits trop rares ;
- changer la cible : 4096 bits, fingerprint à comptage, rayon 3 ;
- changer l'architecture : plus large, plus profonde, connexions résiduelles, normalisation de batch ;
- ajouter des entrées : le vecteur des pertes neutres de A ;
- un ensemble de plusieurs modèles (moyenne des prédictions) ;
- un objectif secondaire, par exemple prédire aussi le nombre d'atomes de chaque élément.

**Lectures pour trouver des idées.** Lis le résumé et la figure principale de chaque article, et note une idée applicable à votre échelle :
- **CSI:FingerID** (Dührkop et al., PNAS 2015) : la méthode de référence historique pour prédire un fingerprint à partir d'un spectre, intégrée au logiciel SIRIUS ;
- **MIST** (Goldman et al., Nature Machine Intelligence, 2023) : un transformer qui prédit des fingerprints à partir des pics, en s'aidant des formules des fragments ;
- **MassSpecGym** (Bushuiev et al., NeurIPS 2024) : un banc d'essai qui définit exactement notre tâche de recherche de structure parmi des candidats, avec des baselines ;
- **DreaMS** (Bushuiev et al., 2025) : un modèle pré-entraîné sur des millions de spectres non annotés, dont les représentations peuvent servir d'entrées.

**Livrable.** Les résultats dans `results.csv` et un paragraphe d'interprétation par expérience.

---

## Sprint 5 — Restitution

- Rédiger les parties « Notions de chimie », « Catalogue et filtre par masse » et « Modèle » du rapport.
- Préparer 2 ou 3 diapositives : l'idée du fingerprint (avec un exemple de clés MACCS lisibles), la courbe tolérance/candidats, les courbes d'apprentissage.
- Relire ton journal IA et choisir les 2 meilleures anecdotes.

## Idées d'entrées pour ton journal IA

- Le parser d'adducts généré par un assistant : accepte-t-il des cas qu'il devrait refuser ? Comment l'as-tu découvert ?
- Les explications de chimie : lesquelles étaient exactes, lesquelles imprécises ou fausses une fois vérifiées dans une source ?
- Les hyperparamètres suggérés pour le MLP : venaient-ils d'une justification, ou d'une habitude générique ?

## Récapitulatif de tes livrables

| Sprint | Livrable | Qui l'attend |
|---|---|---|
| 0 | `docs/fiche_chimie.md`, notebook RDKit | Tout le monde |
| 1 | `compute_neutral_masses`, liste des adducts non interprétables | A et C |
| 1 | `catalog.parquet`, `get_candidates`, tolérance retenue | C |
| 2 | `smiles_to_fp`, `catalog_fp.npy`, analyse des cibles | A (sprint 3) et C |
| 3 | MLP entraîné, `predict_fp` | C |
| 4 | Expériences d'amélioration | Rapport |
