# Dictionnaire des données

Description de chaque colonne de `train.parquet` (18 colonnes, 2 539 608 spectres) et de `test.parquet` (12 colonnes, 1 213 spectres, 400 molécules).

Les pourcentages de valeurs manquantes (NA) sont calculés **sur tout le fichier**. Le test décrit ici est le test **visible**, un échantillon tiré du train ; il sera remplacé par le test caché lors de l'évaluation Kaggle (voir `fiche_kaggle.md`).

**Légende de la colonne « Feature ? »**
- **oui** : disponible dans le test et utilisable en entrée du modèle ;
- **non** : absente du test, identifiant, ou réponse à prédire ;
- **fuite** : calculée à partir de la vraie structure ; l'utiliser comme feature fausserait la validation ;
- **à risque** : disponible, mais son usage demande une précaution (voir les cas particuliers).

## 1. Vue d'ensemble

| Colonne | Description | Type | Exemple | NA train | NA test | Dans le test ? | Feature ? |
|---|---|---|---|---|---|---|---|
| `ingest_lib` | Bibliothèque spectrale d'origine (11 valeurs) | texte | `enveda-180` | 0 % | — | non | non : absente du test |
| `normalized_smiles` | Structure de la molécule (SMILES normalisé) | texte | `O=C1NC(=O)c2cc(...)cc21` | 0 % | — | non | non : c'est la réponse |
| `inchikey` | InChIKey complète (27 caractères, avec stéréochimie) | texte | `AAALVYBICLMAMA-UHFFFAOYSA-N` | 0 % | — | non | non : c'est la réponse |
| `inchikey14` | 14 premiers caractères de l'InChIKey : identité 2D de la molécule | texte | `AAALVYBICLMAMA` | 0 % | — | non | non : c'est la réponse (clé du split et du score) |
| `molecular_formula` | Formule brute | texte | `C20H15N3O2` | 0 % | — | non | non : déduite de la réponse |
| `ionization_mode` | Mode d'ionisation : `positive` ou `negative` | texte | `positive` | 0 % | 0 % | oui | **oui** |
| `instrument_type` | Type de spectromètre (82 libellés bruts) | texte | `timsTOF` | 1,30 % (+ 0,06 % de faux NA) | 0 % | oui | **oui**, après normalisation |
| `adduct` | Adduct normalisé (ion formé dans la source) | texte | `[M+H]+` | 0 % | 0 % | oui | **oui** (sert aussi à la masse neutre) |
| `adduct_orig` | Adduct tel qu'écrit par la bibliothèque | texte | `[M+H4N]+` | 0 % | — | non | non : absente du test, `adduct` suffit |
| `precursor_mz` | m/z de l'ion précurseur mesuré | nombre | `330.1237` | 0 % | 0 % | oui | **oui** (sert surtout à la masse neutre) |
| `precursor_error_ppm` | Écart entre la masse mesurée et la masse théorique de la vraie structure | nombre | `1.67` | 0,17 % | — | non | **fuite** (utilisable pour nettoyer le train) |
| `ms2_mzs` | m/z des pics du spectre MS/MS | liste de nombres | `[67.041, 73.007, …]` | 0 % | 0 % | oui | **oui** : entrée principale du modèle |
| `ms2_normalized_intensities` | Intensités des pics, normalisées (pic le plus haut = 1) | liste de nombres | `[0.002, 0.000, …]` | 0 % | 0 % | oui | **oui** : entrée principale du modèle |
| `num_peaks` | Nombre de pics du spectre | entier | `52` | 0 % | — | non | **oui**, recalculée par `len(ms2_mzs)` |
| `base_peak_intensity` | Intensité absolue du pic le plus haut, avant normalisation | nombre | `11000701.0` | 45,80 % | 0 % | oui | **à risque** : proposition de ne pas l'utiliser (voir section 2) |
| `collision_energy_ev` | Énergie(s) de collision converties en eV | liste de nombres | `[20.0, 40.0, 60.0]` | 13,29 % | 0 % | oui | **oui**, après traitement de la liste |
| `collision_energy_orig` | Énergie de collision telle qu'écrite par la source | texte | `"20,40,60"` | 12,62 % | 0 % | oui | non : texte brut, redondant avec `collision_energy_ev` |
| `collision_energy_orig_units` | Unité d'origine : `eV`, `NCE`, `V` ou `unknown` | texte | `NCE` | 0 % | 0 % | oui | **à risque** |
| `molecule_id` | Identifiant d'une molécule inconnue du test | texte | `m_005e53` | — | 0 % | test seul | non : identifiant (regroupe les spectres d'une requête) |
| `spectrum_id` | Identifiant d'un spectre du test | texte | `s_4d19d522` | — | 0 % | test seul | non : identifiant |

**En résumé :** 10 colonnes sont communes au train et au test. 8 colonnes n'existent que dans le train (la réponse, ses dérivés, et des informations de provenance), et 2 que dans le test (les identifiants). `num_peaks` est la seule colonne absente du test qu'on peut recalculer.

## 2. Cas particuliers

### Identité de la molécule : `inchikey14`, `inchikey`, `normalized_smiles`

- 275 810 `inchikey14` distincts, 275 950 `inchikey` et 277 566 `normalized_smiles`.
- `inchikey14` ne code que la **connectivité 2D**. 139 molécules ont plusieurs `inchikey` (des stéréo-isomères, que le score Kaggle ne distingue pas), et 1 612 ont plusieurs SMILES (écritures différentes d'une même structure 2D).
- C'est donc `inchikey14` qui identifie une molécule : c'est la clé du split (`split.parquet`), du catalogue et du score.

### `adduct` et `adduct_orig`

- Ils diffèrent pour 183 785 spectres (7,2 %). Ce sont les **mêmes ions écrits différemment** : `adduct` est la version normalisée. Exemples : `[M+H4N]+` → `[M+NH4]+` (53 452 spectres), `[M+CH2O2]-` et `[M+CHO2]-` → `[M+CH2O2-H]-` (83 188), `[M-HO]+` → `[M-H2O+H]+` (21 682), `[Cat]+` → `[M]+` (8 770).
- Seul `adduct` existe dans le test : on n'utilise jamais `adduct_orig`.
- 121 adducts dans le train, 7 dans le test visible (`[M+H]+` 79 %, `[M-H]-` 16 %, puis `[M+CH2O2-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+Cl]-`).

### `instrument_type`

- 82 libellés bruts pour quelques familles d'appareils : `timsTOF` (45 %), `Orbitrap` (29 %), puis `QTOF`, `LC-ESI-QTOF`, `ESI-QFT`, `LC-ESI-QFT`, `LC-ESI-ITFT`, `qTof`… avec des variantes de casse (`Orbitrap` / `orbitrap`, `QTOF` / `qtof` / `qTof`). **À normaliser** en familles avant tout usage.
- **Faux NA** : 1 645 spectres contiennent un texte au lieu d'une vraie valeur manquante (`"nan"` : 972, `"N/A"` : 376, chaîne vide : 297). `isna()` ne les détecte pas : il faut les convertir en NA. Avec eux, 1,36 % des spectres n'ont pas d'instrument.
- Le test visible est **100 % timsTOF**. On ne sait pas si c'est aussi le cas du test caché.

### `precursor_mz` et `precursor_error_ppm`

- `precursor_mz` est la mesure : avec `adduct`, elle donne la masse neutre de la molécule (`compute_neutral_masses`, B1.1). Quelques valeurs aberrantes dans le train : 52 spectres sous 50 et 25 au-dessus de 2 000 (maximum 2,4 millions). Dans le test visible, elle va de 245 à 460.
- `precursor_error_ppm` compare la mesure à la masse théorique **de la vraie structure** : elle n'existe que si l'on connaît la réponse. C'est une **fuite** : jamais comme feature. En revanche, elle mesure la qualité de l'étiquette et peut servir **à nettoyer le train** (A1.4).
- Sa distribution : médiane 1,5 ppm, 92,9 % des spectres sous 10 ppm, mais un 99e centile à 2 884 ppm (maximum 470 millions) : ces valeurs extrêmes signalent des étiquettes fausses (mauvais adduct, mauvaise structure). Sur timsTOF, l'erreur est beaucoup plus faible : 99 % des spectres sous 4,4 ppm.

### Les pics : `ms2_mzs`, `ms2_normalized_intensities`, `num_peaks`

- Deux listes de même longueur par spectre, sans aucun NA ni spectre vide. Elles pèsent 6,3 Go une fois décompressées : on les lit avec `load_spectra(row_ids)`, jamais en entier.
- `num_peaks` vaut `len(ms2_mzs)`. Elle n'est pas dans le test mais se recalcule.
- **Écart important entre train et test** : médiane de 42 pics par spectre dans le train (10 % sous 5 pics), contre 230 dans le test visible (10 % sous 36). À analyser en A1.3.

### `base_peak_intensity`

- Intensité absolue du pic le plus haut avant normalisation. Elle dépend de l'appareil et de la concentration de l'échantillon, pas seulement de la molécule.
- Le NA dépend **entièrement de la bibliothèque** : 0 % de NA pour `enveda-180`, `gnps`, `enveda-np-examples` et `masaryk`, 100 % pour les sept autres. Le simple fait qu'elle manque révèle donc la bibliothèque d'origine. Ces sept bibliothèques diffusent des spectres déjà normalisés : l'intensité absolue n'existe plus à la source.
- **Ce n'est pas toujours une mesure.** Dans gnps, 34 % des valeurs sont des constantes de normalisation : exactement 100 (31 %), 999, 1 000, 1 ou 10 000. Le spectre avait été normalisé avant d'être déposé.
- Quand c'est une vraie mesure, l'échelle dépend de l'appareil. Médiane en `log10` : Orbitrap 6,4, timsTOF 5,3, QTOF 4,4, soit un facteur 100 entre Orbitrap et QTOF. À appareil égal, elle varie encore d'un facteur 30 environ entre le premier et le dernier quartile (concentration, réglages).
- Elle est toujours remplie dans le test visible, avec des valeurs proches de celles d'enveda-180 (médiane d'environ 474 000).

**Pourquoi elle est peu utile pour prédire la structure**

En résumé, la valeur dépend de quatre propriétés de la bibliothèque et du laboratoire, et presque pas de la molécule :

```
base_peak_intensity = f( format de la base,          → existe ou pas (NA à 0 % ou 100 %)
                         convention du déposant,     → 100, 999, 1 000…
                         type de détecteur,          → facteur 100 entre appareils
                         réglages et concentration ) → facteur 30 à appareil égal
                      + (un peu) la molécule
```

- **Elle renseigne peu sur la structure.** Elle indique surtout *combien* de molécules étaient présentes et *avec quel appareil* on les a mesurées, pas *lesquelles*. Le seul lien chimique (certaines molécules s'ionisent mieux que d'autres) est noyé sous les effets de la concentration et de l'appareil.
- **Elle risque de biaiser le modèle.** Sa présence, ou une valeur comme 100, révèle la bibliothèque d'origine. Un modèle peut s'en servir comme raccourci (« ce spectre ressemble à ceux de gnps, je prédis des molécules de gnps »). Ce raccourci peut sembler marcher sur la validation, tirée du même mélange de bibliothèques, et échouer sur le test caché.
- **On ne peut pas l'imputer.** Pour les sept bibliothèques sans valeur, elle n'a jamais été enregistrée. Un imputeur se trompe d'un facteur 100 sur une bibliothèque qu'il n'a jamais vue (voir la section 3).
- **On ne peut pas non plus la retrouver ailleurs.** Ce n'est pas une propriété de la molécule, consultable dans une base comme PubChem, mais une propriété d'une acquisition précise : une injection, un jour, une concentration, un appareil. Les bases d'origine ne publient que des spectres normalisés, et même une valeur retrouvée dans les fichiers bruts resterait dans l'unité propre à son détecteur, sans comparaison possible avec les autres bibliothèques ni avec le test.

**Proposition** : ne pas l'utiliser comme feature, mais la garder comme indicateur de qualité (une intensité très faible signale souvent un spectre bruité), pour le nettoyage (A1.4) ou la pondération des spectres. À confirmer en A4.1 en comparant le MRR@25 avec et sans la variable, sur toute la validation et sur timsTOF à part.

### `collision_energy_ev`, `collision_energy_orig` et `collision_energy_orig_units`

- `collision_energy_ev` est une **liste** : 1 énergie pour 67 % des spectres, plusieurs pour 19,5 % (acquisition « étagée » : 2 à 6 valeurs, par exemple `[20, 40, 60]`), aucune (NA) pour 13,3 %. Les valeurs sont positives et triées, de 0 à 466 eV (médiane 40).
- `collision_energy_orig` est le texte d'origine, parfois signé en mode négatif (`"-60,-40,-20"` dans le test, converti en `[20.0, 40.0, 60.0]`), parfois inexploitable (`"15, 30, 45, 60, 70 or 90 (nominal)"`).
- 17 558 spectres ont un texte d'origine mais pas de valeur en eV, surtout quand l'unité est `NCE` (11 447) : l'énergie normalisée des Orbitrap ne se convertit pas directement en eV.
- `collision_energy_orig_units` : `eV` (54 %), `NCE` (31 %), `unknown` (13 %), `V` (2 %). Toujours `eV` dans le test visible.
- Pour une feature, il faudra résumer la liste (nombre d'énergies, minimum, maximum…) et traiter les NA (voir `meta_features`, A2.2).

## 3. Valeurs manquantes

Analyse complète : `notebooks/A/02_valeurs_manquantes.ipynb`. Outils : `src/casmi/missing.py`.

Colonnes concernées dans le train : `base_peak_intensity` (45,80 %), `collision_energy_ev` (13,29 %), `collision_energy_orig` (12,62 %), `instrument_type` (1,37 % une fois les faux NA convertis) et `precursor_error_ppm` (0,17 %). **Aucune colonne n'a de NA dans le test visible.**

### D'où viennent les NA

Ils dépendent presque entièrement de la **bibliothèque** :

| Bibliothèque | Spectres | `base_peak_intensity` | `collision_energy_ev` | `instrument_type` |
|---|---|---|---|---|
| enveda-180 (timsTOF) | 1 153 785 | 0 % | 0 % | 0 % |
| pluskal_ms2 | 527 581 | 100 % | 0 % | 0 % |
| riken | 347 171 | 100 % | 9,1 % | 4,7 % |
| gnps | 220 849 | 0 % | 100 % | 0,2 % |
| massbank | 101 727 | 100 % | 1,9 % | 0 % |
| mona | 92 416 | 100 % | 17,0 % | 0,6 % |
| spectraverse | 50 933 | 100 % | 92,4 % | 1,9 % |
| msdial | 40 765 | 100 % | 41,7 % | 33,0 % |
| drug_plus | 2 545 | 100 % | 100 % | 100 % |
| enveda-np-examples | 1 184 | 0 % | 0 % | 0 % |
| masaryk | 652 | 0 % | 100 % | 45,6 % |

- Les NA ne sont pas simultanés : il y a 17 combinaisons différentes. Seuls 1 154 969 spectres (45 %) sont complets, et ce sont presque exactement ceux d'enveda-180 : **supprimer les spectres incomplets revient à n'entraîner que sur timsTOF**.
- `collision_energy_ev` et `collision_energy_orig` manquent presque toujours ensemble. Quand seule la valeur en eV manque, c'est en général une énergie NCE non convertie.
- `precursor_error_ppm` n'est pas une feature (fuite) : ses NA n'ont pas à être traités.

### Reconstruction par règle

Pour une énergie NCE simple, la valeur en eV vaut exactement `NCE × precursor_mz / 500`, arrondie au dixième (vérifié sur 342 722 spectres, exact à 99,99 %). La règle (`nce_to_ev`) reconstruit **11 064** des 337 443 NA de `collision_energy_ev`. Pour les autres, le texte d'origine manque aussi.

### Choix des imputeurs

Méthode : on masque des valeurs observées, on les prédit, et on compare l'erreur de trois imputeurs (médiane ou mode global ; par groupe ; modèle de gradient boosting qui n'utilise que des variables présentes dans le test). Deux masquages : au hasard (20 %), et par bibliothèque entière, qui reproduit le vrai mécanisme des NA. Calcul sur les molécules du split `train` seulement.

| Variable | Imputeur retenu | Erreur en masquage aléatoire | Erreur en masquage par bibliothèque |
|---|---|---|---|
| `base_peak_intensity` | **aucun n'est fiable** | 0,67 en `log10` (facteur 4,7) | ≈ 2,0 en `log10` sur gnps (facteur 100), pire qu'une médiane |
| `collision_energy_ev` | **règle NCE → eV**, puis **modèle** | 11,5 eV | 15,3 eV en moyenne (médiane par groupe : 16,6 ; globale : 19,4) |
| `instrument_type` (famille) | **modèle** | 96 % de familles retrouvées | de 49 % à 100 % selon la bibliothèque |

Le masquage aléatoire surestime nettement la qualité de l'imputation : sans le masquage par bibliothèque, on aurait cru `base_peak_intensity` imputable.

### Stratégies à comparer en A4.1

Critère : MRR@25 sur `dev`, puis sur toute la validation et sur les requêtes timsTOF à part.

- `base_peak_intensity` : supprimer la variable, ou l'imputer par le modèle avec une indicatrice « était manquant ».
- `collision_energy_ev` et `instrument_type` : retirer les spectres concernés de l'entraînement, ou les imputer (règle puis modèle) avec une indicatrice.

Les requêtes de validation ne sont jamais supprimées : il faut répondre pour toutes les molécules.
