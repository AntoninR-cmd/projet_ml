# Décisions de l'équipe

Chaque décision indique qui l'a proposée et son statut : *proposée* (à discuter en réunion) ou *validée*.

## Sprint 1

### Catalogue de structures — proposée par B

- Une ligne par `inchikey14`. Si une molécule a plusieurs SMILES, on garde le plus fréquent parmi ses spectres ; à égalité, le premier dans l'ordre alphabétique.
- La formule et la masse exacte sont calculées par RDKit **depuis le SMILES**, et non depuis le précurseur.
- Le catalogue est trié par masse croissante : `cat_idx` suit l'ordre des masses, et les candidats d'une fenêtre de masse forment un intervalle contigu de `cat_idx`.
- Pour une molécule déjà chargée (SMILES avec `[N+]`…), `ExactMolWt` de RDKit retire la masse d'un électron, alors que `compute_neutral_masses` l'inclut. L'écart (environ 1 ppm) est négligeable devant la tolérance et n'est pas corrigé.

Détails : `notebooks/B/02_filtre_masse.ipynb`, section 1.

### Tolérance du filtre par masse : 10 ppm — proposée par B

Sur la validation provisoire, à 10 ppm : rappel du filtre de 95,2 % par spectre et 98,6 % par molécule, avec 41 candidats en médiane (180 au 90e centile). C'est le coude de la courbe « rappel / nombre de candidats » : au-delà, le rappel progresse peu alors que le nombre de candidats double à chaque palier.

À revoir avec l'évaluation de C : comparer le MRR à 10 et à 20 ppm (riken passe de 73,9 % à 87,2 % de rappel à 20 ppm).

Détails : `notebooks/B/02_filtre_masse.ipynb`, section 3.

### Une masse par molécule : la médiane — proposée par B, à discuter avec C

Une molécule a souvent plusieurs spectres, donc plusieurs masses neutres. Prendre la médiane de ces masses fait passer le rappel à 10 ppm de 95,2 % (par spectre) à 98,6 % (par molécule), et protège contre un spectre isolé mal étiqueté.

### Provisoire, en attendant A

- **Chemins** : les données brutes sont lues dans le dossier donné par la variable d'environnement `CASMI_DATA_DIR` (à défaut, `kaggle/input/`), et les fichiers produits sont écrits dans son sous-dossier `processed/` (à défaut, `data/processed/`). À reprendre dans `config.py`.
- **Split de validation** : 10 % des molécules tirées avec `np.random.default_rng(42)` parmi les `inchikey14` triés, et 2 000 d'entre elles pour `dev`. À remplacer par `split.parquet`.
