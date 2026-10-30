# Fiche Kaggle — Enveda CASMI 2026

**Compétition :** Enveda CASMI 2026 — Molecule ID From Mass Spectra  
**Organisateur :** Enveda Therapeutics, Inc.  
**Page officielle :** https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra  
**Informations vérifiées le :** 28 octobre 2026

## 1. Objectif et origine des molécules

À partir de spectres LC-MS/MS, il faut proposer jusqu’à 25 structures moléculaires au format SMILES pour chaque `molecule_id`, classées de la plus probable à la moins probable.

La compétition porte sur des molécules pouvant ressembler à celles présentes dans des échantillons naturels de plantes, de mammifères ou de microbes. Cela inclut des produits naturels confirmés ou hypothétiques, leurs analogues et des molécules synthétiques plausibles dans la nature.

Les prédictions sont faites **par molécule**, et non par spectre. Les informations des différents spectres d’une même molécule doivent donc être regroupées pour produire une seule liste de candidats.

## 2. Métrique et comparaison des structures

La métrique utilisée est le **Mean Reciprocal Rank à 25 (MRR@25)** :

\[
\mathrm{MRR@25} = \frac{1}{U}\sum_{u=1}^{U}\frac{1}{rang_u}
\]

`rang_u` est la position de la première structure correcte pour la molécule `u`. Si aucune proposition des 25 premières n’est correcte, le score de cette molécule est 0.

Exemples :

- bonne réponse au rang 1 : score de 1 ;
- bonne réponse au rang 2 : score de 0,5 ;
- bonne réponse au rang 25 : score de 0,04.

Une structure est considérée correcte si elle a la même connectivité atomique que la réponse. Kaggle applique une canonicalisation des tautomères avec RDKit 2026.03.3, puis compare les 14 premiers caractères de l’InChIKey (`InChIKey14`). La stéréochimie et la forme tautomère ne départagent donc pas les réponses.

### Traitement des SMILES invalides et des candidats en double

- **SMILES invalide :** le notebook demande à RDKit de lire chaque SMILES. Si RDKit ne peut pas le parser, le candidat est traité comme une non-correspondance ; la soumission n’est pas rejetée pour ce seul candidat. Il garde toutefois sa position dans la liste et peut donc faire descendre le rang d’un candidat correct. Une ligne sans aucune proposition non vide, elle, est rejetée.
- **Candidats en double :** le notebook ne déduplique pas les propositions d’une ligne. Chaque entrée compte comme un rang, même si la structure a déjà été proposée. Seule la première proposition correspondant à la réponse détermine le score. Des doublons peuvent donc faire descendre une bonne réponse.
- **À distinguer :** répéter un `molecule_id` dans plusieurs lignes est une erreur qui fait rejeter la soumission ; répéter un SMILES dans la liste d’une ligne ne la fait pas rejeter.

Notebook de métrique : https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank

## 3. Format de soumission

La soumission est un fichier CSV avec deux colonnes : `molecule_id` et `smiles`.

- Chaque `molecule_id` doit apparaître exactement une fois.
- La colonne `smiles` contient jusqu’à 25 propositions, dans l’ordre de préférence.
- Les SMILES d’une même ligne sont séparés par des points-virgules (`;`).
- Le fichier ne doit contenir ni valeur manquante ni `molecule_id` dupliqué.
- Une ligne ne peut pas contenir plus de 25 propositions ; moins de 25 est autorisé.

Exemple :

```csv
molecule_id,smiles
m_0014ef,SMILES_1;SMILES_2;SMILES_3
```

## 4. Données de la compétition

Le jeu d’entraînement contient environ 2,5 millions de spectres couvrant environ 275 000 structures uniques.

Le fichier `test.parquet` visible avant l’évaluation est un échantillon tiré des données d’entraînement. Il sera remplacé par le test caché lors de la réexécution du notebook. Le test caché est annoncé comme comprenant environ 1 500 spectres pour 400 molécules, avec 1 à 16 spectres par molécule (médiane : 3).

Les données du concours sont sous licence **CC BY-NC 4.0**. Leur utilisation est limitée aux fins non commerciales autorisées par le règlement, notamment la compétition, les forums Kaggle, la recherche académique et l’enseignement. Elles ne doivent pas être redistribuées à des personnes qui n’ont pas accepté les règles.

## 5. Données externes, licences et outils

Le règlement autorise l’utilisation de données externes si elles sont publiquement accessibles et également accessibles aux participants sans coût, ou si elles respectent les critères d’accessibilité raisonnable et de coût minimal.

Kaggle indique que les données externes et les modèles pré-entraînés librement et publiquement accessibles sont autorisés, sous réserve du règlement.

Le code d’une solution gagnante doit être publié sous une licence open source approuvée par l’OSI qui n’interdit pas l’usage commercial. L’organisateur recommande d’éviter les données, modèles ou outils externes dont les licences seraient incompatibles avec ces obligations.

Avant d’utiliser une ressource externe, il faut donc vérifier et noter sa licence.

## 6. Exigences de soumission par notebook

La soumission doit être effectuée depuis un notebook Kaggle :

- durée maximale : 9 heures sur CPU ou GPU ;
- accès à Internet désactivé ;
- fichier produit nommé `submission.csv`.

## 7. Équipe, limites et calendrier

- Taille maximale d’une équipe : **5 personnes**.
- Nombre maximal de soumissions : **5 par jour**.
- Nombre de soumissions finales sélectionnables : **2**.

Dates officielles, en UTC :

- Début : 14 septembre 2026.
- Date limite d’inscription : 7 décembre 2026 à 23 h 59.
- Date limite de fusion des équipes : 7 décembre 2026 à 23 h 59.
- Date limite de soumission finale : 14 décembre 2026 à 23 h 59.

## 8. Classement public et privé

Le classement public est calculé sur environ 33 % des données de test. Le classement privé final utilise les 67 % restants. Le score public peut donc différer du résultat final.

## 9. Conséquences pour notre projet

- **Évaluation locale :** le `test.parquet` visible est un échantillon tiré du train ; il sert à vérifier que notre pipeline fonctionne, pas à mesurer sa performance. Pour comparer nos méthodes, nous utiliserons une validation séparée par molécule. Le classement Kaggle, lui, est calculé sur les données de test cachées.
- **Catalogue de candidats :** les fichiers fournis ne contiennent pas de liste de structures candidates. Nous devons construire notre propre catalogue à partir des sources autorisées par le règlement.
- **Classement des candidats :** avant de produire une soumission, nous retirerons les SMILES illisibles par RDKit et dédupliquerons les structures avec la même procédure que le scorer officiel : suppression de la stéréochimie, canonicalisation des tautomères avec RDKit 2026.03.3, puis comparaison de l’InChIKey14. Les candidats invalides ou répétés gardent autrement une place dans la liste et peuvent faire descendre la bonne réponse.
- **Validation par molécule :** le test caché est annoncé avec 1 à 16 spectres par molécule, médiane 3. Nous coordonnerons avec A la façon de limiter ou échantillonner les spectres de validation pour que `build_queries` ressemble au test.
- **Exécution Kaggle :** le notebook doit fonctionner sans accès à Internet et produire `submission.csv`. Les fichiers nécessaires à l’inférence qui ne sont pas fournis par Kaggle devront être disponibles comme entrées du notebook, sous réserve des licences et du règlement.
- **Données et GitHub :** ne pas ajouter les fichiers du concours (`train.parquet`, `test.parquet` ou leurs copies) à Git. Avant de partager publiquement un catalogue, des fingerprints ou d’autres fichiers dérivés du train, vérifier qu’ils respectent les règles de données et les licences. En cas de doute, les garder hors du dépôt public.

Sources :
- Données et fichiers fournis : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data
- Métrique officielle : https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
- Règles de la compétition : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules

## Sources

- Overview et Evaluation : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview
- Data : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data
- Rules : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
- Leaderboard : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard
- Notebook de métrique : https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
- Rappel de l’organisateur sur les licences : https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359