# Contribuer au projet CASMI 2026

## Organisation du code

- Le code réutilisable va dans `src/casmi/`.
- Les notebooks d’exploration vont dans `notebooks/A/`, `notebooks/B/` ou `notebooks/C/`.
- Les tests vont dans `tests/`.
- Les données brutes et les données traitées vont dans `data/`. Elles ne doivent pas être ajoutées à Git.
- Chaque membre travaille principalement sur les fichiers qui lui sont attribués dans le plan du projet. Pour modifier le fichier d’un autre membre, il faut se coordonner avec lui.
- Les noms des fonctions et des fichiers du contrat d’interfaces ne changent qu’après une décision en réunion, consignée dans `docs/decisions.md`.

## Mise en place

À faire une seule fois par personne. Ne pas recloner si le dépôt est déjà présent sur l’ordinateur.

```bash
git clone git@github.com:AntoninR-cmd/projet_ml.git
cd projet_ml
python -m venv .venv
source .venv/bin/activate
python -m pip install nbstripout
nbstripout --install
```
### Créer et activer l’environnement Python

Sur **Mac ou Linux** :

```bash
python -m venv .venv
source .venv/bin/activate
```

Sur **Windows** :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Une fois l’environnement activé, installer `nbstripout` :

```bash
python -m pip install nbstripout
nbstripout --install
```

Quand le fichier `requirements.txt` aura été ajouté au dépôt, installer aussi les dépendances du projet :

```bash
python -m pip install -r requirements.txt
```

## Workflow Git

1. Vérifier l’état du dépôt avec `git status`. Enregistrer ou mettre de côté toute modification locale avant de changer de branche.
2. Mettre à jour `main` avant de commencer une nouvelle tâche.
3. Créer une branche pour la tâche, en suivant le format :

   `s<sprint>/<membre>-<sujet>`

   Exemples : `s1/A-split`

4. Faire les modifications sur cette branche, jamais directement sur `main`.
5. Avant un commit, vérifier les fichiers modifiés avec `git status`.
6. Ajouter et committer uniquement les fichiers liés à la tâche.
7. Lancer les tests disponibles avant d’envoyer la branche.
8. Envoyer la branche sur GitHub avec `git push`.
9. Ouvrir une pull request vers `main`, en décrivant les changements et les vérifications effectuées.
10. Demander une relecture à un autre membre. Fusionner la pull request après validation.

Exemple pour commencer une nouvelle tâche :

```bash
git status
git switch main
git pull --ff-only
git switch -c s0/C-conventions
```

Remplacer `s0/C-conventions` par le nom correspondant à la tâche. Ne pas créer une branche si elle existe déjà.

Pour enregistrer et publier le travail :

```bash
git add <fichiers-à-committer>
git commit -m "Description courte du changement"
git push -u origin <nom-de-la-branche>
```

## Relecture d’une pull request

Vérifier les points suivants :

- Le code est clair et respecte le contrat d’interfaces.
- Les tests passent.
- Seuls les fichiers concernés par la tâche sont modifiés.
- Aucune donnée ni gros fichier généré n’a été ajouté.

Laisser des commentaires si des changements sont nécessaires, puis approuver la pull request quand elle est prête.

## Fichier d’un autre membre pas encore prêt

Si le travail dépend d’un fichier qui n’est pas encore disponible, écrire si nécessaire une version provisoire (*stub*) qui respecte la même interface. La remplacer quand le fichier définitif est disponible. Ne pas modifier directement le fichier appartenant à l’autre membre.

## Conflits Git

Ne jamais utiliser `--force` pour résoudre un conflit. Examiner les changements en conflit dans VS Code. Si la résolution n’est pas claire, demander au propriétaire du fichier avant de choisir entre les versions.

## Données et notebooks

- Ne pas ajouter à Git les données du concours, notamment les fichiers `.parquet`. Elles sont sous licence CC BY-NC et ne doivent pas être redistribuées.
- Vérifier que les données restent dans `data/`, ignoré par `.gitignore`.
- Retirer les sorties inutiles des notebooks avec `nbstripout` avant de les committer.
- Ne pas ajouter de gros fichiers générés ou de résultats temporaires au dépôt.
- Le fichier de suivi `reports/results.csv` fait partie des livrables du projet et peut être versionné ; il ne doit pas contenir les données brutes du concours.

## Tests

Lancer les tests depuis la racine du dépôt :

```bash
python -m pytest
```

La commande et les dépendances nécessaires seront à confirmer avec le membre responsable de l’environnement du projet.

## Résultats et journal IA

- Ajouter chaque expérience évaluée à `reports/results.csv` avec la fonction `log_result` de C. Le module où cette fonction sera placée reste à confirmer par l’équipe.
- Chacun tient son journal IA dans `journal_ia/journal_<membre>.md`, en utilisant le modèle `journal_ia/_modele.md`.

## Pull request

Chaque pull request doit expliquer :

- ce qui a été ajouté ou modifié ;
- les fichiers concernés ;
- les vérifications ou tests effectués ;
- les éventuels points restant à traiter.

La branche `main` doit rester fonctionnelle. Les changements y sont intégrés par pull request après relecture.