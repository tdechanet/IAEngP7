# Réalisez une analyse de sentiments grâce au Deep Learning

## But du projet

Ce projet à pour objectif la mise en place d'une API donnant accès aux utilisateurs à une IA. Cette dernière prédit les sentiments d'un tweet (positif, négatif) et elle permet aussi aux utilisateurs de données un feedback au cas où l'API se tromperait dans sa prédiction.

## Découpage des dossiers

Nous avons pour ce projet: 
- 4 notebooks ayant servi à l'élaboration de l'IA. 
- Un Pipfile et un requirements listant les différentes bibliothèques nécessaires au bon fonctionnement du projet.
- Un fichier python contenant le code de L'API
- Un dossier de tests qui permet d'assurer la validité de cette dernière
- Un dossier github contenant un yml, celui ci détaillant une pipeline exécuter lors du déploiment
- Un dossier models et deployed_models contenant les modèles que l'API utilise
- Un dossier tokenizer_artifacts contenant les tokenizer, nécessaire à la mise en route du modèle
- Un fichier gitignore classique
- Un fichier gitattributes, qui permet d'utiliser git LFS pour le déploiment des modèles, trop lourd pour github