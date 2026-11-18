# retail-media-uplift

[![tests](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml/badge.svg)](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml)

Measure what advertising causes, not what it is correlated with, on Criteo's randomized ad experiment (13.9 million users).

**Dashboard:** https://retail-media-uplift-plus.streamlit.app/

## Architecture

### Data and quality gate

```mermaid
flowchart LR
    HF[Hugging Face<br/>criteo/criteo-uplift<br/>13,979,592 users<br/>12 features, treatment, conversion]

    subgraph DEV[Local machine]
        LP[scripts/launch_processing.py]
        KEYS[RSA key pair<br/>kept outside the repo]
    end

    subgraph AWS[AWS eu-west-3]
        PROC[SageMaker processing job<br/>ml.m5.xlarge, scikit-learn 1.4-2<br/>prepare_data.py]
        subgraph S3[S3 bucket sagemaker-retail-media-uplift-3769]
            RAW[raw/<br/>criteo_uplift_full.parquet<br/>218 MB]
        end
        SFROLE[IAM role snowflake-retail-media-role<br/>read only on this bucket<br/>trusts one Snowflake account<br/>with an external ID]
        LAM[Lambda retail-media-quality-gate<br/>Python 3.12<br/>quality_gate.py]
        LROLE[IAM role<br/>logs only]
    end

    subgraph SF[Snowflake]
        INT[storage integration<br/>and raw_stage]
        TAB[(experiment_data<br/>13,979,592 rows)]
        WH[warehouse RETAIL_MEDIA_WH<br/>X-Small, stops after 60 s]
        SVC[lambda_service_user<br/>read only role<br/>key pair login]
        SQL[randomization check<br/>standardized mean difference<br/>of each feature below 0.05]
    end

    LP -->|start| PROC
    HF --> PROC
    PROC --> RAW
    RAW --> SFROLE
    SFROLE --> INT
    INT -->|COPY INTO| TAB
    KEYS -.->|private key, encrypted setting| LAM
    KEYS -.->|public key| SVC
    LROLE -.-> LAM
    LAM --> SVC
    SVC --> WH
    WH --> SQL
    TAB --> SQL
    SQL -->|passed true or false| LAM
```

### Pipeline, tuning and training

```mermaid
flowchart TB
    START([Start execution])

    subgraph SFN[Step Functions retail-media-uplift-pipeline]
        GATE[RunQualityGate<br/>Lambda to Snowflake]
        CHK{CheckResult}
        FAIL[NotifyFail<br/>SNS email<br/>training not started]
        TUNE[Tune<br/>SageMaker automatic model tuning<br/>12 Bayesian trials, 3 in parallel<br/>goal: validation Qini]
        BEST[GetBestSettings<br/>reads the best trial]
        CODE[AddCodeSettings<br/>location of the training code]
        TRAIN[Train<br/>SageMaker training job<br/>ml.m5.2xlarge]
        DONE[NotifyDone<br/>SNS email with the reports location]
    end

    subgraph TRIAL[Each tuning trial]
        T1[70 percent train, 30 percent test]
        T2[train split again<br/>80 percent fit, 20 percent validation]
        T3[S-learner with LightGBM<br/>on the fit part]
        T4[prints valid_qini<br/>test set never used]
    end

    subgraph FINAL[Final training job]
        F1[randomization check and average effect<br/>with 95 percent CI]
        F2[4 uplift models<br/>T-learner logistic, T-learner LightGBM<br/>S-learner LightGBM, class transformation]
        F3[best model chosen<br/>on validation Qini]
        F4[propensity model<br/>control group only]
        F5[Qini, AUUC, uplift at 10 percent<br/>on 4,193,878 test users]
        F6[profit per budget, 2 to 100 percent<br/>paired bootstrap, 200 resamples]
    end

    subgraph OUT[Results]
        REP[(S3 models/.../model.tar.gz<br/>summary.json, curves,<br/>test scores sample of 1M users)]
        GIT[GitHub, reports folder]
        APP[Streamlit dashboard<br/>profit recomputed live<br/>for any cost and value]
        CW[CloudWatch logs]
    end

    PROLE[IAM role retail-media-pipeline-role<br/>this Lambda, this topic,<br/>uplift jobs only]

    START --> GATE --> CHK
    CHK -->|failed| FAIL
    CHK -->|passed| TUNE --> BEST --> CODE --> TRAIN --> DONE
    TUNE -.-> T1 --> T2 --> T3 --> T4
    TRAIN -.-> F1 --> F2 --> F3 --> F5
    F2 --> F4 --> F6
    F3 --> F6
    TRAIN --> REP --> GIT --> APP
    TUNE -.-> CW
    TRAIN -.-> CW
    PROLE -.-> SFN
```

## Data

- Criteo uplift dataset: a randomized experiment, users randomly kept out of advertising
- 13,979,592 users, 40,774 converters (0.29 %), 85 % treated, 15 % control
- 12 anonymous features
- Split: 70 % train, 30 % test (4,193,878 users); 20 % of train kept for validation

## Method

| Step | Approach |
|---|---|
| Quality gate | Standardized mean difference of each feature between groups, below 0.05 |
| Average effect | Difference in conversion rates, Welch 95 % CI |
| Uplift models | T-learner (logistic, LightGBM), S-learner (LightGBM), class transformation |
| Model choice | Best validation Qini; test set used once |
| Evaluation | Qini coefficient, AUUC, uplift at 10 % |
| Business layer | Profit per budget: 40 $ per extra conversion, minus 0.10 $ per ad shown |
| Policy comparison | Uplift targeting vs propensity targeting, same users, paired bootstrap 95 % CI |

## Results

- Advertising effect: +0.00115 conversion (95 % CI +0.00108 to +0.00122), +59.4 % relative lift

| Model | Qini coefficient | AUUC | Uplift at 10 % |
|---|---|---|---|
| S-learner, LightGBM | **+0.000389** | 3,685.2 | 0.00807 |
| T-learner, logistic | +0.000379 | 3,641.5 | 0.00805 |
| T-learner, LightGBM | +0.000314 | 3,371.6 | 0.00758 |
| Class transformation | −0.000153 | 1,412.3 | 0.00279 |

## Findings

- Three of the four models rank users better than random targeting
- Uplift targeting never earns significantly more than propensity targeting, at any budget
- Final run: propensity targeting earns significantly more from 6 % to 35 % of the audience (6 % budget: −8,389 $, 95 % CI −12,834 $ to −3,801 $)
- Fragile: a second tuning run with the same Qini found a significant gap only at 6 %
- Reason: the two lists target mostly the same people (82 % of the top 6 %, 81 % of the top 20 %, rank correlation 0.85)
- Ads lift conversion by about 0.45 points in the top 20 % of either list, four times the average: the users ads move most are mostly users already likely to buy
- Earlier 3M-row sample: propensity looked significantly better from 35 % to 84 %; with 4.6 times more converters, this gap disappears
- SageMaker tuning: validation Qini 0.000376 to 0.000393; small trees and slow learning win; a fast learning rate (0.2) makes the S-learner ignore the treatment column

## Limitations

- One dataset, one advertiser; 12 anonymous features
- Profit depends on fixed assumptions: 0.10 $ per ad shown, 40 $ per extra conversion (both editable in the dashboard)
- The dashboard uses a random sample of 1M test users; all results above use the 4.19M
- The Snowflake trial ended in September 2026: if the account is suspended, the pipeline stops at the quality gate, before training

## Cost

- Processing job: about 0.05 $
- Pipeline run (quality gate, 12 tuning trials, final training): about 0.70 $
- Snowflake: under 0.1 credit
- Nothing runs between pipeline runs

## Stack

- AWS: S3, SageMaker AI (processing, automatic model tuning, training), Lambda, Step Functions, SNS, IAM
- Snowflake: storage integration, SQL quality check, key pair login
- Python: pandas, scikit-learn, LightGBM, SciPy, Streamlit, pytest, GitHub Actions

## Structure

```
src/              data prep, uplift learners, Qini and AUUC, profit simulation, training
sagemaker/        code that runs on SageMaker machines
scripts/          start SageMaker jobs from a computer
snowflake/        storage integration, table, quality check SQL
lambda/           quality gate
step-functions/   the pipeline
aws/              IAM policies
notebooks/        exploration, uplift models, budget allocation (3M-row sample)
app/              dashboard
tests/            pytest
reports/          results of the pipeline run
```

## Run

Local sample:

```bash
pip install -r requirements.txt
python -m src.download --rows 3000000
python -m src.train
pytest tests/
streamlit run app/streamlit_app.py
```

Full dataset on SageMaker:

```bash
python scripts/launch_processing.py
python scripts/launch_tuning.py
python scripts/launch_training.py
```

---

# retail-media-uplift

[![tests](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml/badge.svg)](https://github.com/FatineKa/retail-media-uplift/actions/workflows/tests.yml)

Mesurer ce que la publicité provoque, et non ce à quoi elle est corrélée, sur l'expérience publicitaire randomisée de Criteo (13,9 millions d'utilisateurs).

**Tableau de bord :** https://retail-media-uplift-plus.streamlit.app/

## Architecture

### Données et contrôle qualité

```mermaid
flowchart LR
    HF[Hugging Face<br/>criteo/criteo-uplift<br/>13 979 592 utilisateurs<br/>12 variables, traitement, conversion]

    subgraph DEV[Ordinateur local]
        LP[scripts/launch_processing.py]
        KEYS[Paire de clés RSA<br/>gardée hors du dépôt]
    end

    subgraph AWS[AWS eu-west-3]
        PROC[SageMaker processing job<br/>ml.m5.xlarge, scikit-learn 1.4-2<br/>prepare_data.py]
        subgraph S3[Compartiment S3 sagemaker-retail-media-uplift-3769]
            RAW[raw/<br/>criteo_uplift_full.parquet<br/>218 Mo]
        end
        SFROLE[Rôle IAM snowflake-retail-media-role<br/>lecture seule sur ce compartiment<br/>fait confiance à un seul compte Snowflake<br/>avec un identifiant externe]
        LAM[Lambda retail-media-quality-gate<br/>Python 3.12<br/>quality_gate.py]
        LROLE[Rôle IAM<br/>logs uniquement]
    end

    subgraph SF[Snowflake]
        INT[intégration de stockage<br/>et raw_stage]
        TAB[(experiment_data<br/>13 979 592 lignes)]
        WH[entrepôt RETAIL_MEDIA_WH<br/>X-Small, s'arrête après 60 s]
        SVC[lambda_service_user<br/>rôle en lecture seule<br/>connexion par clés]
        SQL[contrôle de randomisation<br/>différence moyenne standardisée<br/>de chaque variable sous 0,05]
    end

    LP -->|lance| PROC
    HF --> PROC
    PROC --> RAW
    RAW --> SFROLE
    SFROLE --> INT
    INT -->|COPY INTO| TAB
    KEYS -.->|clé privée, réglage chiffré| LAM
    KEYS -.->|clé publique| SVC
    LROLE -.-> LAM
    LAM --> SVC
    SVC --> WH
    WH --> SQL
    TAB --> SQL
    SQL -->|réussi vrai ou faux| LAM
```

### Pipeline, réglage et entraînement

```mermaid
flowchart TB
    START([Lancement])

    subgraph SFN[Step Functions retail-media-uplift-pipeline]
        GATE[RunQualityGate<br/>Lambda vers Snowflake]
        CHK{CheckResult}
        FAIL[NotifyFail<br/>e-mail SNS<br/>entraînement non lancé]
        TUNE[Tune<br/>réglage automatique SageMaker<br/>12 essais bayésiens, 3 en parallèle<br/>objectif : Qini de validation]
        BEST[GetBestSettings<br/>lit le meilleur essai]
        CODE[AddCodeSettings<br/>emplacement du code d'entraînement]
        TRAIN[Train<br/>training job SageMaker<br/>ml.m5.2xlarge]
        DONE[NotifyDone<br/>e-mail SNS avec l'emplacement des résultats]
    end

    subgraph TRIAL[Chaque essai de réglage]
        T1[70 pour cent entraînement, 30 pour cent test]
        T2[entraînement redécoupé<br/>80 pour cent apprentissage, 20 pour cent validation]
        T3[S-learner avec LightGBM<br/>sur la partie apprentissage]
        T4[affiche valid_qini<br/>le test n'est jamais utilisé]
    end

    subgraph FINAL[Entraînement final]
        F1[contrôle de randomisation et effet moyen<br/>avec IC à 95 pour cent]
        F2[4 modèles d'uplift<br/>T-learner logistique, T-learner LightGBM<br/>S-learner LightGBM, transformation de classe]
        F3[meilleur modèle choisi<br/>sur le Qini de validation]
        F4[modèle de propension<br/>groupe témoin uniquement]
        F5[Qini, AUUC, uplift à 10 pour cent<br/>sur 4 193 878 utilisateurs de test]
        F6[profit par budget, de 2 à 100 pour cent<br/>bootstrap apparié, 200 tirages]
    end

    subgraph OUT[Résultats]
        REP[(S3 models/.../model.tar.gz<br/>summary.json, courbes,<br/>échantillon de 1 M de scores de test)]
        GIT[GitHub, reports folder]
        APP[Tableau de bord Streamlit<br/>profit recalculé en direct<br/>pour tout coût et toute valeur]
        CW[Logs CloudWatch]
    end

    PROLE[Rôle IAM retail-media-pipeline-role<br/>cette Lambda, ce sujet SNS,<br/>jobs uplift uniquement]

    START --> GATE --> CHK
    CHK -->|échec| FAIL
    CHK -->|réussi| TUNE --> BEST --> CODE --> TRAIN --> DONE
    TUNE -.-> T1 --> T2 --> T3 --> T4
    TRAIN -.-> F1 --> F2 --> F3 --> F5
    F2 --> F4 --> F6
    F3 --> F6
    TRAIN --> REP --> GIT --> APP
    TUNE -.-> CW
    TRAIN -.-> CW
    PROLE -.-> SFN
```

## Données

- Jeu de données uplift de Criteo : une expérience randomisée, des utilisateurs tenus à l'écart de la publicité au hasard
- 13 979 592 utilisateurs, 40 774 acheteurs (0,29 %), 85 % traités, 15 % témoins
- 12 variables anonymes
- Découpage : 70 % entraînement, 30 % test (4 193 878 utilisateurs) ; 20 % de l'entraînement gardés pour la validation

## Méthode

| Étape | Approche |
|---|---|
| Contrôle qualité | Différence moyenne standardisée de chaque variable entre les groupes, sous 0,05 |
| Effet moyen | Différence des taux de conversion, IC à 95 % de Welch |
| Modèles d'uplift | T-learner (logistique, LightGBM), S-learner (LightGBM), transformation de classe |
| Choix du modèle | Meilleur Qini de validation ; jeu de test utilisé une seule fois |
| Évaluation | Coefficient de Qini, AUUC, uplift à 10 % |
| Couche métier | Profit par budget : 40 $ par conversion supplémentaire, moins 0,10 $ par publicité affichée |
| Comparaison des stratégies | Ciblage par uplift contre ciblage par propension, mêmes utilisateurs, IC à 95 % par bootstrap apparié |

## Résultats

- Effet de la publicité : +0,00115 de conversion (IC à 95 % de +0,00108 à +0,00122), +59,4 % en relatif

| Modèle | Coefficient de Qini | AUUC | Uplift à 10 % |
|---|---|---|---|
| S-learner, LightGBM | **+0,000389** | 3 685,2 | 0,00807 |
| T-learner, logistique | +0,000379 | 3 641,5 | 0,00805 |
| T-learner, LightGBM | +0,000314 | 3 371,6 | 0,00758 |
| Transformation de classe | −0,000153 | 1 412,3 | 0,00279 |

## Constats

- Trois des quatre modèles classent les utilisateurs mieux qu'un ciblage au hasard
- Le ciblage par uplift ne rapporte jamais significativement plus que le ciblage par propension, quel que soit le budget
- Exécution finale : le ciblage par propension rapporte significativement plus de 6 % à 35 % de l'audience (budget de 6 % : −8 389 $, IC à 95 % de −12 834 $ à −3 801 $)
- Fragile : une deuxième exécution du réglage, avec le même Qini, ne trouve un écart significatif qu'à 6 %
- Raison : les deux listes ciblent surtout les mêmes personnes (82 % du top 6 %, 81 % du top 20 %, corrélation de rang 0,85)
- La publicité augmente la conversion d'environ 0,45 point dans le top 20 % des deux listes, quatre fois la moyenne : les utilisateurs que la publicité fait le plus changer sont surtout ceux qui allaient déjà probablement acheter
- Ancien échantillon de 3 M de lignes : la propension semblait significativement meilleure de 35 % à 84 % ; avec 4,6 fois plus d'acheteurs, cet écart disparaît
- Réglage SageMaker : Qini de validation de 0,000376 à 0,000393 ; les petits arbres et l'apprentissage lent gagnent ; un taux d'apprentissage élevé (0,2) fait ignorer la colonne de traitement au S-learner

## Limites

- Un seul jeu de données, un seul annonceur ; 12 variables anonymes
- Le profit dépend d'hypothèses fixes : 0,10 $ par publicité affichée, 40 $ par conversion supplémentaire (modifiables dans le tableau de bord)
- Le tableau de bord utilise un échantillon aléatoire de 1 M d'utilisateurs de test ; tous les résultats ci-dessus utilisent les 4,19 M
- L'essai Snowflake s'est terminé en septembre 2026 : si le compte est suspendu, le pipeline s'arrête au contrôle qualité, avant l'entraînement

## Coût

- Processing job : environ 0,05 $
- Exécution du pipeline (contrôle qualité, 12 essais de réglage, entraînement final) : environ 0,70 $
- Snowflake : moins de 0,1 crédit
- Rien ne tourne entre deux exécutions

## Outils

- AWS : S3, SageMaker AI (processing, réglage automatique, entraînement), Lambda, Step Functions, SNS, IAM
- Snowflake : intégration de stockage, contrôle qualité en SQL, connexion par clés
- Python : pandas, scikit-learn, LightGBM, SciPy, Streamlit, pytest, GitHub Actions

## Structure

```
src/              préparation des données, modèles d'uplift, Qini et AUUC, simulation de profit, entraînement
sagemaker/        code exécuté sur les machines SageMaker
scripts/          lancement des jobs SageMaker depuis un ordinateur
snowflake/        intégration de stockage, table, contrôle qualité en SQL
lambda/           contrôle qualité
step-functions/   le pipeline
aws/              politiques IAM
notebooks/        exploration, modèles d'uplift, allocation du budget (échantillon de 3 M de lignes)
app/              tableau de bord
tests/            pytest
reports/          résultats de l'exécution du pipeline
```

## Lancer

Échantillon local :

```bash
pip install -r requirements.txt
python -m src.download --rows 3000000
python -m src.train
pytest tests/
streamlit run app/streamlit_app.py
```

Jeu complet sur SageMaker :

```bash
python scripts/launch_processing.py
python scripts/launch_tuning.py
python scripts/launch_training.py
```
