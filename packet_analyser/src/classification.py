import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_creator import (
    plot_classification,
    plot_tcp_sizes,
    plot_test_results,
)

def split_dataset(data):
    # Chaque capture reste entière dans un seul ensemble
    suffixes = {
        "train": "_01",
        "validation": "_02",
        "test": "_03",
    }

    valid_names = data["capture_id"].str.endswith(tuple(suffixes.values()))

    if not valid_names.all():
        names = sorted(data.loc[~valid_names, "capture_id"].unique())

        raise ValueError(
            "Les captures doivent finir par _01, _02 ou _03 : "
            + ", ".join(names)
        )

    return {
        name: data.loc[data["capture_id"].str.endswith(suffix)].copy()
        for name, suffix in suffixes.items()
    }


def train_and_evaluate(data, output):
    # Le payload définit la classe, mais ne sert pas d'entrée
    report = {
        "task": "TCP avec ou sans payload",
        "class_definition": {
            "0": "TCP sans payload",
            "1": "TCP avec payload",
        },
        "features": ["ip_length"],
        "split_method": (
            "Captures entières : "
            "_01 train, _02 validation, _03 test"
        ),
        "prediction_threshold": 0.5,
        "scope": (
            "Classification TCP au niveau transport, "
            "pas du contrôle applicatif."
        ),
        "evaluation_note": (
            "Les captures doivent être des sessions distinctes. "
            "Le test ne sert pas à choisir les modèles "
            "ou les caractéristiques."
        ),
    }

    partitions = split_dataset(data)
    report["partitions"] = {}

    # Décrire les ensembles
    for name, part in partitions.items():
        report["partitions"][name] = {
            "packets": len(part),
            "fraction": (
                len(part) / len(data) if len(data) else 0
            ),
            "class_counts": {
                str(label): int(part["label"].eq(label).sum())
                for label in [0, 1]
            },
            "captures": sorted(
                part["capture_id"].unique().tolist()
            ),
        }

    # Vérifier la présence des deux classes
    invalid = [
        name
        for name, part in partitions.items()
        if set(part["label"]) != {0, 1}
    ]

    if invalid:
        report["status"] = "insufficient_data"
        report["reason"] = (
            "Deux classes nécessaires dans chaque ensemble. "
            "Ensembles incomplets : "
            + ", ".join(invalid)
            + ". Vérifier les captures et les suffixes "
            "_01, _02 et _03."
        )

    else:
        report["status"] = "completed"

        train = partitions["train"]
        validation = partitions["validation"]
        test = partitions["test"]

        # Comparer deux jeux de caractéristiques
        feature_sets = {"length": ["ip_length"]}

        if "ip_version" in data.columns:
            feature_sets["length_ip_version"] = [
                "ip_length",
                "ip_version",
            ]

        models = {}
        scores = {}

        for feature_name, columns in feature_sets.items():
            estimators = [
                ("linear", LinearRegression()),
                ("ridge", Ridge(alpha=1.0)),
            ]

            for name, estimator in estimators:
                key = f"{name}_{feature_name}"

                # La normalisation est apprise sur train seulement
                model = make_pipeline(
                    StandardScaler(),
                    estimator,
                )

                model.fit(
                    train[columns],
                    train["label"],
                )

                predictions = (
                    model.predict(validation[columns]) >= 0.5
                ).astype(int)

                scores[key] = float(
                    f1_score(
                        validation["label"],
                        predictions,
                        labels=[0, 1],
                        average="macro",
                        zero_division=0,
                    )
                )

                models[key] = (model, columns)

        # Choisir le modèle avec la validation seulement
        best_name = max(scores, key=scores.get)
        best_model, features = models[best_name]

        # Évaluer le modèle choisi sur le test
        prediction_scores = best_model.predict(test[features])
        predictions = (prediction_scores >= 0.5).astype(int)

        report["features"] = features
        report["feature_sets"] = feature_sets
        report["selected_model"] = best_name
        report["validation_macro_f1"] = scores

        report["test"] = classification_report(
            test["label"],
            predictions,
            labels=[0, 1],
            target_names=["sans_payload", "avec_payload"],
            output_dict=True,
            zero_division=0,
        )

        report["selection_note"] = (
            "En cas d’égalité sur validation, retenir le premier "
            "modèle dans l’ordre de comparaison."
        )

        figures_dir = output / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_tcp_sizes(
            train,
            figures_dir / "train_tcp_sizes.pdf",
            "Entraînement — tailles IP par classe TCP",
        )

        # Référence simple : toujours prédire la classe majoritaire du train
        majority = int(train["label"].mode().iloc[0])

        baseline_validation = [majority] * len(validation)
        baseline_test = [majority] * len(test)

        baseline_f1 = float(
            f1_score(
                validation["label"],
                baseline_validation,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )

        report["majority_baseline"] = {
            "class": majority,
            "validation_macro_f1": baseline_f1,
            "test": classification_report(
                test["label"],
                baseline_test,
                labels=[0, 1],
                target_names=["sans_payload", "avec_payload"],
                output_dict=True,
                zero_division=0,
            ),
        }

        # Conserver les prédictions pour examiner les erreurs
        evaluated = test.copy()

        evaluated["prediction"] = predictions
        evaluated["prediction_score"] = prediction_scores
        evaluated["correct"] = evaluated["label"].eq(
            evaluated["prediction"]
        )

        # Résultats séparés pour chaque capture du test
        report["test_by_capture"] = {}

        for capture, selected in evaluated.groupby("capture_id"):
            report["test_by_capture"][capture] = {
                "packets": len(selected),
                "class_counts": {
                    str(label): int(
                        selected["label"].eq(label).sum()
                    )
                    for label in [0, 1]
                },
                "metrics": classification_report(
                    selected["label"],
                    selected["prediction"],
                    labels=[0, 1],
                    target_names=["sans_payload", "avec_payload"],
                    output_dict=True,
                    zero_division=0,
                ),
            }

        # Résumer les erreurs du test
        errors = evaluated.loc[~evaluated["correct"]]

        report["errors"] = {
            "count": len(errors),
            "by_capture": {
                capture: int(count)
                for capture, count
                in errors.groupby("capture_id").size().items()
            },
            "ip_length_summary": {
                name: float(value)
                for name, value
                in errors["ip_length"].describe().dropna().items()
            },
            "examples": (
                errors[[
                    "capture_id",
                    "packet_id",
                    "ip_length",
                    "label",
                    "prediction",
                ]]
                .head(20)
                .to_dict(orient="records")
            ),
        }

        # Figures des tailles et des erreurs
        plot_test_results(
            evaluated,
            figures_dir,
        )

        
        model_names = {
            "linear_length": "Linéaire\nTaille IP",
            "ridge_length": "Ridge\nTaille IP",
            "linear_length_ip_version": "Linéaire\nTaille et version IP",
            "ridge_length_ip_version": "Ridge\nTaille et version IP",
        }

        plotted_scores = {
            model_names[name]: score
            for name, score in scores.items()
        }

        plotted_scores["Référence\nmajoritaire"] = baseline_f1

        matrix = confusion_matrix(
            test["label"],
            predictions,
            labels=[0, 1],
        )

        plot_classification(
            matrix,
            ["Sans charge utile", "Avec charge utile"],
            plotted_scores,
            figures_dir,
        )

    (output / "classification.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return report