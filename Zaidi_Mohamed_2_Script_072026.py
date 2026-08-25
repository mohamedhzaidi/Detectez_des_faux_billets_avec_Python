# -*- coding: utf-8 -*-
"""
P12_application.py

Script d'application : charge le modèle entraîné (pipeline StandardScaler + Régression Logistique)
et prédit l'authenticité de nouveaux billets à partir de leurs mesures géométriques.

Deux modes d'utilisation (conformément au cahier des charges) :
  1) --csv_path : un fichier CSV contenant un ou plusieurs billets
  2) les 6 valeurs géométriques d'un seul billet, passées directement en arguments

Justification du modèle (cf. notebook d'analyse, section "Sélection du meilleur modèle") :
la Régression Logistique obtient des performances quasi identiques au Random Forest (99%
d'accuracy) et supérieures à KNN et à KMeans utilisé via ses centroïdes (98,7%), tout en
restant plus simple et plus interprétable — c'est pourquoi elle a été retenue comme modèle final.
"""

import pandas as pd
import joblib
import os
import argparse
import sys

try:
    from IPython.display import display as ipython_display
except ImportError:
    def ipython_display(*args, **kwargs):
        if args:
            print(args[0])

EXPECTED_COLUMNS = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']


def predict_bill_authenticity_from_csv(new_csv_file_path: str, model_path: str = 'modele_detection_faux_billets.joblib') -> pd.DataFrame:
    """
    Charge le modèle entraîné, lit un fichier CSV de nouveaux billets, et prédit leur authenticité.

    Args:
        new_csv_file_path (str): chemin du CSV contenant les nouveaux billets.
                                  Colonnes attendues : diagonal, height_left, height_right,
                                  margin_low, margin_up, length. Colonne 'id' optionnelle.
        model_path (str): chemin du fichier .joblib du modèle sauvegardé.

    Returns:
        pd.DataFrame: colonnes 'id', 'is_genuine_predicted', 'probabilite_vrai'.
                      DataFrame vide en cas d'erreur.
    """
    if not os.path.exists(new_csv_file_path):
        print(f"Erreur: Le fichier CSV '{new_csv_file_path}' est introuvable.")
        return pd.DataFrame()

    try:
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis : {model_path}")
    except FileNotFoundError:
        print(f"Erreur: Le fichier modèle '{model_path}' est introuvable. Assurez-vous qu'il est dans le bon répertoire.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return pd.DataFrame()

    try:
        df_new_bills = pd.read_csv(new_csv_file_path, sep=None, engine="python")
        print(f"Nouvelles données chargées depuis '{new_csv_file_path}'. Aperçu :")
        if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
            ipython_display(df_new_bills.head())
        else:
            print(df_new_bills.head())
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV '{new_csv_file_path}': {e}")
        return pd.DataFrame()

    if 'id' in df_new_bills.columns:
        bill_ids = df_new_bills['id']
        X_new = df_new_bills.drop(columns=['id'])
    else:
        bill_ids = pd.Series(range(len(df_new_bills)), name='id')
        X_new = df_new_bills.copy()

    if not all(col in X_new.columns for col in EXPECTED_COLUMNS):
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in X_new.columns]
        print(f"Erreur: Les colonnes suivantes sont manquantes dans '{new_csv_file_path}': {missing_cols}")
        return pd.DataFrame()

    X_new = X_new[EXPECTED_COLUMNS]
    return _predict_dataframe(X_new, bill_ids, model_path)


def predict_single_bill(diagonal: float, height_left: float, height_right: float,
                         margin_low: float, margin_up: float, length: float,
                         model_path: str = 'modele_detection_faux_billets.joblib') -> pd.DataFrame:
    """
    Prédit l'authenticité d'un seul billet à partir de ses 6 mesures géométriques,
    fournies directement (sans passer par un fichier CSV).

    Returns:
        pd.DataFrame à une ligne : colonnes 'id', 'is_genuine_predicted', 'probabilite_vrai'.
    """
    try:
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis : {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return pd.DataFrame()

    X_new = pd.DataFrame([{
        "diagonal": diagonal, "height_left": height_left, "height_right": height_right,
        "margin_low": margin_low, "margin_up": margin_up, "length": length,
    }])[EXPECTED_COLUMNS]
    bill_ids = pd.Series(["billet_unique"], name="id")
    return _predict_dataframe(X_new, bill_ids, model_path, model=model)


def _predict_dataframe(X_new, bill_ids, model_path, model=None):
    if model is None:
        model = joblib.load(model_path)
    y_pred = model.predict(X_new)
    y_proba = model.predict_proba(X_new)[:, 1]
    return pd.DataFrame({
        'id': bill_ids,
        'is_genuine_predicted': y_pred,
        'probabilite_vrai': y_proba.round(3)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire l'authenticité de billets à partir d'un fichier CSV ou des valeurs d'un seul billet.")
    parser.add_argument('--csv_path', type=str, default=None,
                        help="Chemin d'accès au fichier CSV contenant un ou plusieurs billets.")
    parser.add_argument('--diagonal', type=float, default=None, help="Diagonale du billet (mm) — mode billet unique.")
    parser.add_argument('--height_left', type=float, default=None, help="Hauteur côté gauche (mm) — mode billet unique.")
    parser.add_argument('--height_right', type=float, default=None, help="Hauteur côté droit (mm) — mode billet unique.")
    parser.add_argument('--margin_low', type=float, default=None, help="Marge basse (mm) — mode billet unique.")
    parser.add_argument('--margin_up', type=float, default=None, help="Marge haute (mm) — mode billet unique.")
    parser.add_argument('--length', type=float, default=None, help="Longueur du billet (mm) — mode billet unique.")
    parser.add_argument('--model_path', type=str, default='modele_detection_faux_billets.joblib',
                        help="Chemin d'accès au fichier du modèle sauvegardé.")

    is_interactive_colab = 'ipykernel' in sys.modules or hasattr(sys, 'ps1')

    if is_interactive_colab and len(sys.argv) == 1:
        print("Note: ce bloc est conçu pour une exécution en ligne de commande.")
        print("Dans Colab/Jupyter, appelle directement une des deux fonctions, par exemple :\n")
        print("  predict_bill_authenticity_from_csv('billets_nouveaux_test.csv', 'modele_detection_faux_billets.joblib')")
        print("  predict_single_bill(171.8, 104.2, 104.1, 4.3, 3.2, 113.1)")
    else:
        args = parser.parse_args()
        single_bill_args = [args.diagonal, args.height_left, args.height_right, args.margin_low, args.margin_up, args.length]

        if args.csv_path:
            predictions_df = predict_bill_authenticity_from_csv(args.csv_path, args.model_path)
        elif all(v is not None for v in single_bill_args):
            predictions_df = predict_single_bill(*single_bill_args, model_path=args.model_path)
        else:
            print("Erreur : fournissez soit --csv_path, soit les 6 valeurs géométriques du billet")
            print("(--diagonal --height_left --height_right --margin_low --margin_up --length).")
            print("\nExemples :")
            print("  python P12_application.py --csv_path billets_production.csv")
            print("  python P12_application.py --diagonal 171.8 --height_left 104.2 --height_right 104.1 --margin_low 4.3 --margin_up 3.2 --length 113.1")
            sys.exit(1)

        if not predictions_df.empty:
            print("\nPrédictions terminées. Résultats :")
            print(predictions_df)
            output_file = 'predictions.csv'
            predictions_df.to_csv(output_file, index=False)
            print(f"Les prédictions ont été sauvegardées dans '{output_file}'.")
        else:
            print("Aucune prédiction n'a pu être générée.")