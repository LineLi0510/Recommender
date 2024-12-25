import optuna


def initialize_study():
    storage_url = "mysql+mysqlconnector://user:password@optuna-db/optuna"
    study_name = "default_study"

    try:
        # Versuche, eine bestehende Study zu laden
        optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Study '{study_name}' existiert bereits.")
    except KeyError:
        # Falls die Study nicht existiert, erstelle sie
        optuna.create_study(study_name=study_name, storage=storage_url)
        print(f"Study '{study_name}' wurde erstellt.")


if __name__ == "__main__":
    initialize_study()