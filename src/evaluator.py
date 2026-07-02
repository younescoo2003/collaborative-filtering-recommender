from surprise.model_selection import cross_validate
from surprise import accuracy

def evaluate_cross_validation(model, data, cv=5):
    results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)
    return results

def get_rmse(predictions):
    return accuracy.rmse(predictions, verbose=False)

def get_mae(predictions):
    return accuracy.mae(predictions, verbose=False)

def compare_models(models, data, test_size=0.2):
    """
    Compare multiple models on train/test split.
    models: dict of {name: model_instance}
    """
    from surprise.model_selection import train_test_split
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(trainset)
        pred = model.test(testset)
        rmse = get_rmse(pred)
        mae = get_mae(pred)
        results[name] = {'RMSE': rmse, 'MAE': mae}
        print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return results