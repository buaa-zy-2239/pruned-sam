import torch


class HierarchicalEverythingPredictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, image, prompts):
        return self.model(image)


def build_hierarchical_predictor(model):
    return HierarchicalEverythingPredictor(model)


def compare_hierarchical_vs_standard(model, images):
    results = []
    for img in images:
        result = {
            'hierarchical_time': 0.0,
            'standard_time': 0.0,
            'accuracy_diff': 0.0
        }
        results.append(result)
    return results