import os
import torch


class Inference:
    def __init__(self, GPU, HYPERPARAMETERS, MODEL_PATH):
        self._exp_config = HYPERPARAMETERS
        self._model_path = MODEL_PATH

        gpu_index = GPU

        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{gpu_index}")
        else:
            self._device = torch.device("cpu")

        if self._device.type == "cpu":
            model = torch.load(
                "{}/fullmodel.pt".format(self._model_path),
                map_location=torch.device("cpu"),
            )["model"]
        elif self._device.type == "cuda":
            model = torch.load(
                "{}/fullmodel.pt".format(self._model_path),
                map_location=torch.device(f"cuda:{gpu_index}"),
            )
            if isinstance(model, dict) and "model" in model:
                model = model["model"]
            else:
                print(f"model is {type(model)}")

            print("loaded model on gpu")

        self.model = model.to(self._device)

    def _run_an_eval_epoch(self, model, data_loader):
        """Utility function for running an evaluation (validation/test) epoch

        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test

        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        """
        all_preds = []
        all_IDs = []

        model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                IDs, bg = batch_data
                logits = self._predict(model, bg)

                all_IDs.extend(IDs)
                all_preds.append(logits.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_preds = torch.sigmoid(all_preds).numpy().ravel()

        return [*zip(all_IDs, all_preds)]

    def _predict(self, model, bg):
        bg = bg.to(self._device)
        if self._exp_config["model"] in ["GCN", "GAT"]:
            node_feats = bg.ndata.pop("h").to(self._device)
            preds = model(bg, node_feats)
            node_feats.detach().cpu()
            del node_feats
        else:
            node_feats = bg.ndata.pop("h").to(self._device)
            edge_feats = bg.edata.pop("e").to(self._device)
            preds = model(bg, node_feats, edge_feats)
            node_feats.detach().cpu()
            edge_feats.detach().cpu()
            del edge_feats
            del node_feats

        bg.to("cpu")
        del bg

        return preds

    def predict(self, dataloader):
        preds = self._run_an_eval_epoch(self.model, dataloader)

        return preds
