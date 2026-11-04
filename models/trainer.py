"""Unified training loop (used by all three training scripts)."""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, params: dict, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.params = params
        self.train_ds = train_dataset
        self.val_ds = val_dataset

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda')  # <- 2.x API, not 1.x
        self.best_state = None
        self.best_val_loss = float('inf')

    def _train_epoch(self, loader):
        self.model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
        return total / len(loader)

    @torch.inference_mode()
    def _validate(self, loader):
        self.model.eval()
        total = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            total += self.criterion(self.model(x), y).item()
        return total / len(loader)

    def train(self, epochs=30, patience=5):
        train_loader = DataLoader(self.train_ds, batch_size=self.params['batch_size'],
                                  shuffle=True, num_workers=self.params.get('num_workers', 2),
                                  pin_memory=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.params['batch_size'],
                                num_workers=self.params.get('num_workers', 2), pin_memory=True)

        patience_counter = 0
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)
            print(f"Epoch {epoch}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.best_state, self.best_val_loss
