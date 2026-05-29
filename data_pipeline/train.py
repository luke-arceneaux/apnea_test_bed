''' 
Train a deep learning model for apnea detection.
Defines a Model() class with train() and test() methods '''
import numpy as np
import time 

from dataloader import ApneaDataloader
from models import CNN, LSTM
from utils import *

import torch
from torch import nn
from torch.optim import Adam

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


np.set_printoptions(suppress=True) 

''' Model definition '''
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
class Model:
    def __init__(self, cfg, data):
        print('Initializing model')
        self.cfg = cfg
        # setup dataloader
        self.train_loader, self.val_loader = ApneaDataloader(cfg, data).get_data()
        # setup model, params
        self.setup_model_and_params()
        self.early_stopping = EarlyStopping(patience=5, delta=0.001)
        
    def setup_model_and_params(self):
        input_size = 1
#         output_size = 2 # binary classification
        output_size = 3 # standard classification w/ less data
    
        # Select model type to use (see src/models.py)
        if self.cfg.model_type == "cnn":
            print('Using model: CNN')
            self.model = CNN(input_size=input_size, \
                             output_size=output_size).double()
        else:
            print('Using model: LSTM')
            self.model = LSTM(input_size=input_size,
                          output_size=output_size).double()
        self.model.to(self.cfg.device)

        # If retrain, load pre-trained model to continue training 
        if self.cfg.retrain:
            print('Retraining, loading params')
            self.model.load_state_dict(torch.load(self.cfg.model_file))

        # Setup loss (criterion) + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optim = Adam(self.model.parameters(), lr=float(self.cfg.learning_rate))

    # Training/Validation 
    def validate(self):
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for seq, label in self.val_loader:
                pred = self.model(seq.to(self.cfg.device))
                val_loss = self.criterion(pred, label.to(self.cfg.device))
                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            return avg_val_loss
        
    def train(self, plot_loss=False):
        print("\n-----------------------Training-----------------------\n")

        self.model.train()

        self.train_losses = []
        self.train_errors = []

        self.val_losses = []
        self.val_errors = []

        for epoch in range(int(self.cfg.epochs)): 


            print(f"Epoch #{epoch}")
            batch_losses = []
            batch_errors = []
            
            ''' -------------Train-------------'''
            for n_batch, (seq, label) in enumerate(self.train_loader):
                    
                self.optim.zero_grad()

                # feed sequence of dim (B, T, C) through model, outputs a prediction
                pred = self.model(seq.to(self.cfg.device))

                # binary prediction
                pred_bin = torch.argmax(pred, dim=1).cpu()

                # compute loss using prediction, label
                loss = self.criterion(pred.to(self.cfg.device), label.to(self.cfg.device))
                batch_losses += [loss.item()]
                # calculate error rate across current batch 
                try:
                    errs = torch.count_nonzero(pred_bin - label)
                except Exception as e:
                    raise Exception('Error during training: ', e)
                err_rate = errs/len(pred_bin)
                batch_errors += [err_rate]
                
                loss.backward()
                self.optim.step() 

                # Log every 10 batches
                if (n_batch) % 10 == 0:
                    print("Epoch: [{}/{}], Batch: {}, Loss: {}, Acc: {}".format(
                        epoch, self.cfg.epochs, n_batch, loss.item(), 1-err_rate))


            epoch_loss = np.mean(batch_losses)
            epoch_errs = np.mean(batch_errors)

            self.train_losses.append(epoch_loss)
            self.train_errors.append(epoch_errs)  
            
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss}")

            self.early_stopping(val_loss)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self):

        self.model.eval()
        test_losses = []
        print("\n-----------------------Testing------------------------\n")
        with torch.no_grad():
            test_acc = []
            for n_batch, (seq, label) in enumerate(self.val_loader):
                start = time.time()
                pred = self.model(seq.to(self.cfg.device))

                # measure inference time
                end = time.time()
                inference_time = end - start 
                # get binary prediction
                pred_bin = torch.argmax(pred, dim=1).to(self.cfg.device)
                label = label.to(self.cfg.device)
                # compute loss using prediction, label
                test_loss = self.criterion(pred, label.to(self.cfg.device))
                test_losses += [test_loss.item()]
                
                # calculate error rate across current batch 
                errs = (torch.count_nonzero(pred_bin - label)).item()
                acc_rate = 1-(errs/len(pred_bin))
                test_acc += [acc_rate]

                print(f"batch #{n_batch} test loss: {test_loss.item()}, acc: {acc_rate}")

            final_test_acc= np.mean(test_acc)
       
            print(f"Overall test accuracy: {final_test_acc}")
        return final_test_acc, inference_time
    

    # def test(self):
    #     self.model.eval()
    #     print("\n-----------------------Testing------------------------\n")

    #     all_preds = []
    #     all_labels = []
    #     all_probs = []
    #     inference_times = []

    #     with torch.no_grad():
    #         for n_batch, (seq, label) in enumerate(self.val_loader):
    #             start = time.time()
    #             pred = self.model(seq.to(self.cfg.device))
    #             end = time.time()
    #             inference_times.append(end - start)

    #             probs = torch.softmax(pred, dim=1)
    #             preds = torch.argmax(probs, dim=1).cpu().numpy()
    #             labels = label.cpu().numpy()

    #             all_preds.extend(preds)
    #             all_labels.extend(labels)
    #             all_probs.extend(probs.cpu().numpy())

    #     # Metrics
    #     all_preds = np.array(all_preds)
    #     all_labels = np.array(all_labels)
    #     all_probs = np.array(all_probs)

    #     acc = np.mean(all_preds == all_labels)
    #     cm = confusion_matrix(all_labels, all_preds)
    #     report = classification_report(all_labels, all_preds, digits=3, output_dict=True)

    #     # Sensitivity (Recall) & Specificity per class
    #     sensitivities = []
    #     specificities = []
    #     for i in range(len(cm)):
    #         TP = cm[i, i]
    #         FN = np.sum(cm[i, :]) - TP
    #         FP = np.sum(cm[:, i]) - TP
    #         TN = np.sum(cm) - (TP + FP + FN)
    #         sensitivity = TP / (TP + FN + 1e-8)
    #         specificity = TN / (TN + FP + 1e-8)
    #         sensitivities.append(sensitivity)
    #         specificities.append(specificity)

    #     try:
    #         roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    #     except Exception:
    #         roc_auc = None

    #     avg_inference_time = np.mean(inference_times)
        
    #     print("\nConfusion Matrix:\n", cm)
    #     print("\nClassification Report:")
    #     print(classification_report(all_labels, all_preds, digits=3))
    #     print(f"Overall Accuracy: {acc:.3f}")
    #     print(f"ROC AUC (macro OVR): {roc_auc:.3f}" if roc_auc else "ROC AUC: N/A")
    #     print(f"Avg Inference Time (s): {avg_inference_time:.5f}")

    #     return {
    #         "accuracy": acc,
    #         "roc_auc": roc_auc,
    #         "sensitivity_per_class": sensitivities,
    #         "specificity_per_class": specificities,
    #         "avg_inference_time": avg_inference_time,
    #         "classification_report": report,
    #         "confusion_matrix": cm.tolist()
    #     }
