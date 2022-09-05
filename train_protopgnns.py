import os
import hydra
import torch
import shutil
import warnings
from torch.optim import Adam
from omegaconf import OmegaConf
from utils import check_dir
from protop_gnnNets import *
from dataset import get_dataset, get_dataloader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import logging
import random

log = logging.getLogger(__name__)
def print(*args):
    log.info(*args)
    
class TrainModel(object):
    def __init__(self,
                 model,
                 dataset,
                 device,
                 graph_classification=True,
                 save_dir=None,
                 save_name='model',
                 ** kwargs):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.loader = None
        self.device = device
        self.graph_classification = graph_classification
        self.node_classification = not graph_classification

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        if self.graph_classification:
            dataloader_params = kwargs.get('dataloader_params')
            self.loader = get_dataloader(dataset, **dataloader_params)

    def __emb_loss__(self, data, out, mask=None):
        logits, embs, l2s = out
        labels = data.y
        
        if self.graph_classification:
            L_id = F.cross_entropy(logits, labels.long())
        else:
            L_id = F.cross_entropy(logits[mask], labels[mask].long())
        L_compactness = 0
        L_separation = 0
        if self.graph_classification:
            n_graphs = logits.shape[0]
            for g in range(n_graphs):
                nodes = data.batch == g
                label = labels[g].int().item()
                L_compactness += torch.min(l2s[nodes,label*self.model.num_basis_per_class: (label+1)*self.model.num_basis_per_class])
                if label != 0:
                    L_separation += -torch.min(l2s[nodes,:label*self.model.num_basis_per_class])
                if label != self.model.output_dim -1:
                    L_separation += -torch.min(l2s[nodes,(label+1)*self.model.num_basis_per_class:])

            L_compactness = L_compactness/n_graphs
            L_separation = L_separation/n_graphs
            L_cs = (0.8*L_compactness + 0.08*L_separation)
        else:
            labels = data.y[mask]
            logits = logits[mask,:]
            embs = embs[mask,:]
            l2s = l2s[mask,:]
            n_nodes = logits.shape[0]
            for node in range(n_nodes):
                label = labels[node].int().item()
                L_compactness += torch.min(l2s[node,label*self.model.num_basis_per_class: (label+1)*self.model.num_basis_per_class])
                if label != 0:
                    L_separation += -torch.min(l2s[node,:label*self.model.num_basis_per_class])
                if label != self.model.output_dim -1:
                    L_separation += -torch.min(l2s[node,(label+1)*self.model.num_basis_per_class:])
            L_compactness = L_compactness/n_nodes
            L_separation = L_separation/n_nodes
            L_cs = (0.8*L_compactness + 0.08*L_separation)

        
        
            
        L_total = L_id + L_cs
        return L_total
    
    def __class_loss__(self, data, out, mask=None):
        logits, embs, l2s = out
        labels = data.y
        if self.graph_classification:
            L_id = F.cross_entropy(logits, labels.long())
        else:
            L_id = F.cross_entropy(logits[mask], labels[mask].long())
        L_g = 0
        for c in range(self.model.output_dim):
            L_g += self.model.classifier_weights[:c*self.model.num_basis_per_class, c].abs().sum()
            L_g += self.model.classifier_weights[(c+1)*self.model.num_basis_per_class:, c].abs().sum()
        return L_id + 1e-4*L_g
    
    def __loss__(self, data, out, mask=None):
        if self.model.classifier_weights.requires_grad:
            return self.__class_loss__(data, out, mask)
        return self.__emb_loss__(data, out, mask)

    def _train_batch(self, data, labels):
        out = self.model(data=data)
        mask = None
        if not self.graph_classification:
            mask = data.train_mask
        loss = self.__loss__(data, out, mask=mask)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        self.model.eval()
        out = self.model(data=data)
        logits, _, _ = out
        if self.graph_classification:
            loss = self.__loss__(data, out)
        else:
            mask = kwargs.get('mask')
            if mask is None:
                warnings.warn("The node mask is None")
                mask = torch.ones(labels.shape[0])
            loss = self.__loss__(data, out, mask=mask)
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, accs = [], []
            for batch in self.loader['eval']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.data.to(self.device)
            eval_loss, preds = self._eval_batch(data, data.y, mask=data.val_mask)
            eval_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        return eval_loss, eval_acc

    def test(self):
        if self.save:
            state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
        self.model.eval()
        if self.graph_classification:
            losses, preds, accs = [], [], []
            for batch in self.loader['test']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
        else:
            data = self.dataset.data.to(self.device)
            test_loss, preds = self._eval_batch(data, data.y, mask=data.test_mask)
            test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")
        return test_loss, test_acc, preds

    def train(self, train_params=None, optimizer1_params=None, optimizer2_params=None):
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']
            
        if optimizer1_params is None:
            self.optimizer1 = Adam(self.model.parameters())
        else:
            self.optimizer1 = Adam(self.model.parameters(), **optimizer1_params)    

        if optimizer2_params is None:
            self.optimizer2 = Adam(self.model.parameters())
        else:
            self.optimizer2 = Adam(self.model.parameters(), **optimizer2_params)        


        if milestones is not None and gamma is not None:
            lr_schedule1 = MultiStepLR(self.optimizer1,
                                      milestones=milestones,
                                      gamma=gamma)
            lr_schedule2 = MultiStepLR(self.optimizer2,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule1 = None
            lr_schedule2 = None
            
        N1 = 50
        N2 = 10

        use_pretrained = train_params.get('use_pretrained')
        if use_pretrained:
            N1 = 10
            pretrained_dict = torch.load(train_params['pretrained_dir'])['net']
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if 'conv' in k}
            
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict) 
            self.model.load_state_dict(model_dict)

        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.classifier_weights.requires_grad = False
            if use_pretrained and epoch < 5:
                self.model.convs.requires_grad = False
            self.optimizer = self.optimizer1
            for t in range(N1):
                is_best = False
                self.model.train()
                if self.graph_classification:
                    losses = []
                    for batch in self.loader['train']:
                        batch = batch.to(self.device)
                        loss = self._train_batch(batch, batch.y)
                        losses.append(loss)
                    train_loss = torch.FloatTensor(losses).mean().item()

                else:
                    data = self.dataset.data.to(self.device)
                    train_loss = self._train_batch(data, data.y)

                eval_loss, eval_acc = self.eval()
                with torch.no_grad():
                    self.model.basis_concepts.div_(torch.linalg.norm(self.model.basis_concepts, dim=0))
                print(f'Emb Epoch:{epoch}-{t}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}')
            
            
            with torch.no_grad():
                new_basis_concepts = torch.ones_like(self.model.basis_concepts).to(self.device)
                num_basis = self.model.output_dim*self.model.num_basis_per_class
                best_l2s = torch.inf*torch.ones(num_basis).to(self.device)
                if self.graph_classification:
                    for batch in self.loader['train']:
                        batch = batch.to(self.device)
                        out = self.model(data=batch)
                        node_labels = batch.y[batch.batch]
                        logits, embs, l2s = out
                        for i in range(num_basis):
                            c = i // self.model.num_basis_per_class
                            if l2s[node_labels==c, i].numel() == 0:
                                continue
                            best_node = l2s[node_labels==c, i].argmin()
                            best_l2 = l2s[node_labels==c, i][best_node]
                            if best_l2 < best_l2s[i]:
                                best_graph = batch.batch[node_labels==c][best_node]
                                self.model.basis_data[i] = batch[best_graph]
                                assert(batch[best_graph].y == c)
                                best_l2s[i] = best_l2
                                best_emb = embs[node_labels==c][best_node]
                                new_basis_concepts[:, i] = best_emb
                else:
                    data = self.dataset.data.to(self.device)
                    out = self.model(data=data)
                    node_labels = data.y[data.train_mask]
                    logits, embs, l2s = out
                    logits = logits[data.train_mask]
                    embs = embs[data.train_mask]
                    l2s = l2s[data.train_mask]
                    for i in range(num_basis):
                        c = i // self.model.num_basis_per_class
                        if l2s[node_labels==c, i].numel() == 0:
                            continue
                        best_node = l2s[node_labels==c, i].argmin()
                        best_l2 = l2s[node_labels==c, i][best_node]
                        if best_l2 < best_l2s[i]:
                            self.model.basis_data[i] = best_node
                            best_l2s[i] = best_l2
                            best_emb = embs[node_labels==c][best_node]
                            new_basis_concepts[:, i] = best_emb
                self.model.basis_concepts.copy_(new_basis_concepts)    
            
            for p in self.model.parameters():
                p.requires_grad = False
            
            self.model.classifier_weights.requires_grad = True
            self.optimizer = self.optimizer2
            for t in range(N2):
                is_best = False
                self.model.train()
                if self.graph_classification:
                    losses = []
                    for batch in self.loader['train']:
                        batch = batch.to(self.device)
                        loss = self._train_batch(batch, batch.y)
                        losses.append(loss)
                    train_loss = torch.FloatTensor(losses).mean().item()
                else:
                    data = self.dataset.data.to(self.device)
                    train_loss = self._train_batch(data, data.y)
                
                eval_loss, eval_acc = self.eval()
                if best_eval_acc < eval_acc:
                    is_best = True
                    best_eval_acc = eval_acc
                recording = {'epoch': epoch, 'is_best': str(is_best)}
                print(f'Cla Epoch:{epoch}-{t}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}')
                if self.save:
                    self.save_model(is_best, recording=recording)
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break
                    
                    
            if lr_schedule1:
                lr_schedule1.step()  
            if lr_schedule2:
                lr_schedule1.step()            

    def save_model(self, is_best=False, recording=None):
        self.model.to('cpu')
        state = {'net': self.model.state_dict(), 'basis_data': self.model.basis_data}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print('saving best...')
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        saved = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))
        state_dict = saved['net']
        self.model.basis_data = saved['basis_data']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
@hydra.main(config_path="protop_config", config_name="config")
def main(config):
    config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'protop_checkpoints', str(config.seed))
    config.models.param = config.models.param[config.datasets.dataset_name]
    print(OmegaConf.to_yaml(config))

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    import numpy as np
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    dataset = get_dataset(dataset_root=config.datasets.dataset_root,
                          dataset_name=config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.seed}

    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    
    pretrained_dir = None
    use_pretrained = config.use_pretrained
    pretrained_dataset = config.pretrained_dataset
    if use_pretrained:
        if pretrained_dataset is None:
            pretrained_dataset = config.datasets.dataset_name
        pretrained_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(config.seed), pretrained_dataset,  f'{config.models.gnn_name}_{len(config.models.param.gnn_latent_dim)}l_best.pth')
        config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'protop_checkpoints_pretrained', str(config.seed))
    
    save_dir_name = config.datasets.dataset_name
    if pretrained_dataset != save_dir_name:
        save_dir_name = f"{save_dir_name}_from_{pretrained_dataset}"

    print(model)
    train_params = {'num_epochs': config.models.param.num_epochs,
                    'num_early_stop': config.models.param.num_early_stop,
                    'milestones': config.models.param.milestones,
                    'gamma': config.models.param.gamma,
                    'use_pretrained':use_pretrained,
                    'pretrained_dir':pretrained_dir}
    optimizer1_params = {'lr': config.models.param.learning_rate1,
                        'weight_decay': config.models.param.weight_decay1}
    optimizer2_params = {'lr': config.models.param.learning_rate2,
                        'weight_decay': config.models.param.weight_decay2}

    if config.models.param.graph_classification:
        trainer = TrainModel(model=model,
                             dataset=dataset,
                             device=device,
                             graph_classification=config.models.param.graph_classification,
                             save_dir=os.path.join(config.models.gnn_saving_dir,
                                                   save_dir_name),
                             save_name=f'{config.models.gnn_name}_{config.models.param.num_basis_per_class}_{len(config.models.param.gnn_latent_dim)}l',
                             dataloader_params=dataloader_params)
    else:
        trainer = TrainModel(model=model,
                             dataset=dataset,
                             device=device,
                             graph_classification=config.models.param.graph_classification,
                             save_dir=os.path.join(config.models.gnn_saving_dir,
                                                   save_dir_name),
                             save_name=f'{config.models.gnn_name}_{config.models.param.num_basis_per_class}_{len(config.models.param.gnn_latent_dim)}l')
    trainer.train(train_params=train_params, optimizer1_params=optimizer1_params, optimizer2_params=optimizer2_params)
    _, _, _ = trainer.test()


if __name__ == '__main__':
    main()

