import torch
from torch import nn
import torch.nn.functional as F

class AnchorStore(nn.Module):

    def __init__(self, K=1024, dim=50257, knn=1, n_class=2):
        super(AnchorStore, self).__init__()

        self.register_buffer("queue_anchor", torch.randn(K, dim))
        self.register_buffer("queue_label", torch.zeros(K, dtype=torch.long))
        self.queue_label.fill_(-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.knn = knn
        self.n_class = n_class

    def enqueue(self, anchors, labels):

        ptr = int(self.queue_ptr)
        bs = anchors.shape[0]

        self.queue_anchor[ptr:ptr + bs, :] = anchors
        self.queue_label[ptr:ptr + bs] = labels
        self.queue_ptr[0] = ptr + bs

    def knn_infer(self, query):

        # kl_div.shape = [1, len(self.queue_anchor)]
        kl_distance = torch.mean(self.queue_anchor[:, None, :] * (self.queue_anchor[:, None, :].log() - query.log()), dim=2).transpose(1, 0)
        if self.knn == 1:
            # directly return the nearest neighbor
            return self.queue_label[kl_distance.argmin(dim=1)].tolist()
        else:
            values, indices = torch.topk(kl_distance, self.knn, dim=1, largest=False)
            # count for each category within k nearest neighbors, and return the dominant category
            # knn_cnt.shape = [1, self.n_class]
            knn_cnt = torch.zeros((query.shape[0], self.n_class))
            for i in range(self.n_class):
                knn_cnt[:, i] = (self.queue_label[indices] == i).sum(dim=1)
            return knn_cnt.argmax(dim=1).tolist()
