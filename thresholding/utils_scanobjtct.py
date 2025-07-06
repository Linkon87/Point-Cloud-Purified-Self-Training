import torch


class ThresholdingHook:

    def __init__(self, num_classes=15, momentum=0.999):
        self.num_classes = num_classes
        self.m = momentum

        self.p_model = torch.ones((self.num_classes)) / self.num_classes  #

        self.global_threshold = torch.tensor([])
        self.queue = []
        self.posterior_p = torch.tensor([])
        self.posterior_q = torch.tensor([])

    @torch.no_grad()
    def update(self, probs_x_ulb):

        self.queue.extend(probs_x_ulb)
        self.queue = self.queue[-512:]
        max_probabilities = []
        for tensor in self.queue:
            max_prob, _ = torch.max(tensor, dim=-1)
            max_probabilities.append(max_prob)

        self.global_threshold = torch.mean(torch.stack(max_probabilities))  # eq2
        self.posterior_p = torch.mean(torch.stack(self.queue), dim=0)  # eq3

        normalized_in = self.p_model / self.posterior_p * probs_x_ulb
        l1_norm = torch.norm(normalized_in, p=1, dim=1, keepdim=True)
        self.posterior_q = normalized_in / l1_norm  # eq3

    @torch.no_grad()
    def masking(self, logits_x_ulb):

        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.global_threshold.is_cuda:
            self.global_threshold = self.global_threshold.to(logits_x_ulb.device)

        probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        self.update(probs_x_ulb)

        max_probs, max_idx = self.posterior_q.max(dim=-1)
        mask = max_probs.ge(self.global_threshold).to(max_probs.dtype)
        return mask  # , self.global_threshold



