import sys

sys.path.insert(0, '../../src')

from vtrain.core import Engine, vrun
import timm
import torch
import torchmetrics as metrics
from idatasets.famous import CUB200


class Trainer(Engine):

    @Engine.init
    def initialize(self):
        net = timm.create_model("deit3_small_patch16_224_in21ft1k", pretrained=True, num_classes=200)

        self.add_net('bird', net)

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = metrics.Accuracy().cuda()
        self.add_watch(name='Train@top1')
        self.out = open('./out.txt', 'w')

    def configure_optimizers(self):
        net = self.get_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.00001)
        self.set_opt_sch('bird', opt, None)

    def on_train_step(self, datas, **kwargs):
        images, labels = datas[:2]

        logits = self.forward(images, name='bird')
        loss = self.loss(logits, labels)
        loss.backward()
        self._step(name='bird')

        acc = self.acc(logits, labels)
        self.upd_watch('Train@top1', acc)

        return loss

    def on_validation_step(self, datas, **kwargs):
        images, labels = datas[:2]
        logits = self.forward(images, name='bird')
        loss = self.loss(logits, labels)

        return loss

    def create_loader(self):
        bird = CUB200(data_root='/data/image/FGCV/CUB_200_2011', is_ddp=(self.cconf.backend=='ddp'))

        train_loader = bird.create('train', batch_size=16, num_workers=8, shuffle=True, pin_memory=True)
        if self.cconf.mode == 'test':
            test_loader = bird.create('test', batch_size=16, num_workers=8, pin_memory=True, with_path=True)
        else:
            test_loader = bird.create('test', batch_size=16, num_workers=4, pin_memory=True)

        return train_loader, test_loader

    def on_test_step(self, datas, **kwargs):
        images, labels, paths = datas[:3]
        logits = self.forward(images, name='bird')

        pred = torch.softmax(logits, dim=1)
        prob, pidx = torch.max(pred, dim=1)

        for i in range(logits.shape[0]):
            self.out.write("%f %d %s\n" % (prob[i].item(), pidx[i].item(), paths[i]))
        self.acc(logits, labels)

    def on_test_epoch_end(self, **kwargs):
        self.logger.info("test top1: %f" % (self.acc.compute()))


if __name__ == '__main__':
    vrun(Trainer)
