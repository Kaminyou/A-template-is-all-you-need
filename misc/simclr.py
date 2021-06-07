import numpy as np
import torch

def simclrtrain(net, data_loader, train_optimizer, temperature=0.5, device="cpu"):
   total_loss, total_num = 0.0, 0
   for now, (pos_1, pos_2, label, ID) in enumerate(data_loader):
       size = len(ID)
       print(f"process {now+1} | {len(data_loader)}", end = "\r")
       pos_1, pos_2 = pos_1.float().to(device, non_blocking=True), pos_2.float().to(device, non_blocking=True)
       out_1 = net(pos_1)
       out_2 = net(pos_2)
       # [2*B, D]
       out = torch.cat([out_1, out_2], dim=0)
       # [2*B, 2*B]
       sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
       mask = (torch.ones_like(sim_matrix) - torch.eye(2 * size, device=sim_matrix.device)).bool()
       # [2*B, 2*B-1]
       sim_matrix = sim_matrix.masked_select(mask).view(2 * size, -1)

       # compute loss
       pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
       # [2*B]
       pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
       loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
       train_optimizer.zero_grad()
       loss.backward()
       train_optimizer.step()

       total_num += size
       total_loss += loss.item() * size
       #train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

   return total_loss / total_num