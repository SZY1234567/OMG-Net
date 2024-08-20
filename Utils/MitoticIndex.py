import numpy as np
import torch
import lightning as L
class KDE_Estimator(L.LightningModule): # torch version of the above.
    def __init__(self, bandwidth):
        super(KDE_Estimator, self).__init__()
        self.bandwidth = bandwidth

    def forward(self, x, y, z):
        dist_sq       = torch.cdist(torch.stack((x, y), dim=2).squeeze(), torch.stack((x, y), dim=2).squeeze(), p=2) ** 2
        kernel_matrix = torch.exp(-0.5 * dist_sq / (self.bandwidth ** 2))
        attention_KDE = torch.sum(kernel_matrix * z, axis=1)
        attention_KDE = (attention_KDE - attention_KDE.min()) / (attention_KDE.max() - attention_KDE.min())
        return attention_KDE

    def predict(self, x, y, z):
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        z = torch.tensor(z, dtype=torch.float32).to(self.device)
        return self.forward(x, y, z).cpu().numpy()


def AIMitoticIndex(data):
    if data.shape[0]==0:
        return (0,0), 0
    print(data,data.dtypes)
    # Prepare data
    x, y = data['coords_x'].to_numpy().astype(int), data['coords_y'].to_numpy().astype(int)
    z = data['pred_1'].to_numpy()
    
    # Estimate density
    kde_model = KDE_Estimator(bandwidth=10).to('cuda:0')
    density   = kde_model.predict(x=x,y=y,z=z)
    max_density_index = np.argmax(density)
    center    = (x[max_density_index], y[max_density_index])
    
    r = 3750
    data_in = data[((data['coords_x'] - center[0]) ** 2 + (data['coords_y'] - center[1]) ** 2) < r ** 2]
    print('data_in', data_in)
    if(len(data_in)>0):
        return center, len(data_in)
    else:
        return center, 1
