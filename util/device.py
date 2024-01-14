import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
device = torch.device('cuda:0')
# device = torch.device('cuda:1')
#用于将数据转移到GPU上
def to_device(data):
    if isinstance(data, (list, tuple)):
        return [to_device(x) for x in data]
    return data.to(device)