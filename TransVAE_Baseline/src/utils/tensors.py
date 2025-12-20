import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device='cpu').expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate(batch): #for old script compatibility
    # print('\nBEFORE',batch[0]['dance'].shape)
    databatch = [torch.concat([b['dance'][:,-3:],b['dance'][:,3:3*24]], 1) for b in batch]
    # print(databatch[0].shape,'AFTER\n')
    
    labelbatch = [torch.cat((b['music'], b['lyrics']), dim=1) for b in batch]
    lenbatch = [180 for b in batch]

    databatchTensor = collate_tensors(databatch)
    databatchTensor = databatchTensor.view(len(databatch), 180, 24, 3).permute(0, 2, 3, 1).float()
    labelbatchTensor = collate_tensors(labelbatch).float()
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch

def collate72(batch):
    # print('\nBEFORE',batch[0]['dance'].shape)
    databatch = [torch.concat([b['dance'][:,-3:],b['dance'][:,3:3*24]], 1) for b in batch]
    # print(databatch[0].shape,'AFTER\n')
    
    labelbatch = [torch.cat((b['music'], b['lyrics']), dim=1) for b in batch]
    lenbatch = [180 for b in batch]

    databatchTensor = collate_tensors(databatch)
    databatchTensor = databatchTensor.view(len(databatch), 180, 24, 3).permute(0, 2, 3, 1).float()
    labelbatchTensor = collate_tensors(labelbatch).float()
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch

def collate78(batch):
    # print('\nBEFORE',batch[0]['dance'].shape)
    databatch = [torch.concat([b['dance'][:,-3:],b['dance'][:,3:3*26]], 1) for b in batch]
    # print(databatch[0].shape,'AFTER\n')
    
    labelbatch = [torch.cat((b['music'], b['lyrics']), dim=1) for b in batch]
    lenbatch = [180 for b in batch]

    databatchTensor = collate_tensors(databatch)
    databatchTensor = databatchTensor.view(len(databatch), 180, 26, 3).permute(0, 2, 3, 1).float()
    labelbatchTensor = collate_tensors(labelbatch).float()
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    batch = {"x": databatchTensor, "y": labelbatchTensor,
             "mask": maskbatchTensor, "lengths": lenbatchTensor}
    return batch

