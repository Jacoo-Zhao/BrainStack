import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


@torch.no_grad()
def validate(model, val_loader, criterion, loss_fn, device):
    model.eval()
    running_loss = 0.0
    total = 0

    correct_fused = 0
    correct_all = 0
    correct_conformer = 0

    full_labels = []
    full_logits = []

    # If the model defines brain area information, initialize the correct count for each area
    if hasattr(model, 'eeg_areas'):
        correct_areas = [0 for _ in range(len(model.eeg_areas))]
        area_names = list(model.eeg_areas.keys())
    else:
        correct_areas = None
        area_names = []

    for batch in val_loader:
        inputs, labels = batch["eeg"].to(device), batch["label"].to(device)

        full_labels.append(labels.cpu().detach())

        outputs = model(inputs)

        fused_logits = outputs["out"]
        distill_teacher_logits = outputs.get("distill_teacher", None)
        conformer_logits = outputs.get("all_logits", None)

        area_logits = outputs.get("area_logits", None)

        loss_dict = loss_fn(outputs, labels)
        loss = loss_dict["loss"]

        running_loss += loss.item()
        total += labels.size(0)

        # fused
        _, pred_fused = torch.max(fused_logits, 1)
        full_logits.append(fused_logits.cpu().detach())
        correct_fused += (pred_fused == labels).sum().item()

        kappa = cohen_kappa_score(labels.cpu().numpy(), pred_fused.cpu().numpy())
        # print("Cohen's Kappa:", kappa)
        
        # distill_teacher
        if distill_teacher_logits is not None:
            _, pred_all = torch.max(distill_teacher_logits, 1)
            correct_all += (pred_all == labels).sum().item()


        # global branch 
        if conformer_logits is not None:
            _, pred_conformer = torch.max(conformer_logits, 1)
            correct_conformer += (pred_conformer == labels).sum().item()

        
        # local branch
        if area_logits is not None and correct_areas is not None:
            for i, area_logit in enumerate(area_logits):
                _, pred_area = torch.max(area_logit, 1)
                correct_areas[i] += (pred_area == labels).sum().item()

    # Summary
    loss_avg = running_loss / len(val_loader)
    fused_acc = 100 * correct_fused / total
    all_acc = 100 * correct_all / total if distill_teacher_logits is not None else 0

    conformer_acc = 100 * correct_conformer / total if conformer_logits is not None else 0
    
    area_accs = [100 * c / total for c in correct_areas] if area_logits is not None else 0

    full_labels = torch.cat(full_labels, dim=0).cpu().numpy()  # Flatten and convert to NumPy array
    full_logits = torch.cat(full_logits, dim=0).argmax(dim=1).cpu().numpy()  # Get predicted classes and flatten

    f1 = f1_score(full_labels, full_logits, average='macro')
    kappa = cohen_kappa_score(full_labels, full_logits)

    return loss_avg, fused_acc, all_acc, area_accs, [conformer_acc], f1, kappa