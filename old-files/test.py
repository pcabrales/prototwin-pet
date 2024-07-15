import torch
from tqdm import tqdm
from utils import RE_loss, range_loss
from zmq.error import NotDone

def test(trained_model, test_loader, results_dir='.'):
    # Test loop (after the training is complete)
    RE_loss_list = []
    R90_list = []
    R50_list = []
    R10_list = []
    with torch.no_grad():
        for batch_input, batch_target in tqdm(test_loader):
            batch_input = batch_input.to(device)
            batch_output = trained_model(batch_input)
            batch_output = batch_output.detach().cpu()
            RE_loss_list.append(RE_loss(batch_output, batch_target))
            R90_list.append(range_loss(batch_output, batch_target, 0.9))
            R50_list.append(range_loss(batch_output, batch_target, 0.5))
            R10_list.append(range_loss(batch_output, batch_target, 0.1))
        torch.cuda.empty_cache()

    RE_loss = torch.cat(RE_loss_list)
    R90_loss = torch.cat(R90_list)
    R50_loss = torch.cat(R50_list)
    R10_loss = torch.cat(R10_list)

    text_results = f"Relative Error: {torch.mean(RE_loss)} +- {torch.std(RE_loss)}\n" \
           f"R90: {torch.mean(R90_loss)} +- {torch.std(R90_loss)}\n" \
           f"R50: {torch.mean(R50_loss)} +- {torch.std(R50_loss)}\n" \
           f"R10: {torch.mean(R10_loss)} +- {torch.std(R10_loss)}"
    print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)
        
    return None



