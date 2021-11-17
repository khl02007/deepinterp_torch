import torch
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    given = []
    result = []
    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            pred = model(X.float())
            given.append(y.float())
            result.append(pred)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
    return given, result

def cat_numpy(given, result):
    given_cat = torch.cat(given, dim=0).detach().cpu().numpy()
    result_cat = torch.cat(result, dim=0).detach().cpu().numpy()
    return given_cat, result_cat