from sklearn import metrics
import seaborn as sn

def model_pred_on_test(model,model_h5_file,batch_size):
    test_loader  = data.DataLoader(test_dataset , batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.load_state_dict(torch.load(model_h5_file))
    model.eval()
    meta_preds = torch.zeros(1,1).to(device).float()
    meta_labels = torch.zeros(1,1).to(device).float()
    running_hits = 0

    for batch, label in tqdm(test_loader):
        images = batch.to(device)
        labels = label.to(device)

        out = model(images)
        _, pred = torch.max(out, 1)
        pred = pred.view(1,pred.shape[0])
        meta_preds = torch.cat((meta_preds,pred.float()),1)
        meta_labels = torch.cat((meta_labels,torch.unsqueeze(labels.float(),1)),0)
        running_hits += (torch.sum(pred == labels)).item()

    meta_preds = meta_preds.view(len(test)+1)
    meta_preds = meta_preds[1:].cpu().numpy()
    meta_labels = meta_labels.view(len(test)+1)
    meta_labels = meta_labels[1:].cpu().numpy()
    test_acc  = running_hits / len(test)
    fpr, tpr, thresholds = metrics.roc_curve(meta_labels, meta_preds, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(meta_labels, meta_preds).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 = (2*recall*precision)/(recall+precision)
    specificity = tn/(tn+fp)

    print("AUROC = ",np.round(auroc,4))
    print("F1 score = ",np.round(F1,4))
    print("Sensitivity = ",np.round(recall,4))
    print("Specificity = ",np.round(specificity,4))
    print("Accuracy on test set = ",np.round(test_acc,4))

    conf_matrix = metrics.confusion_matrix(y_true=meta_labels, y_pred=meta_preds)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size=18)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('True', fontsize=18)
    plt.title('Confusion Matrix, Test Set', fontsize=18)
    plt.savefig('cm.png', dpi=1200)
    plt.show()

    return(meta_preds,meta_labels)
