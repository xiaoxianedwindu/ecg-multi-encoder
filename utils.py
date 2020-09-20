from __future__ import division, print_function
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report
import os

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)

def loaddata_nosplit(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata.hdf5')
    testlabelData= ddio.load('dataset/labeldata.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def loaddata_nosplit_scaled(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    #np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def loaddata_nosplit_scaled_std(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled_std.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled_std.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    #np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def loaddata_nosplit_scaled_index(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled.hdf5')
    indexData= ddio.load('dataset/index_scaled.hdf5')

    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    X , y = att[:,:input_size], att[:, input_size:]

    import pandas as pd
    subjectLabel = (np.array(pd.DataFrame(indexData)[1]))
    nums = ['100','101','103','105','106','107','108','109','111','112','113','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    num_index = 0
    group = []
    for x in subjectLabel:
        for beat in range(x):    
            group.append(nums[num_index])
        num_index += 1
    #group = np.array(group)
    return (X, y, group)

def loaddata_nosplit_scaled_vae(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def loaddata_LOGO(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata.hdf5')
    testlabelData= ddio.load('dataset/labeldata.hdf5')
    indexData= ddio.load('dataset/index.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    import pandas as pd
    subjectLabel = (np.array(pd.DataFrame(indexData)[1]))
    group = []
    for x in subjectLabel:
        for beat in range(x):    
            group.append(x)
    group = np.array(group)
    return (X, y, group)

class LearningRateSchedulerPerBatch(LearningRateScheduler):
    """ code from https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
    Callback class to modify the default learning rate scheduler to operate each batch"""
    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
        self.count += 1


def plot_confusion_matrix(y_true, y_pred, classes, feature,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """Modification from code at https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"""
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    mkdir_recursive('results')
    fig.savefig('results/confusionMatrix-'+feature+title+'.png')
    return ax

# Precision-Recall curves and ROC curves for each class
def PR_ROC_curves(ytrue, ypred, classes, ypred_mat):
    ybool = ypred == ytrue
    f, ax = plt.subplots(3,4,figsize=(10, 10))
    ax = [a for i in ax for a in i]

    e = -1
    for c in classes:
        idx1 = [n for n,x in enumerate(ytrue) if classes[x]==c]
        idx2 = [n for n,x in enumerate(ypred) if classes[x]==c]
        idx = idx1+idx2
        if idx == []:
            continue
        bi_ytrue = ytrue[idx]
        bi_prob = ypred_mat[idx, :]
        bi_ybool = np.array(ybool[idx])
        bi_yscore = np.array([bi_prob[x][bi_ytrue[x]] for x in range(len(idx))])
        try:
            print("AUC for {}: {}".format(c, roc_auc_score(bi_ybool+0, bi_yscore)))
            e+=1
        except ValueError:
            continue
        ppvs, senss, thresholds = precision_recall_curve(bi_ybool, bi_yscore)
        cax = ax[2*e]
        cax.plot(ppvs, senss, lw=2, label="Model")
        cax.set_xlim(-0.008, 1.05)
        cax.set_ylim(0.0, 1.05)
        cax.set_title("Class {}".format(c))
        cax.set_xlabel('Sensitivity (Recall)')
        cax.set_ylabel('PPV (Precision)')
        cax.legend(loc=3)

        fpr, tpr, thresholds = roc_curve(bi_ybool, bi_yscore)
        cax2 = ax[2*e+1]
        cax2.plot(fpr, tpr, lw=2, label="Model")
        cax2.set_xlim(-0.1, 1.)
        cax2.set_ylim(0.0, 1.05)
        cax2.set_title("Class {}".format(c))
        cax2.set_xlabel('1 - Specificity')
        cax2.set_ylabel('Sensitivity')
        cax2.legend(loc=4)

    mkdir_recursive("results")
    '''
    plt.savefig("results/model_prec_recall_and_roc.eps",
        dpi=400,
        format='eps',
        bbox_inches='tight')
    '''
    plt.close()

def print_results(config, model, Xval, yval, classes, title):
    model2 = model
    if config.trained_model:
        model.load_weights(config.trained_model)
    else:    
        model.load_weights('models/MLII-'+str(title)+'latest.hdf5'.format(config.feature))
        #model.load_weights('models/{}-'+str(title)+'latest.hdf5'.format(config.feature))
    # to combine different trained models. On testing  
    if config.ensemble:
        model2.load_weight('models/weights-V1.hdf5')
        ypred_mat = (model.predict(Xval) + model2.predict(Xval))/2
    else:
        ypred_mat = model.predict(Xval)  
    ypred_mat = ypred_mat[:,0]
    yval = yval[:,0]

    ytrue = np.argmax(yval,axis=1)
    yscore = np.array([ypred_mat[x][ytrue[x]] for x in range(len(yval))])
    ypred = np.argmax(ypred_mat, axis=1)
    print(classification_report(ytrue, ypred))
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False, title = title)
    print("F1 score:", f1_score(ytrue, ypred, average=None))
    print("Average F1 score: ", np.mean(f1_score(ytrue, ypred, average=None)))
    #PR_ROC_curves(ytrue, ypred, classes, ypred_mat)

def print_results_ae_multi(config, model, Xval, yval, classes):
    model2 = model
    ypred_mat = model2.predict(Xval)


    ytrue = np.argmax(yval,axis=1)
    ypred = np.argmax(ypred_mat, axis=1)
    print(classification_report(ytrue, ypred))
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False)
    print("F1 score:", f1_score(ytrue, ypred, average=None))
    print("Average F1 score: ", np.mean(f1_score(ytrue, ypred, average=None)))

