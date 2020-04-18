import matplotlib.pyplot as plt
import os

train_loss = []
train_acc = []
val_loss = []
val_acc = []
names = []
for filename in os.listdir('./'):
    if filename == 'requirements.txt':
        continue
    if filename[-4:] == '.txt':
        f = open(filename, 'r')
        lines = f.readlines()
        lines = lines[1:len(lines)-2]
        names.append(filename[10:len(filename)-13])
        train_loss.append([])
        train_acc.append([])
        val_loss.append([])
        val_acc.append([])
        for line in lines:
            words = line.split()
            train_loss[-1].append(float(words[1]))
            train_acc[-1].append(float(words[3]))
            val_loss[-1].append(float(words[5]))
            val_acc[-1].append(float(words[7]))

titles = ['Training accuracy', 'Training loss', 'Validation accuracy', 'Validation loss']
png_names = ['accuracy/train_cnn_acc','loss/train_cnn_loss','accuracy/val_cnn_acc','loss/val_cnn_loss']
y_label = ['Accuracy','Loss','Accuracy','Loss']
data = [train_acc, train_loss, val_acc, val_loss]
for i in range(len(titles)):
    for opt in data[i]:
        plt.plot(opt)
    plt.title(titles[i]+' of cnn using different optimizers')
    plt.ylabel(y_label[i])
    plt.xlabel('Epoch')
    plt.legend(names, loc='upper right')
    plt.savefig(png_names[i]+'.png')
    plt.show()

# plt.plot(train_loss)
# # plt.plot(val_loss)
# plt.title('Loss of '+filename[:len(filename)-4])
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('loss/'+filename[:len(filename)-4]+' acc.png')
# plt.show()