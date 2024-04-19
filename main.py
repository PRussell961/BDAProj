import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from keras import layers
from keras import models, Sequential
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from sklearn.metrics import confusion_matrix
#Load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Define the model
myModel = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    #flatten input to be one dimensional
    layers.Flatten(),
    #elu is used for linear units
    #softmax converts vectors of values to probablity distribution
    layers.Dense(128, activation='elu'),
    layers.Dense(128, activation='elu'),
    layers.Dense(10, activation='softmax')
])

#Compile and print a summary of the model
#Adam optimizer performs gradient descent optimization. The loss function measures dissimilarity or discrepancy between the models predictions and actual target. 
myModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
myModel.summary()
#Run the compile model and save the data
myModel_history = myModel.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

#Loss throughout training
plt.plot(myModel_history.history['loss'], label='training')
plt.plot(myModel_history.history['val_loss'], label='validation')
plt.ylabel('loss')
plt.legend()
plt.show()
#Accuracy throughout training
plt.plot(myModel_history.history['accuracy'], label='training')
plt.plot(myModel_history.history['val_accuracy'], label='validation')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#Confusion Matrix
y_pred = myModel.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
figure, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.show()

#UI from here and below
def getValues():
  
    pos = ent_pos.get()
    print(pos)
    #Shape data and get predictions
    problem = myModel.predict(x_test[eval(pos)].reshape(1,28,28))
    prediction = np.argmax(problem, axis=1)
    print(problem[0], " => ", prediction[0])
    plt.imshow(x_test[eval(pos)], cmap="Greys")
    plt.show()
    

# Set up the window
window = tk.Tk()
window.title("Number Checker")
window.resizable(width=False, height=False)


frm_entry = tk.Frame(master=window)
ent_pos = tk.Entry(master=frm_entry, width=10)
lbl_pos = tk.Label(master=frm_entry, text="Enter Pos")

ent_pos.grid(row=0, column=0, sticky="e")
lbl_pos.grid(row=0, column=1, sticky="w")

btn_display = tk.Button(
    master=window,
    text="\N{RIGHTWARDS BLACK ARROW}",
    command=getValues
)


frm_entry.grid(row=0, column=0, padx=10)
btn_display.grid(row=0, column=1, pady=10)

# Run the application
window.mainloop()
