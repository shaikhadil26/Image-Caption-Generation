import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading  
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input


def idx2word(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
      if index == integer:
          return word
  return None

def predict_caps(model, image, tokenizer, max_length):
  in_text = 'startseq'
  for i in range(max_length):
      sequence = tokenizer.texts_to_sequences([in_text])[0]
      sequence = pad_sequences([sequence], max_length)
      yhat = model.predict([image, sequence], verbose=0)
      yhat = np.argmax(yhat)
      word = idx2word(yhat, tokenizer)
      if word is None:
          break
      in_text += " " + word
      if word == 'endseq':
          break

  return in_text

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ppm *.pgm")])
    if file_path:
        saved_image_path = save_and_display_image(file_path)
        
        # Start a thread to predict the image caption
        thread = threading.Thread(target=predict_caption, args=(saved_image_path,))
        thread.start()

def save_and_display_image(file_path):
    saved_image_path = "prediction_image.png"  
    image = Image.open(file_path)
    image.save(saved_image_path)
    
    image = Image.open(saved_image_path)
    image = image.resize((300, 300))  
    img = ImageTk.PhotoImage(image)
    img_label.config(image=img)
    img_label.image = img
    
    return saved_image_path

def update_caption(caption_text):
    caption_entry.config(state=tk.NORMAL)
    caption_entry.delete(0, tk.END)  
    caption_entry.insert(0, caption_text)
    caption_entry.config(state=tk.DISABLED)  
def predict_caption(image_path):
    image = Image.open(r'C:\Users\Adil\Desktop\dlnlp\prediction_image.png')
    image = image.resize((224, 224))
    vgg_model = tf.keras.models.load_model(r'C:\Users\Adil\Desktop\dlnlp\vgg_model.h5')
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)

    model = keras.models.load_model(r'C:\Users\Adil\Desktop\dlnlp\prog20_model.h5')
    with open(r'C:\Users\Adil\Desktop\dlnlp\tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    cap = predict_caps(model, feature, tokenizer, 35)
    caption = cap.replace('startseq', '')
    predicted_caption = caption.replace('endseq', '')
    update_caption(predicted_caption)

root = tk.Tk()
root.geometry("400x400")
root.title("Image Caption generator")

img_label = tk.Label(root)
img_label.pack()

upload_button = tk.Button(root, text="Upload Image", command=browse_image)
upload_button.pack()

caption_entry = tk.Entry(root, state=tk.DISABLED, width=60) 
caption_entry.insert(0, "Caption...")
caption_entry.pack()

root.mainloop()
