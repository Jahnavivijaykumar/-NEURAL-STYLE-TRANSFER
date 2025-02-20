import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Function to load and preprocess an image
def load_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path).resize(target_size)
    img = np.array(img, dtype=np.float32) / 255.0
    return img[np.newaxis, ...]

# Function to apply style transfer
def apply_style_transfer(content_path, style_path):
    try:
        content_image = load_image(content_path)
        style_image = load_image(style_path)

        # Load pre-trained model
        model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        # Apply style transfer
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

        return np.array(stylized_image)

    except Exception as e:
        print(f"Error in style transfer: {e}")
        return None

# GUI application class
class StyleTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Style Transfer")
        self.root.geometry("600x400")

        # Labels & Buttons
        self.label1 = tk.Label(root, text="Upload Content Image:")
        self.label1.pack()

        self.content_btn = tk.Button(root, text="Choose Content Image", command=self.load_content_image)
        self.content_btn.pack()

        self.label2 = tk.Label(root, text="Upload Style Image:")
        self.label2.pack()

        self.style_btn = tk.Button(root, text="Choose Style Image", command=self.load_style_image)
        self.style_btn.pack()

        self.run_btn = tk.Button(root, text="Apply Style Transfer", command=self.run_style_transfer)
        self.run_btn.pack()

        self.result_label = tk.Label(root, text="Stylized Image:")
        self.result_label.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Image Paths
        self.content_path = None
        self.style_path = None

    # Function to load content image
    def load_content_image(self):
        self.content_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.jfif")])
        if self.content_path:
            print(f"Selected Content Image: {self.content_path}")

    # Function to load style image
    def load_style_image(self):
        self.style_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.jfif")])
        if self.style_path:
            print(f"Selected Style Image: {self.style_path}")

    # Function to run style transfer
    def run_style_transfer(self):
        if self.content_path and self.style_path:
            output_image = apply_style_transfer(self.content_path, self.style_path)

            if output_image is not None:
                # Convert array to image
                output_image = np.squeeze(output_image, axis=0)  # Remove batch dimension
                output_pil = Image.fromarray((output_image * 255).astype(np.uint8))  # Convert to PIL format

                # Display image in Tkinter window
                output_pil = output_pil.resize((256, 256))  # Resize for display
                output_tk = ImageTk.PhotoImage(output_pil)

                self.image_label.config(image=output_tk)
                self.image_label.image = output_tk  # Keep reference
        else:
            print("Please select both content and style images.")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()
