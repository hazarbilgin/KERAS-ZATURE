#yazar=hazarbilgin
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers,models, preprocessing
#gerekli dosyaların bulunduğu dosyalara ulaşma
train_dir = 'C:\\Users\\Hazar\\xray_dataset_covid19\\train'
test_dir = 'C:\\Users\\Hazar\\xray_dataset_covid19\\test'
img_size = (150, 150)
batch_size = 32
#train edilcek görsellerin elde edilmesi
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',  
    validation_split=0.2,
    subset='training',
    seed=123
)
#validation edilcek görsellerin elde edilmesi
val_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',  
    validation_split=0.2,
    subset='validation',
    seed=123
)
#test edilcek görsellerin elde edilmesi
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='int',  
    seed=123
)
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#modelimizin katmanları sinir ağların oluşturulduğu yer
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
#modelimizi compile etme ve 'adam' optimazasyonu ile optime edilmesi metrik olarak doğruluk 'accuracy' metriği kullanımı
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
#modelimizi eğitmeye başlanması 10 aşamalık epochsu
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
#modelin bilgisayara kaydedilmesi
model.save('pneumonia.h5')
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print("\nTest accuracy:", test_acc)

test_images, test_labels = next(iter(test_ds))
predictions = model.predict(test_images)
#görüntüleri ekrana verilmesi fonksiyonu plt kütüphanesi ile
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i].numpy().astype("uint8"))
    plt.title(f"Zature: {predictions[i][0]:.2f}")
    plt.axis('off')
plt.tight_layout()
plt.show()


#                     ------------------------------------- Yeni resimleri sisteme yükleme--------------------------------#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

#kaydettiğimiz modeli tekrar  yükleme 
model_path = 'C:\\Users\\Hazar\\pneumonia.h5'
model = tf.keras.models.load_model(model_path)
#yeni çekilcek resimin dosya yolu 
image_path = 'C:\\Users\\Hazar\\xray_dataset_covid19\\test\\ZATURE\\streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day3.jpg'
#kullanılcak resmi ön işleme
def preprocess_image(img_path, img_size=(150, 150)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size used during training
    return img_array

img = preprocess_image(image_path)

predictions = model.predict(img)

prediction = predictions[0][0]
#yüksek olasılıkla zatureli akciğeri mi tespiti 
class_label = 'Pneumonia' if prediction > 0.5 else 'Normal'
#yapay zeka modelimizin zatureli hastanın akciğer filminin tespitini ekranda gösterilmesi
plt.figure(figsize=(6, 6))
plt.imshow(image.load_img(image_path))
plt.title(f"Zature: {class_label} ({prediction:.2f})")
plt.axis('off')
plt.show()
#gerekli kütüphaneler
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#modelimizde resnet50 kullanımı
model = resnet50(pretrained=True)
model.evalu()

target_layers = [model.layer4[-1]]
#dosya yolu
image_path = 'C:\\Users\\Hazar\\xray_dataset_covid19\\test\\ZATURE\\streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day3.jpg'

# görselin ön işlenmesi 
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

input_tensor = preprocess_image(image_path)

def tensor_to_np(image_tensor):
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
    image = image - np.min(image)
    image = image / np.max(image)
    return image

rgb_img = tensor_to_np(input_tensor)

# GRAD-CAM'ı modelimize dahil etme
cam = GradCAM(model=model, target_layers=target_layers)
#bunu gerçek hedef sınıfıyla değiştirebilirsiniz 
targets = [ClassifierOutputTarget(281)] 

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#elde edilen görüntünün ekrana verilmesi plt kütüphanesi ile
plt.imshow(visualization)
plt.axis('off')
plt.show()




#yazar=hazarbilgin
