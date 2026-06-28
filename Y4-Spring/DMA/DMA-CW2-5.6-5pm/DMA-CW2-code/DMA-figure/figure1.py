from PIL import Image
import matplotlib.pyplot as plt

img1 = Image.open("price_distribution.png")
img2 = Image.open("outliers.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(img1)
axes[0].axis('off')
axes[0].set_title('Price Distribution')

axes[1].imshow(img2)
axes[1].axis('off')
axes[1].set_title('Outlier Analysis')

plt.tight_layout()
plt.show()