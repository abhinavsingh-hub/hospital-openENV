from PIL import Image
import os

input_path = "/Users/abhinavsingh/.gemini/antigravity/brain/41405bf8-1019-4863-88f9-b7da4da4cc3a/hospital_ai_logo_black_bg_1775345382985.png"
output_path = "/Users/abhinavsingh/Documents/scaler/assets/logo_transparent.png"

img = Image.open(input_path).convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    # If the pixel is very dark, make it transparent
    if item[0] < 30 and item[1] < 30 and item[2] < 30:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save(output_path, "PNG")
print(f"Processed logo saved to {output_path}")
