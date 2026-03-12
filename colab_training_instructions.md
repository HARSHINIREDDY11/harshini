# Training on Google Colab for High Accuracy

Since training a Swin Transformer locally can be slow and memory-intensive, using Google Colab's free GPUs is a fantastic idea to reach your 94-96% accuracy target quickly. Here is a step-by-step guide on how to set it up:

## 1. Prepare Your Repository
Ensure all the changes we just made to `src/model.py`, `src/train.py`, and `src/dataset.py` are pushed to your GitHub repository.

```bash
git add src/model.py src/train.py src/dataset.py
git commit -m "Optimize model and training for 94% accuracy"
git push origin main
```

## 2. Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click on **New Notebook**.
3. Go to **Runtime > Change runtime type**.
4. Select **T4 GPU** (or any available GPU) as the Hardware accelerator and click **Save**.

## 3. Clone and Setup in Colab
In the first cell of your Colab notebook, enter the following commands to clone your repository and install dependencies:

```python
!git clone https://github.com/HARSHINIREDDY11/harshini.git
%cd harshini
!pip install -r requirements.txt
```
*(Run this cell by clicking the play button on the left or pressing `Shift + Enter`)*

## 4. Run the Training Script
In the next cell, run your training script exactly as you would locally. The script is already configured to stop automatically once validation accuracy hits 94.0%.

```python
!python src/train.py --epochs 50 --batch_size 16 --lr 1e-4
```

## 5. Download Your Best Model
Once the training stops (because it hit 94%+), it will save a file called `best_model.pth`. You can download this back to your local computer to use in your Streamlit app!

Run this in a final cell:
```python
from google.colab import files
files.download('best_model.pth')
```

After downloading it, just replace the existing `best_model.pth` in your local `c:\Users\harsh\Downloads\mwdst\` folder with this newly trained high-accuracy one, and your `app.py` will automatically use it!
