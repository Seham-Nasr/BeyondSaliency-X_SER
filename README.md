## BEYOND SALIENCY: ENHANCING EXPLANATION OF SPEECH EMOTION RECOGNITION WITH EXPERT-REFERENCED ACOUSTIC CUES

Current saliency-based methods, adapted from vision, highlight spectrogram regions but fail to show whether these regions correspond to meaningful acoustic markers of emotion, limiting faithfulness and interpretability. We propose a framework that overcomes these limitations by quantifying the magnitudes of cues within salient regions. This clarifies “what” is highlighted and connects it to “why” it matters, linking saliency to expert-referenced acoustic cues of speech emotions.

<img width="1782" height="1470" alt="image" src="https://github.com/user-attachments/assets/54e6e759-160d-4ac2-bbda-cdd37562f0f3" />

Experiments on benchmark SER datasets show that our approach improves explanation quality by explicitly linking salient regions to theory-driven speech emotions expert-referenced acoustics. Compared to standard saliency methods, it provides more understandable and plausible
explanations of SER models, offering a foundational step towards trustworthy speech-based affective computing.


## Quickstart

### GradCAM
```python


```



## Reproduce our results
First, get the dataset from datasoucres one of the them is Kaggel as below
 #### Toronto emotional speech set (TESS)
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

print("Path to dataset files:", path)

```
#### Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ejlok1/cremad")

print("Path to dataset files:", path)
```

Second, Rum the data processing for each dataset Follow the python command:

```python python Models/SER_data.py --data_path ./TESS_df.pkl --dataset_name TESS --sample_rate 16000 ```



### 📃 Citation
```
@article{

}
```
