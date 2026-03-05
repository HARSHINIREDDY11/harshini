# Multi-View Waste Segmentation using Swin Transformer & 3D CNN Fusion for Smart Recycling System

## Authors
**ANJALI V S**
CSE. Alliance University (UGC)
Bangalore, India
anjalivs0710@gmail.com

**AAKASHI JAISWAL**
CSE. Alliance University (UGC)
Bangalore, India
jaiswalaakashi123@gmail.com

**BODAM HARSHINI**
CSE. Alliance University (UGC)
Bangalore, India
rharshini085@gmail.com

**KATTA DEEPAK**
CSE. Alliance University (UGC)
Bangalore, India
xxxxx@gmail.com

---

## Abstract
The rapid urbanization and increasing generation of solid waste pose significant challenges to municipal recycling systems. Traditional automated sorting methods relying on single-view computer vision often struggle with material ambiguities, specular reflections (such as glass and metal), and occlusions, leading to sub-optimal classification accuracy. This paper introduces EcoView AI, a novel intelligent waste classification and segmentation system that employs a hybrid architecture combining Swin Transformers and 3D Convolutional Neural Networks (CNN) for multi-view image fusion. By processing multiple camera views of an object through shared-weight transformer backbones and fusing the resulting features via a 3D CNN layer, the system explicitly models spatial correlations across different angles. The proposed dual-task learning framework simultaneously performs 6-class waste categorization on the TrashNet dataset and pixel-level segmentation. Furthermore, Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated to provide explainable AI insights into the model's decision-making process. Experimental results demonstrate that EcoView AI achieves a classification accuracy of 94.8%, significantly outperforming traditional CNN (85.2%) and single-view transformer (91.4%) baselines.

**Keywords—** Waste Segmentation, Multi-View Fusion, Swin Transformer, 3D CNN, Deep Learning, Explainable AI.

---

## I. INTRODUCTION

Effective waste management and recycling are critical pillars of sustainable urban development. As the global volume of municipal solid waste continues to rise, the limitations of manual sorting—including high labor costs, inefficiency, and health hazards—have driven the demand for automated recycling facilities. Computer vision and deep learning have emerged as promising technologies to address these challenges, enabling non-contact, high-speed waste characterization.

However, existing automated classification systems primarily rely on single-view Convolutional Neural Networks (CNNs). While these models perform adequately on clear, distinct objects, they frequently fail when presented with challenging materials. For example, transparent glass or highly reflective metallic items exhibit variable appearances depending on the lighting and viewing angle. Moreover, objects are often deformed, crushed, or partially occluded, causing single-view systems to miss crucial identifying features. 

To overcome these limitations, this paper proposes a multi-view fusion approach. By capturing and analyzing an object from 2 to 4 distinct angles simultaneously, the system can leverage inter-view spatial correlations to make robust predictions. We introduce EcoView AI, which utilizes a Swin Transformer backbone to extract high-level semantic features from each view. These features are then aggregated using a 3D CNN fusion layer. Additionally, the system provides both classification and segmentation outputs, representing a comprehensive dual-task learning framework tailored for advanced sorting pneumatics or robotic arms.

The main contributions of this paper are:
1) A novel hybrid architecture utilizing Swin Transformers and 3D CNNs to explicitly model cross-view relationships in waste objects.
2) A dual-task learning approach that achieves both high-accuracy classification and pixel-level segmentation.
3) The application of Grad-CAM to introduce explainability to deep learning-based recycling decisions.

## II. RELATED WORK

The foundation of modern image-based waste classification was established with the introduction of the TrashNet dataset [1], which categorized items into six common household waste types. Early approaches applied standard deep learning models like ResNet and VGG16 to this dataset, achieving baseline accuracies hovering between 85% and 90% [2], [3].

Recent advancements have seen the integration of Vision Transformers (ViT) into this domain. Transformers, with their self-attention mechanisms, have shown superior capability in capturing global context compared to CNNs, pushing single-view accuracies slightly above 90% [4]. Despite these improvements, the fundamental limitation of single-perspective imaging remains.

Multi-view representation learning has seen success in 3D object recognition and medical imaging, but its application to waste sorting is limited. Methods generally either concatenate view features early or average predictions late. The proposed work distinguishes itself by using a 3D CNN layer to inherently learn the spatial relationship between the extracted Swin Transformer feature maps of the distinct views, maximizing the retention of spatial-angular information.

## III. PROPOSED SYSTEM ARCHITECTURE

The overall workflow of the EcoView AI system is designed to seamlessly process multi-view inputs and produce highly actionable outputs for recycling machinery. The architecture comprises four core components: Data Preprocessing, Feature Extraction, 3D Feature Fusion, and Dual-Task Output.

### A. Feature Extraction via Swin Transformer
The system accepts an input tensor of dimension $\[V \times 3 \times 224 \times 224\]$, where $V$ represents the number of camera views (ranging from 2 to 4). These views are passed through a shared-weight Swin Transformer backbone. Unlike traditional CNNs, the Swin Transformer computes self-attention within non-overlapping local windows and employs a shifted window partitioning scheme. This hierarchical architecture provides computational efficiency while modeling long-range dependencies across the image patch sequences. Leveraging pre-trained weights, the backbone extracts rich semantic feature maps from each individual view independently.

### B. 3D CNN Fusion Layer
The isolated feature maps retrieved from the $V$ views are stacked along a new temporal/angular dimension, forming a 3D feature volume. Traditional 2D fusion methods (like simple averaging or concatenation) often destroy inter-view spatial topologies. To preserve and learn from these correlations, the stacked features are processed by a sequence of 3D Convolutional layers. The 3D kernels slide across spatial dimensions (height and width) as well as the view dimension, allowing the network to recognize features that exist only through the combination of specific angles (e.g., recognizing a crushed can by combining a side profile with a top-down occlusion).

### C. Dual-Task Learning: Classification and Segmentation
Following the 3D fusion, the aggregated semantic vector is flattened and passed into a fully connected Classifiction Head. This network predicts the probability distribution across the six primary categories (cardboard, glass, metal, paper, plastic, trash).

Simultaneously, a parallel Segmentation Decoder network upsamples the intermediate fused feature maps. Through a series of transposed convolutions, it reconstructs a $224 \times 224$ spatial mask. This pixel-level mask isolates the exact boundaries of the target object against the background or conveyor belt surface, vital for robotic grasping algorithms.

### D. Explainable AI (Grad-CAM)
To ensure system transparency, Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated. Grad-CAM utilizes the gradients of the target concept flowing into the final convolutional layer to produce a coarse localization map. This heatmap highlights the specific regions of the input views that contributed most strongly to the model's final classification decision, thereby verifying that the system is identifying material textures rather than background artifacts.

## IV. EXPERIMENTAL RESULTS

### A. Dataset and Training
The model was evaluated using a customized adaptation of the TrashNet dataset. To train the multi-view architecture, synthetic multi-view instances were generated by grouping augmented images of the same base class. The network was optimized using the AdamW optimizer with a Cosine Annealing learning rate schedule. A combined loss function consisting of Cross-Entropy Loss (for classification) and Dice Loss (for segmentation) was utilized to train the dual tasks simultaneously.

### B. Performance Evaluation
The proposed EcoView AI model was benchmarked against a standard single-view CNN (representing the traditional industry standard) and a Single-View Swin Transformer.

As shown in TABLE I, the proposed multi-view fusion architecture achieved a significant leap in accuracy. Note specifically the performance on challenging materials like glass and metal, where specularities and complex topological self-occlusions severely hamper single-view methods.

**TABLE I. QUANTITATIVE PERFORMANCE COMPARISON**

| Model | Average Accuracy | Glass Accuracy | Metal Accuracy |
|-----------------------|-----------|----------------|----------------|
| CNN Baseline | 85.2% | 78.0% | 82.0% |
| Swin Single-View | 91.4% | 85.0% | 89.0% |
| **EcoView (Proposed)**| **94.8%** | **96.0%** | **95.0%** |

The multi-view system improved overall accuracy by 9.6% over the CNN baseline and 3.4% over the single-view transformer. Furthermore, the segmentation mask accurately bounded complex object geometries, demonstrating the efficacy of the dual-task loss optimization.

## V. CONCLUSION

This paper presented EcoView AI, a state-of-the-art multi-view waste segmentation system designed for automated recycling. By capitalizing on the robust feature representation of Swin Transformers and the spatial correlation capabilities of 3D CNNs, the system successfully addresses the longstanding challenges of material specularity and structural occlusion in single-view systems. Reaching a benchmark accuracy of 94.8% and rendering explainable decision roadmaps via Grad-CAM, the architecture proves highly viable for deployment in modern smart recycling facilities. Future work will investigate embedding inference models on edge-computing devices to further minimize latency in high-speed sorting environments.

## REFERENCES

[1] G. Thung and M. Yang, "Classification of Trash for Recyclability Status," CS229 Project Report, Stanford University, 2016.
[2] C. Bircanoğlu, M. Atay, F. Beşer, Ö. Genç, and M. A. Kızrak, "RecycleNet: Intelligent waste sorting using deep neural networks," in *2018 Innovations in Intelligent Systems and Applications (INISTA)*, pp. 1-7, IEEE, 2018.
[3] O. Adedeji and Z. Wang, "Intelligent waste classification system using deep learning convolutional neural networks," *Procedia Manufacturing*, vol. 35, pp. 607-612, 2019.
[4] Z. Liu et al., "Swin transformer: Hierarchical vision transformer using shifted windows," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 10012-10022, 2021.
