"""
Generates a realistic sample PDF for testing the RAG pipeline.
Run with: python3 data/sample_pdfs/create_sample_pdf.py
Requires: pip install reportlab
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "machine_learning_notes.pdf")

content = [
    # Page 1
    ("Introduction to Machine Learning", [
        """Machine learning (ML) is a subfield of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed. The term was coined by Arthur Samuel in 1959. Modern ML is driven by three pillars: data, algorithms, and compute.""",
        """There are three primary paradigms in machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each paradigm differs in the type of feedback the learning algorithm receives during training.""",
        """Supervised learning is the most commonly used paradigm. In supervised learning, the model is trained on a labelled dataset, where each input sample is paired with a corresponding output label. The goal is to learn a mapping function f(x) = y that generalises well to unseen data. Common supervised tasks include classification (predicting a category) and regression (predicting a continuous value).""",
        """Unsupervised learning involves training on unlabelled data. The algorithm must discover hidden structure or patterns on its own. Common techniques include clustering (e.g., K-Means), dimensionality reduction (e.g., PCA), and generative modelling (e.g., VAEs).""",
    ]),
    # Page 2
    ("Key Algorithms in Machine Learning", [
        """Linear Regression is one of the simplest and most widely used ML algorithms. It models the relationship between a dependent variable y and one or more independent variables x using a linear equation: y = wX + b, where w is the weight vector and b is the bias term. The model is trained by minimising the Mean Squared Error (MSE) loss using gradient descent.""",
        """Logistic Regression, despite its name, is a classification algorithm. It applies the sigmoid function to the linear combination of features to produce a probability output between 0 and 1. It is commonly used for binary classification tasks such as spam detection and medical diagnosis.""",
        """Decision Trees are hierarchical models that partition the feature space using a series of binary splits. Each internal node represents a feature test, each branch represents an outcome, and each leaf node holds a prediction. Decision trees are interpretable but prone to overfitting on complex datasets.""",
        """Random Forests are ensemble models that aggregate predictions from multiple decision trees trained on random subsets of the data (bagging). This reduces variance and improves generalisation. Random forests are robust to overfitting and handle missing values well.""",
        """Support Vector Machines (SVM) find the optimal hyperplane that maximally separates classes in feature space. The data points closest to the decision boundary are called support vectors. SVMs use the kernel trick to handle non-linearly separable data by implicitly mapping inputs to a higher-dimensional space.""",
    ]),
    # Page 3
    ("Neural Networks and Deep Learning", [
        """An Artificial Neural Network (ANN) is a computational model inspired by the structure of the human brain. It consists of layers of interconnected nodes (neurons). A standard feedforward neural network has an input layer, one or more hidden layers, and an output layer. Each neuron computes a weighted sum of its inputs followed by a non-linear activation function.""",
        """Common activation functions include ReLU (Rectified Linear Unit): f(x) = max(0, x), which is computationally efficient and avoids the vanishing gradient problem. The sigmoid function maps inputs to (0, 1) and is used in output layers for binary classification. The softmax function is used for multi-class classification and produces a probability distribution over classes.""",
        """Backpropagation is the algorithm used to train neural networks. It computes the gradient of the loss function with respect to each weight using the chain rule of calculus. These gradients are then used by an optimiser (e.g., SGD, Adam) to update the weights in the direction that reduces the loss.""",
        """Overfitting occurs when a model learns the training data too well, including noise and outliers, and fails to generalise to new data. Techniques to prevent overfitting include: L1 and L2 regularisation (adding penalty terms to the loss), Dropout (randomly deactivating neurons during training), and Early Stopping (halting training when validation loss stops improving).""",
    ]),
    # Page 4
    ("Evaluation Metrics and Model Selection", [
        """Evaluation metrics quantify how well a model performs on unseen data. For classification tasks, the most common metrics are: Accuracy (fraction of correct predictions), Precision (fraction of predicted positives that are truly positive), Recall (fraction of actual positives correctly identified), and F1-Score (harmonic mean of precision and recall).""",
        """The Confusion Matrix is a table that summarises the performance of a classification model. For binary classification, it shows: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). Precision = TP / (TP + FP) and Recall = TP / (TP + FN).""",
        """For regression tasks, common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). R-squared measures the proportion of variance in the target variable explained by the model.""",
        """Cross-validation is a technique for estimating model generalisation performance. In k-fold cross-validation, the dataset is split into k equal folds. The model is trained on k-1 folds and evaluated on the remaining fold, repeated k times. The final performance is averaged across all folds. This gives a more reliable estimate than a single train/test split.""",
        """Hyperparameter tuning is the process of finding the optimal hyperparameter configuration for a model. Common methods include Grid Search (exhaustive search over a predefined grid), Random Search (randomly sampling hyperparameter combinations), and Bayesian Optimisation (using a probabilistic model to guide the search).""",
    ]),
    # Page 5
    ("Natural Language Processing and Transformers", [
        """Natural Language Processing (NLP) is a branch of AI focused on enabling computers to understand, interpret, and generate human language. Key NLP tasks include text classification, named entity recognition, machine translation, question answering, and text summarisation.""",
        """Word embeddings represent words as dense vectors in a continuous vector space where semantically similar words are close together. Word2Vec (Mikolov et al., 2013) uses shallow neural networks to learn word representations from large text corpora. GloVe (Pennington et al., 2014) learns embeddings by factorising a word co-occurrence matrix.""",
        """The Transformer architecture (Vaswani et al., 2017) revolutionised NLP by replacing recurrent neural networks with a self-attention mechanism. Self-attention allows the model to weigh the importance of each word in the input with respect to every other word, capturing long-range dependencies efficiently.""",
        """BERT (Bidirectional Encoder Representations from Transformers) by Devlin et al. (2019) is a pre-trained Transformer model that uses masked language modelling to learn deep bidirectional representations. BERT can be fine-tuned on downstream tasks such as question answering and text classification with minimal architecture changes.""",
        """Retrieval-Augmented Generation (RAG) combines a dense retrieval component with a sequence-to-sequence generator. Given a query, the retriever fetches relevant passages from a document corpus. These passages are concatenated with the query and passed to the generator, which produces a grounded answer. RAG significantly reduces hallucination in LLM outputs by grounding responses in retrieved evidence. The approach was introduced by Lewis et al. in 2020 at NeurIPS.""",
    ]),
]

def make_pdf():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', parent=styles['Heading1'],
                                  fontSize=16, spaceAfter=12)
    body_style  = ParagraphStyle('Body2',  parent=styles['BodyText'],
                                  fontSize=11, leading=16, spaceAfter=8)
    story = []
    for heading, paragraphs in content:
        story.append(Paragraph(heading, title_style))
        story.append(Spacer(1, 0.3*cm))
        for para in paragraphs:
            story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 0.5*cm))
    doc.build(story)
    print(f"Sample PDF created: {OUTPUT}")

if __name__ == "__main__":
    make_pdf()
