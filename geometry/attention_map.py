import numpy as np
import matplotlib.pyplot as plt

def draw_attention_map(attention_scores):
    num_heads, seq_length, _ = attention_scores.shape

    # Create subplots for each attention head
    num_rows = num_heads // 4
    num_cols = 4

    # Create subplots for attention heads
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

    # Draw attention maps for each attention head
    for i, ax in enumerate(axes.flatten()):
        if i < num_heads:
            ax.imshow(attention_scores[i], cmap='hot', interpolation='nearest')
            ax.set_xlabel('Input Tokens')
            ax.set_ylabel('Output Tokens')
            ax.set_title(f'Attention Head {i+1}')
        else:
            ax.axis('off')  # Turn off the extra subplots if there are fewer attention heads

    plt.tight_layout()
    plt.savefig('attention_map.png')
    plt.close()


# Example usage
attention_scores = np.array([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]])

draw_attention_map(attention_scores)
