import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons
import numpy as np

def create_reverse_animation_from_gif():
    """Create reverse GIF from forward GIF"""
    from PIL import Image
    import os
    
    # Check if forward GIF exists
    if not os.path.exists('sample_evolution.gif'):
        print("Error: Forward GIF file 'sample_evolution.gif' not found")
        return
    
    # Read forward GIF
    print("Reading forward GIF...")
    gif = Image.open('sample_evolution.gif')
    
    # Extract all frames
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    print(f"Extracted {len(frames)} frames")
    
    # Create reverse frame sequence (excluding the pause at first frame)
    # Find the position of the first non-repeated frame
    first_unique_frame = 0
    for i in range(1, len(frames)):
        if frames[i] != frames[0]:
            first_unique_frame = i
            break
    
    # Reverse frame sequence: final pause + reverse playback + initial pause
    reverse_frames = []
    
    # 1. Final state pause (using the last frame)
    final_pause = 1
    reverse_frames.extend([frames[-1]] * final_pause)
    
    # 2. Reverse playback of middle frames
    middle_frames = frames[first_unique_frame:-1]  # Remove initial pause and final frame
    reverse_frames.extend(middle_frames[::-1])  # Reverse order
    
    # 3. Initial state pause (using the first frame)
    initial_pause = 20
    reverse_frames.extend([frames[0]] * initial_pause)
    
    # Save reverse GIF
    print("Saving reverse GIF...")
    # Resize frames to match the forward GIF size (600x600 pixels)
    target_size = (600, 600)
    resized_frames = []
    for frame in reverse_frames:
        resized_frame = frame.resize(target_size, Image.Resampling.LANCZOS)
        resized_frames.append(resized_frame)
    
    resized_frames[0].save(
        'sample_evolution_reverse.gif',
        save_all=True,
        append_images=resized_frames[1:],
        duration=50,  # 50ms per frame
        loop=0
    )
    print("Reverse GIF saved as 'sample_evolution_reverse.gif'")

def main():
    x = torch.Tensor(make_moons(n_samples=1000, noise=0.05)[0])
    x_initial = x.clone()  # Save initial state

    # Set animation parameters
    dt = torch.tensor(0.01)
    T = 8.0
    t = torch.arange(0, T, dt)
    t = torch.cat([t, torch.tensor([T])]) if torch.abs(t[-1] - T) > 1e-6 else t
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], s=10, alpha=0.7)
    # Use text object instead of set_title so it can be updated in animation
    title_text = ax.text(0.5, 0.95, 't = 0.00', transform=ax.transAxes, 
                        ha='center', va='top', fontsize=14, fontweight='bold')
    
    def animate(frame):
        nonlocal x
        with torch.no_grad():
            # If it's one of the first few repeated frames, don't update sample positions, just show initial state
            if frame < pause_frames:
                # Show initial state
                scatter.set_offsets(x_initial.numpy())
                title_text.set_text(f't = {t[0]:.2f}')
            else:
                # Calculate actual animation frame index
                actual_frame = frame - pause_frames
                # Update sample point positions
                x = x + dt * (-0.5 * x) + torch.randn_like(x) * torch.sqrt(dt)
                
                # Update scatter plot data
                scatter.set_offsets(x.numpy())
                
                # Update title to show current time
                current_time = t[actual_frame + 1] if actual_frame + 1 < len(t) else t[-1]
                title_text.set_text(f't = {current_time:.2f}')
            
        return scatter, title_text
    
    # Create animation
    # To reduce GIF size, we only use some frames
    total_frames = len(t) - 1
    step = max(1, total_frames // 100)  # Limit to about 100 frames
    frames_to_animate = list(range(0, total_frames, step))
    
    # Let the first frame pause longer, repeat the first frame several times
    pause_frames = 20  # Number of times to repeat the first frame
    extended_frames = [0] * pause_frames + frames_to_animate
    
    anim = animation.FuncAnimation(
        fig, animate, frames=extended_frames, 
        interval=50, blit=True, repeat=True
    )
    
    # Save forward GIF
    print("Generating forward GIF animation...")
    anim.save('sample_evolution.gif', writer='pillow', fps=20, dpi=100)
    print("Forward GIF saved as 'sample_evolution.gif'")
    
    # Create reverse animation
    print("Generating reverse GIF animation...")
    create_reverse_animation_from_gif()
    print("Reverse GIF saved as 'sample_evolution_reverse.gif'")
    
    # Show animation
    plt.show()

if __name__ == "__main__":
    main()