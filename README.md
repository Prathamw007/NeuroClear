# 🧠 NeuroClear: Agentic Medical Image Restoration

**NeuroClear** is a hybrid AI system designed to restore low-quality MRI scans using a custom-trained U-Net model orchestrated by an n8n Agentic Workflow.

## 🚀 The Stack
* **Orchestration:** n8n (Parallel Agentic Workflow)
* **AI Agent:** Llama 3.2 (running locally via Ollama)
* **Computer Vision:** Custom PyTorch U-Net model + OpenCV Post-Processing
* **Backend:** Python (Flask)

## 💡 How it Works
The system uses a **Parallel Execution Architecture** to handle user interaction and image processing simultaneously:
1.  **User uploads an MRI scan** via the n8n chat interface.
2.  **The Workflow splits:**
    * **Path A (The Agent):** Llama 3.2 engages the user, maintaining conversational context.
    * **Path B (The Engine):** The image is sent to a local Flask server.
3.  **Image Restoration:** The Flask server passes the image through a U-Net model (trained on MRI datasets) and applies CLAHE + Unsharp Masking for diagnostic clarity.

## 🛠️ Setup & Installation
1.  **Backend:**
    ```bash
    pip install -r requirements.txt
    python api.py
    ```
2.  **Workflow:**
    * Import `NeuroClear_Workflow.json` into n8n.
    * Ensure Ollama is running with Llama 3.2.

## 📸 Demo
*(Insert your Before/After images here)*

## 🚧 Challenges
Training the U-Net model locally pushed my hardware to the limit (nearly melted my GPU!), but managing the resource constraints taught me valuable lessons in efficient model checkpointing and inference optimization.
