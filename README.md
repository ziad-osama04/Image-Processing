# FT Mixer & Emphasizer

A web-based interactive application for exploring the Fourier Transform (FT) of images, manipulating their frequency components (Magnitude, Phase, Real, Imaginary), and reconstructing the images.

## Features

- **Multi-Image Upload:** Load up to 4 images simultaneously into independent viewports.
- **Fourier Components:** Instantly view the Magnitude, Phase, Real, or Imaginary frequency spectrums of any loaded image.
- **Interactive UI:** Smooth, mouse-drag controls to adjust brightness and contrast of the viewports dynamically.
- **Image Resizing Policies:** Global policies (Smallest, Largest, Fixed) to normalize input sizes before FFT processing without altering original files.
- **Reconstruction:** Inverse Fourier Transform (IFFT) to reconstruct the spatial image from frequency components and verify round-trip fidelity.
- **Math-Driven Backend:** Dedicated separation of concerns keeps all math and FFT logic in the FastAPI backend, while the Vite + React frontend handles visualization.

## Tech Stack

- **Backend:** Python, FastAPI, NumPy, Pillow
- **Frontend:** TypeScript, React, Vite, Tailwind CSS

## Project Structure

- `/backend` - FastAPI server, Domain logic, Routers, and Pytest suites.
- `/frontend` - React application, API services, and UI components.

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup
1. Open a terminal and navigate to the project root.
2. Create and activate a virtual environment:
   ```bash
   python -m venv backend/venv
   # Windows (PowerShell):
   .\backend\venv\Scripts\Activate.ps1
   # Mac/Linux:
   source backend/venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The backend API will be running at `http://127.0.0.1:8000`.

### Frontend Setup
1. Open a **new terminal window** and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install the dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Access the application in your browser at `http://localhost:5173`.

## Architecture & Concepts
- **ImageModel:** Manages image loading, grayscale conversion, and maintains pure original data arrays (immutable logic).
- **FourierData:** Caches FFT spectrums, computes log-scaled magnitudes and phase, enabling fast switching without recomputation.
- **Session-Based:** Uses UUID sessions so multiple users can operate independently without overriding each other's data.
- **Viewport Independence:** The React frontend uses independent viewports for each image, avoiding global state lockups.

## Testing
To run the backend test suite, navigate to the `Image-Processing` root directory and run Pytest with the activated virtual environment:
```bash
python -m pytest backend/tests/ -v
```