# 🚀 KodeMapper IDPS - Quick Run Guide

Welcome! This is a simple, beginner-friendly guide to running the **Intrusion Detection & Prevention System (IDPS)**, specifically focusing on the new Live Telemetry test environment.

Instead of reading through heavy documentation, use this guide to get things running on your machine quickly.

---

## 🛠️ Prerequisites
Before starting, ensure you have the following installed:
1. **Node.js** (v18 or higher recommended) - [Download here](https://nodejs.org/)
2. **Python** (v3.10 to v3.12) - [Download here](https://www.python.org/downloads/)
3. **MongoDB** - Either local via MongoDB Compass or a cloud instance like MongoDB Atlas.

---

## ⚙️ Step 1: Clone & Configure

1. Open your terminal (or PowerShell) and navigate to the project directory:
   ```bash
   cd "Project Repo"
   ```

2. Create your `.env` configuration file in the backend. We've provided a template for you:
   ```bash
   cd service/api
   cp .env.example .env
   ```

3. Open the newly created `service/api/.env` file and update your MongoDB connection string. 
   - If running MongoDB locally, set it to:
     `MONGODB_URI=mongodb://127.0.0.1:27017`

---

## 🧠 Step 2: Start the Backend (API Server)

The backend connects to MongoDB, reads the test dataset, and runs the unified predictor (ML Baseline + DL Transformers + AE Zero-Day Canary) in parallel to detect anomalies.

1. Install the backend dependencies:
   ```bash
   npm install
   ```

2. **First Time Setup (Load Data):**
   If this is your first time, you need to load the dataset into MongoDB. Run:
   ```bash
   $env:RELOAD_CSV='true'    # Linux/Mac users use: export RELOAD_CSV='true'
   npm start
   ```

3. **Normal Run:**
   After the data is loaded (or if it's already there), kill the server (Ctrl+C) and run it normally so it doesn't overwrite your database every time:
   ```bash
   $env:RELOAD_CSV='false'   # Linux/Mac users use: export RELOAD_CSV='false'
   npm start
   ```

*(Keep this terminal window open!)*

---

## 💻 Step 3: Start the Frontend (Sentinel Dashboard)

Now that the engine is running, let's start the visual dashboard to see the alerts in real-time.

1. Open a **new** terminal window and navigate to the frontend directory:
   ```bash
   cd "Project Repo/service/dashboard"
   ```

2. Install the frontend dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

---

## 🎉 Step 4: View the Dashboard!

Open your browser and navigate to:
👉 **[http://localhost:5173/](http://localhost:5173/)**

You should see the "Sentinel" dashboard displaying live telemetry. Make sure the "Live Auto-Sync" toggle is turned **ON** in the bottom left corner to see alerts flow in!
