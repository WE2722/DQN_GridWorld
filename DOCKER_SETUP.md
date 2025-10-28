# 🐳 Docker Setup Guide - DQN GridWorld

This guide shows you how to run the DQN GridWorld Visualizer using Docker on any desktop.

---

## 📋 Prerequisites

- **Docker Desktop** installed
- **Git** installed (to clone the repository)

---

## 🚀 Quick Start (5 Steps)

### **Step 1: Install Docker Desktop**

Download and install Docker Desktop:
- **Windows/Mac**: https://www.docker.com/products/docker-desktop
- **Linux**: https://docs.docker.com/engine/install/

After installation, verify Docker is working:
```powershell
docker --version
```

You should see something like: `Docker version 28.4.0, build d8eb465`

---

### **Step 2: Clone the Repository**

```powershell
# Navigate to your desired location
cd C:\Users\YourName\Desktop

# Clone the repository
git clone https://github.com/WE2722/DQN_GridWorld.git

# Enter the project directory
cd DQN_GridWorld
```

---

### **Step 3: Build the Docker Image**

```powershell
docker build -t dqn-gridworld .
```

**What this does:**
- Creates a Docker image with Python 3.10
- Installs all dependencies (Streamlit, PyTorch, NumPy, etc.)
- Packages your entire application

**Time:** 5-10 minutes (downloads ~500MB of dependencies)

**Expected output:**
```
[+] Building 339.6s (10/10) FINISHED
=> exporting to image
=> => naming to docker.io/library/dqn-gridworld:local
```

---

### **Step 4: Run the Container**

```powershell
docker run -d -p 8501:8501 --name dqn-app dqn-gridworld
```

**What this does:**
- `-d`: Runs container in background (detached mode)
- `-p 8501:8501`: Maps port 8501 (container) to 8501 (your computer)
- `--name dqn-app`: Names the container "dqn-app" for easy reference
- `dqn-gridworld`: Uses the image we built in Step 3

**Expected output:**
```
d78f1f3c6e70d9d19fda201b0233e3e3d416e4ca031f2f6204f8385266aec0f2
```
(This is your container ID)

---

### **Step 5: Open in Browser**

```powershell
# Windows PowerShell - auto-open browser
Start-Process "http://localhost:8501"

# Or manually open your browser to:
http://localhost:8501
```

🎉 **Your DQN GridWorld app is now running!**

---

## 🎮 Using Your App

Once the app opens in your browser, you can:

- **Configure Environment**: Set grid size, goals, obstacles, defenders
- **Train Agents**: Choose DQN variants (vanilla, deepmind, double)
- **Visualize Training**: Watch live training progress and metrics
- **Export Data**: Download training logs and Q-tables

---

## 🔧 Docker Management Commands

### **Check Container Status**
```powershell
# View all running containers
docker ps

# View specific container
docker ps --filter "name=dqn-app"
```

Expected output:
```
CONTAINER ID   IMAGE           COMMAND                  STATUS         PORTS
d78f1f3c6e70   dqn-gridworld   "streamlit run app.p…"   Up 5 minutes   0.0.0.0:8501->8501/tcp
```

---

### **View Container Logs**
```powershell
# View real-time logs
docker logs -f dqn-app

# View last 50 lines
docker logs --tail 50 dqn-app

# Stop following logs: Press Ctrl+C
```

---

### **Stop the Container**
```powershell
docker stop dqn-app
```

The container is stopped but not deleted. You can start it again later.

---

### **Start the Container Again**
```powershell
docker start dqn-app
```

This restarts the stopped container. Your app will be available again at http://localhost:8501

---

### **Restart the Container**
```powershell
docker restart dqn-app
```

Stops and immediately starts the container.

---

### **Remove the Container**
```powershell
# Stop and remove
docker rm -f dqn-app
```

Use this when you want to create a fresh container with new settings.

---

### **Remove the Image**
```powershell
# List all images
docker images

# Remove the image
docker rmi dqn-gridworld
```

Use this to rebuild from scratch (e.g., after updating the code).

---

## 🔄 Complete Workflow (Run Again Later)

When you want to run the app on another day:

```powershell
# 1. Navigate to project
cd C:\Users\YourName\Desktop\DQN_GridWorld

# 2. Start the container (if it exists)
docker start dqn-app

# 3. Open browser
Start-Process "http://localhost:8501"
```

If the container doesn't exist:
```powershell
# Build image (only if not built yet)
docker build -t dqn-gridworld .

# Run container
docker run -d -p 8501:8501 --name dqn-app dqn-gridworld

# Open browser
Start-Process "http://localhost:8501"
```

---

## 🛠️ Troubleshooting

### **Problem: "Docker daemon is not running"**

**Solution:**
1. Open Docker Desktop application
2. Wait for Docker to start (icon in system tray)
3. Try the command again

---

### **Problem: "Port 8501 is already in use"**

**Solution 1 - Use Different Port:**
```powershell
docker run -d -p 8502:8501 --name dqn-app dqn-gridworld
```
Then open: http://localhost:8502

**Solution 2 - Stop Existing Container:**
```powershell
docker stop dqn-app
docker rm dqn-app
docker run -d -p 8501:8501 --name dqn-app dqn-gridworld
```

---

### **Problem: "Container name 'dqn-app' already in use"**

**Solution:**
```powershell
# Remove old container
docker rm -f dqn-app

# Create new one
docker run -d -p 8501:8501 --name dqn-app dqn-gridworld
```

---

### **Problem: Build fails with "network timeout"**

**Cause:** Docker couldn't download PyTorch (large file)

**Solution:**
- Check your internet connection
- The Dockerfile uses CPU-only PyTorch (200MB) which is much smaller
- Retry the build:
  ```powershell
  docker build -t dqn-gridworld .
  ```

---

### **Problem: App doesn't load in browser**

**Solution:**
```powershell
# 1. Check if container is running
docker ps --filter "name=dqn-app"

# 2. Check container logs for errors
docker logs dqn-app

# 3. Restart container
docker restart dqn-app

# 4. Wait 10-20 seconds, then try: http://localhost:8501
```

---

### **Problem: Want to see what's inside the container**

**Solution:**
```powershell
# Open bash shell inside running container
docker exec -it dqn-app /bin/bash

# Now you can explore:
ls -la
python --version
pip list

# Exit shell:
exit
```

---

## 📦 Docker Image Details

The Docker image includes:
- **Base**: Python 3.10-slim (minimal Debian)
- **Size**: ~2.5GB (mostly PyTorch)
- **Dependencies**:
  - Streamlit 1.28.0+
  - PyTorch 2.0.0+ (CPU-only)
  - NumPy, Pandas, Matplotlib
  - Gym, Pytest
- **Port**: 8501 (Streamlit default)
- **Working Directory**: `/app`

---

## 🔐 Docker Image Optimization

Our Dockerfile uses several optimizations:

1. **CPU-only PyTorch**: Reduces image size by ~700MB
2. **Multi-stage layer caching**: Requirements installed before copying code
3. **.dockerignore**: Excludes tests, .git, .venv, etc.
4. **Build tools removal**: Removes gcc/g++ after installing dependencies
5. **--no-cache-dir**: Reduces pip cache size

---

## 🌐 Alternative: Run Without Docker

If you prefer not to use Docker, see [README.md](README.md) for Python virtual environment setup.

---

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Project Repository**: https://github.com/WE2722/DQN_GridWorld
- **Issues/Questions**: https://github.com/WE2722/DQN_GridWorld/issues

---

## 🎯 Summary - One-Line Commands

**First Time:**
```powershell
git clone https://github.com/WE2722/DQN_GridWorld.git && cd DQN_GridWorld && docker build -t dqn-gridworld . && docker run -d -p 8501:8501 --name dqn-app dqn-gridworld && Start-Process "http://localhost:8501"
```

**Run Again Later:**
```powershell
cd DQN_GridWorld && docker start dqn-app && Start-Process "http://localhost:8501"
```

**Clean Up Everything:**
```powershell
docker rm -f dqn-app && docker rmi dqn-gridworld
```

---

**Happy Training! 🤖🎮**
