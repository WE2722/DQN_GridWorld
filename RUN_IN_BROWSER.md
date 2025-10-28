# 🎮 Try DQN GridWorld in Your Browser!

Users can run this app directly from GitHub without installing anything!

## 🚀 Option 1: GitHub Codespaces (Recommended)

**One-click setup:**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/WE2722/DQN_GridWorld)

### Steps:
1. Click the badge above or go to the repository
2. Click the **"Code"** button → **"Codespaces"** tab
3. Click **"Create codespace on main"**
4. Wait for the environment to build (2-3 minutes)
5. In the terminal that opens, run:
   ```bash
   streamlit run app.py
   ```
6. Click the popup to open the app in your browser!

**Features:**
- ✅ Runs in your browser (no local installation)
- ✅ Full VS Code environment
- ✅ Free 60 hours/month for GitHub users
- ✅ All dependencies auto-installed

---

## 🌐 Option 2: Gitpod

**One-click setup:**

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/WE2722/DQN_GridWorld)

### Steps:
1. Click the badge above
2. Sign in with GitHub
3. Wait for workspace to load
4. Run in the terminal:
   ```bash
   streamlit run app.py
   ```
5. Open the forwarded port in your browser!

**Features:**
- ✅ Browser-based development
- ✅ 50 hours/month free
- ✅ Fast startup

---

## 💻 Option 3: Run Locally

If you prefer to run on your own machine:

```bash
# Clone the repository
git clone https://github.com/WE2722/DQN_GridWorld.git
cd DQN_GridWorld

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📋 What's Included

This project includes:
- **DQN Agent**: Vanilla, DeepMind, and Double DQN variants
- **GridWorld Environment**: Configurable goals, obstacles, and defenders
- **Training Visualization**: Real-time plots and grid rendering
- **Tests**: Comprehensive test suite (pytest)
- **CI/CD**: GitHub Actions workflow

---

## 🧪 Running Tests

In Codespaces/Gitpod terminal:

```bash
pytest tests/ -v
```

---

## 🎯 Quick Demo

Once the app is running:
1. Configure grid size and entities in the sidebar
2. Choose DQN variant (vanilla/deepmind/double)
3. Click "Start Training"
4. Watch real-time training visualization!

---

## ⚠️ Performance Notes

- Training can be CPU-intensive
- For browser-based environments (Codespaces/Gitpod):
  - Use smaller grid sizes (5-8)
  - Limit episodes (< 500)
  - These platforms have resource limits

---

## 📚 Documentation

- **README.md**: Project overview
- **tests/**: Test suite
- **.github/workflows/ci.yml**: CI/CD pipeline

---

## 🤝 Contributing

Feel free to open issues or submit pull requests!

---

**Repository**: https://github.com/WE2722/DQN_GridWorld
