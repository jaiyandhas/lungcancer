# Quick Start for Hackathon Presentation

## Option 1: Use Existing Visualizations (FASTEST)

If you already have Grad-CAM images in `results/gradcam_samples/`, you can use those directly!

1. Check what you have:
   ```bash
   ls -lh results/gradcam_samples/
   ```

2. Copy the best ones to a presentation folder:
   ```bash
   mkdir -p results/presentation
   cp results/gradcam_samples/*.png results/presentation/
   ```

## Option 2: Install Dependencies & Run Script

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation may take 5-10 minutes. Be patient!

### Step 3: Generate Demo Outputs
```bash
python generate_demo_outputs.py --num_samples 5
```

## Option 3: Use Streamlit Dashboard (If Already Installed)

```bash
source venv/bin/activate
streamlit run app/app.py
```

Then upload images through the web interface.

## For Your Presentation RIGHT NOW

### If dependencies aren't installed yet:

1. **Use existing visualizations** from `results/gradcam_samples/`
2. **Show the code structure** - point out:
   - `src/gradcam.py` - explainability implementation
   - `app/app.py` - interactive dashboard
   - `notebooks/` - complete ML pipeline
3. **Explain the improvements**:
   - "We improved Grad-CAM with smoother interpolation and better visualization"
   - "The system provides explainable AI for medical imaging"
   - "We have a complete pipeline from data to deployment"

### Quick Talking Points:

- **Problem**: "Lung cancer detection needs explainable AI"
- **Solution**: "Deep learning + Grad-CAM for transparency"
- **Impact**: "Doctors can verify AI reasoning"
- **Technical**: "ResNet + Transfer Learning + Grad-CAM"

### What to Show:

1. **Code Structure** (2 min)
   - Show the clean, modular codebase
   - Point out documentation and comments

2. **Visualizations** (2 min)
   - Show 2-3 Grad-CAM examples
   - Explain what red/yellow/blue means
   - Emphasize explainability

3. **Dashboard Demo** (1 min)
   - If it works, show live prediction
   - If not, show screenshots

4. **Impact** (1 min)
   - Medical applications need explainability
   - This builds trust in AI-assisted diagnosis

## Emergency Backup Plan

If nothing works, focus on:

1. **The Code Quality**: Show how well-structured and documented it is
2. **The Architecture**: Explain ResNet + Grad-CAM approach
3. **The Impact**: Emphasize explainability in medical AI
4. **The Completeness**: Show you have a full pipeline (EDA → Training → Deployment)

**Remember**: The jury cares about:
- ✅ Understanding the problem
- ✅ Technical approach
- ✅ Code quality
- ✅ Impact and explainability

You've got all of these! Good luck! 🚀
